//! 三种协议共享的 agent 场景、工具运行时和观测输出。

use std::collections::BTreeMap;
use std::error::Error;
use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use midturn::adapters::{
    anthropic::messages::RequestOptions as AnthropicRequestOptions,
    http::{HttpModel, HttpModelConfig, Protocol},
    openai::{
        chat::RequestOptions as OpenAiChatRequestOptions,
        responses::RequestOptions as OpenAiResponsesRequestOptions,
    },
};
use midturn::persistence::SessionId;
use midturn::{
    AbortClassification, AgentError, AgentErrorKind, AgentEvent, AgentSession, Compaction,
    Enqueued, FilteredObserver, IncrementalToolBatch, IncrementalToolCall, InputMessage,
    JsonObject, MailboxInput, Model, ModelRequest, ModelStreamEvent, ModelStreamSink, Observer,
    PortFuture, PromptSource, RunId, SessionConfig, ToolBatchAbortReason, ToolBatchStart,
    ToolCallBatch, ToolCallId, ToolCallReporter, ToolCallSlot, ToolDefinition, ToolResult,
    ToolResultBatch, ToolRuntime, TranscriptItem, TurnOutcome,
};
use serde_json::{json, Value};
use tokio::sync::{mpsc, Semaphore};
use tokio::time::timeout;

use crate::config::{ChatTokenField, SmokeConfig, SmokeProtocol};

type SmokeError = Box<dyn Error + Send + Sync>;
const SMOKE_MAX_OUTPUT_TOKENS: u64 = 256;
const SMOKE_MAX_RESPONSE_BYTES: usize = 1024 * 1024;

pub(crate) async fn run_smoke(
    protocol: SmokeProtocol,
    config: &SmokeConfig,
) -> Result<(), SmokeError> {
    println!("\n== {} ==", protocol.label());
    let (started_tx, mut started_rx) = mpsc::unbounded_channel();
    let gate = Arc::new(Semaphore::new(0));
    let observer = Arc::new(FilteredObserver::new(
        Arc::new(PrintingObserver { protocol }),
        config.level,
    ));
    let model = Arc::new(HttpModel::new(
        HttpModelConfig::new(
            config.endpoint(protocol),
            config.api_key.clone(),
            config.model.clone(),
            wire_protocol(protocol, config),
        )
        .with_timeout(config.timeout)
        .with_wire_log(config.wire_log)
        .with_max_response_bytes(SMOKE_MAX_RESPONSE_BYTES),
    )?);
    let model = Arc::new(PacedModel {
        inner: model,
        min_gap: config.min_request_gap,
        last_started: tokio::sync::Mutex::new(None),
    });
    run_text_probes(model.as_ref(), protocol, config.timeout).await?;
    let tools = Arc::new(SmokeTools {
        started: started_tx,
        gate: gate.clone(),
        incremental_begins: AtomicUsize::new(0),
        incremental_seals: AtomicUsize::new(0),
        incremental_commits: AtomicUsize::new(0),
        legacy_dispatches: AtomicUsize::new(0),
    });
    let mut session_config = SessionConfig::default();
    session_config.model_timeout = Some(config.timeout);
    session_config.tool_timeout = Some(config.timeout);
    // 冒烟测试给整个会话封口：这一趟只该发几次请求、调几次工具。
    session_config.budget.cumulative.max_model_requests = Some(4);
    session_config.budget.cumulative.max_tool_calls = Some(4);
    let session = Arc::new(
        AgentSession::new(
            Arc::new(SmokePrompt),
            tools.clone(),
            Arc::new(NoCompaction),
            model,
            session_config,
        )
        .with_observer(observer),
    );

    let running = session.clone();
    let mut task = tokio::spawn(async move {
        let started = running
            .enqueue(MailboxInput::next_model_request(vec![InputMessage::text(
                "user",
                "请在同一条回复中调用 add 两次，分别计算 2+3 与 5+7；现在不要自己计算。",
            )
            .into()]))
            .await?;
        let Enqueued::Started(handle) = started else {
            return Err(io::Error::other("空闲会话应当因这次投递开跑").into());
        };
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(handle.join().await)
    });

    let call_count = match timeout(config.timeout, started_rx.recv()).await {
        Ok(Some(count)) => count,
        Ok(None) => {
            task.abort();
            return Err(io::Error::other("工具 runtime 通知通道意外关闭").into());
        }
        Err(_) => {
            task.abort();
            return Err(io::Error::other(format!("等待 {} 工具调用超时", protocol.label())).into());
        }
    };
    if tools.incremental_seals.load(Ordering::SeqCst) != 1
        || tools.legacy_dispatches.load(Ordering::SeqCst) != 0
    {
        task.abort();
        return Err(io::Error::other("冒烟测试没有走增量工具 runtime 路径").into());
    }
    println!("增量 runtime 已接收并封口，调用数：{call_count}");

    session
        .enqueue(MailboxInput::next_model_request(vec![InputMessage::text(
            "user",
            "这是工具执行期间插入的消息：结果齐全后只简短报告两个和。",
        )
        .into()]))
        .await?;
    gate.add_permits(1);

    let outcome = match timeout(config.timeout, &mut task).await {
        Ok(result) => result??,
        Err(_) => {
            task.abort();
            return Err(
                io::Error::other(format!("等待 {} agent 完成超时", protocol.label())).into(),
            );
        }
    };
    if call_count != 2 {
        return Err(io::Error::other(format!(
            "期望一个含两个调用的批次，实际调用数：{call_count}"
        ))
        .into());
    }
    if tools.incremental_begins.load(Ordering::SeqCst) != 1
        || tools.incremental_commits.load(Ordering::SeqCst) != 1
        || tools.legacy_dispatches.load(Ordering::SeqCst) != 0
    {
        return Err(io::Error::other("增量工具 runtime 的 begin/commit 路径计数异常").into());
    }

    match outcome {
        TurnOutcome::Completed { output, usage, .. } => {
            println!("最终文本长度：{} 字节", output.text_content().len());
            println!("usage：{usage:?}");
            Ok(())
        }
        TurnOutcome::Stopped { .. } => Err(io::Error::other("冒烟测试被停止").into()),
        TurnOutcome::Failed { stage, error, .. } => Err(io::Error::other(format!(
            "冒烟测试失败：stage={stage:?}, kind={:?}",
            error.kind
        ))
        .into()),
        _ => Err(io::Error::other("冒烟测试返回了当前示例尚未识别的终态").into()),
    }
}

/// 显式保留旧冒烟适配器的请求参数，避免公共适配器的通用默认值改变测试强度。
fn wire_protocol(protocol: SmokeProtocol, config: &SmokeConfig) -> Protocol {
    match protocol {
        SmokeProtocol::OpenAiChat => Protocol::openai_chat_with(openai_chat_options(
            &config.chat_fields,
            config.chat_token_field,
        )),
        SmokeProtocol::OpenAiResponses => Protocol::openai_responses_with(
            OpenAiResponsesRequestOptions::new().with_max_output_tokens(SMOKE_MAX_OUTPUT_TOKENS),
        ),
        SmokeProtocol::AnthropicMessages => {
            Protocol::anthropic_messages_with(AnthropicRequestOptions::new(SMOKE_MAX_OUTPUT_TOKENS))
        }
    }
}

fn openai_chat_options(
    additional_fields: &JsonObject,
    token_field: ChatTokenField,
) -> OpenAiChatRequestOptions {
    let mut options = OpenAiChatRequestOptions::new();
    options.additional_fields = additional_fields.clone();
    match token_field {
        ChatTokenField::MaxTokens => options.with_max_tokens(SMOKE_MAX_OUTPUT_TOKENS),
        ChatTokenField::MaxCompletionTokens => {
            options.with_additional_field("max_completion_tokens", SMOKE_MAX_OUTPUT_TOKENS)
        }
    }
}

/// 应用层节流：两次请求之间至少隔 `min_gap`。限速是服务商与应用之间的事，
/// 内核只知道「一次模型请求」。
struct PacedModel {
    inner: Arc<HttpModel>,
    min_gap: std::time::Duration,
    last_started: tokio::sync::Mutex<Option<std::time::Instant>>,
}

impl PacedModel {
    async fn pace(&self) {
        if self.min_gap.is_zero() {
            return;
        }
        let mut last = self.last_started.lock().await;
        if let Some(started) = *last {
            let elapsed = started.elapsed();
            if elapsed < self.min_gap {
                tokio::time::sleep(self.min_gap - elapsed).await;
            }
        }
        *last = Some(std::time::Instant::now());
    }
}

impl Model for PacedModel {
    fn complete<'a>(
        &'a self,
        request: ModelRequest,
    ) -> PortFuture<'a, Result<midturn::ModelResponse, AgentError>> {
        Box::pin(async move {
            self.pace().await;
            self.inner.complete(request).await
        })
    }

    fn complete_stream<'a>(
        &'a self,
        request: ModelRequest,
        sink: &'a mut dyn ModelStreamSink,
    ) -> PortFuture<'a, Result<midturn::ModelResponse, AgentError>> {
        Box::pin(async move {
            self.pace().await;
            self.inner.complete_stream(request, sink).await
        })
    }
}

async fn run_text_probes(
    model: &PacedModel,
    protocol: SmokeProtocol,
    deadline: std::time::Duration,
) -> Result<(), SmokeError> {
    let session_id = SessionId::new(format!("protocol-smoke-{}", protocol.label()))?;
    let run_id = RunId::new(format!("{}/run/probe", session_id.as_str()));
    let request = |request_index| {
        ModelRequest::new(
            session_id.clone(),
            run_id.clone(),
            request_index,
            vec![
                InputMessage::text("user", "Reply with exactly SMOKE_OK. Do not call tools.")
                    .into(),
            ],
            Vec::new(),
        )
    };

    let non_stream = timeout(deadline, model.complete(request(1)))
        .await
        .map_err(|_| io::Error::other(format!("等待 {} non-stream text 超时", protocol.label())))?
        .map_err(|error| sanitized_probe_error(protocol, "non-stream", error))?;
    validate_text_probe("non-stream", &non_stream.output)?;

    let mut sink = TextProbeSink::default();
    let streamed = timeout(deadline, model.complete_stream(request(2), &mut sink))
        .await
        .map_err(|_| io::Error::other(format!("等待 {} stream text 超时", protocol.label())))?
        .map_err(|error| sanitized_probe_error(protocol, "stream", error))?;
    validate_text_probe("stream", &streamed.output)?;
    if sink.text_events == 0 || sink.saw_tool_event {
        return Err(io::Error::other(format!(
            "{} stream text 未观测到纯文本增量：source={}",
            protocol.label(),
            safe_text_source(&streamed.output)
        ))
        .into());
    }

    println!(
        "text 预检通过：non-stream + stream（{} 个增量事件）",
        sink.text_events
    );
    Ok(())
}

/// 真实凭据场景只公开失败阶段和种类，不传播可能由响应体派生的错误摘要。
fn sanitized_probe_error(protocol: SmokeProtocol, stage: &str, error: AgentError) -> io::Error {
    io::Error::other(format!(
        "{} {stage} text 失败：kind={:?}",
        protocol.label(),
        error.kind
    ))
}

/// 只报告文本来自哪个 wire 字段，不记录响应正文或任意字段值。
fn safe_text_source(output: &midturn::ModelOutput) -> &'static str {
    let Some(message) = output
        .provider_data
        .get("openai.chat.message")
        .and_then(Value::as_object)
    else {
        return "canonical-or-opaque";
    };
    if message
        .get("content")
        .and_then(Value::as_str)
        .is_some_and(|text| !text.is_empty())
    {
        "content"
    } else if message
        .get("refusal")
        .and_then(Value::as_str)
        .is_some_and(|text| !text.is_empty())
    {
        "refusal"
    } else {
        "extension"
    }
}

fn validate_text_probe(label: &str, output: &midturn::ModelOutput) -> Result<(), SmokeError> {
    if output.text_content().trim().is_empty() || !output.tool_calls.is_empty() {
        return Err(io::Error::other(format!("{label} text 预检未返回纯文本")).into());
    }
    Ok(())
}

#[derive(Default)]
struct TextProbeSink {
    text_events: usize,
    saw_tool_event: bool,
}

impl ModelStreamSink for TextProbeSink {
    fn emit<'a>(&'a mut self, event: ModelStreamEvent) -> PortFuture<'a, Result<(), AgentError>> {
        Box::pin(async move {
            match event {
                ModelStreamEvent::TextDelta { .. } => self.text_events += 1,
                ModelStreamEvent::ToolCallReady { .. } => self.saw_tool_event = true,
                ModelStreamEvent::ToolCallsSealed { call_count } if call_count > 0 => {
                    self.saw_tool_event = true;
                }
                _ => {}
            }
            Ok(())
        })
    }
}

struct SmokePrompt;

impl PromptSource for SmokePrompt {
    fn base_context(&self) -> Result<Vec<TranscriptItem>, AgentError> {
        Ok(vec![InputMessage::text(
            "system",
            "你在执行 agent 协议冒烟测试。必须严格按用户要求调用工具。",
        )
        .into()])
    }
}

struct SmokeTools {
    started: mpsc::UnboundedSender<usize>,
    gate: Arc<Semaphore>,
    incremental_begins: AtomicUsize,
    incremental_seals: AtomicUsize,
    incremental_commits: AtomicUsize,
    legacy_dispatches: AtomicUsize,
}

impl ToolRuntime for SmokeTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        let mut definition = ToolDefinition::new(
            "add",
            json!({
                "type": "object",
                "properties": {
                    "left": {"type": "integer"},
                    "right": {"type": "integer"}
                },
                "required": ["left", "right"],
                "additionalProperties": false
            }),
        );
        definition.description = Some("返回两个整数的和。".to_owned());
        vec![definition]
    }

    fn dispatch<'a>(
        &'a self,
        batch: ToolCallBatch,
    ) -> PortFuture<'a, Result<ToolResultBatch, AgentError>> {
        Box::pin(async move {
            self.legacy_dispatches.fetch_add(1, Ordering::SeqCst);
            self.started.send(batch.calls.len()).map_err(|_| {
                AgentError::new(AgentErrorKind::ToolDispatch, "smoke_receiver_closed")
            })?;
            let _permit =
                self.gate.acquire().await.map_err(|_| {
                    AgentError::new(AgentErrorKind::ToolDispatch, "smoke_gate_closed")
                })?;

            let results = batch
                .calls
                .iter()
                .map(|call| add_result(call.id.clone(), &call.arguments))
                .collect();
            Ok(ToolResultBatch { results })
        })
    }

    fn begin_incremental<'a>(
        &'a self,
        batch: ToolBatchStart,
        reporter: Arc<dyn ToolCallReporter>,
    ) -> PortFuture<'a, Result<Option<Box<dyn IncrementalToolBatch + 'a>>, AgentError>> {
        Box::pin(async move {
            self.incremental_begins.fetch_add(1, Ordering::SeqCst);
            Ok(Some(Box::new(SmokeIncrementalBatch {
                runtime: self,
                batch,
                reporter,
                calls: BTreeMap::new(),
                sealed_count: None,
            }) as Box<dyn IncrementalToolBatch + 'a>))
        })
    }
}

/// 示例 runtime 只接管并缓存规范化调用；真正计算要等模型响应成功后的 `commit`，
/// 结果逐槽通过 `reporter.settled` 上报——那是结果唯一的通道。
///
/// 因而这里的“增量”只降低调用信息的接管延迟，不提前产生工具副作用。如果 A 已经
/// `submit`、B 仍在模型流中传输时断流，`abort` 可以把 A 确定为
/// `CancelledBeforeStart`。需要在 `submit` 阶段提前执行的应用则必须在自己的
/// `abort` 实现中据实区分：已知结果的先 `settled`，剩下的分「确定没开始」和「不知道」。
struct SmokeIncrementalBatch<'a> {
    runtime: &'a SmokeTools,
    batch: ToolBatchStart,
    reporter: Arc<dyn ToolCallReporter>,
    calls: BTreeMap<ToolCallSlot, IncrementalToolCall>,
    sealed_count: Option<u32>,
}

impl IncrementalToolBatch for SmokeIncrementalBatch<'_> {
    fn submit<'a>(
        &'a mut self,
        call: IncrementalToolCall,
    ) -> PortFuture<'a, Result<(), AgentError>> {
        Box::pin(async move {
            // 这些条件由核心保证；违反意味着实现接线错误，不能降级成普通工具失败。
            assert_eq!(call.run_id, self.batch.run_id);
            assert_eq!(call.batch_attempt_id, self.batch.batch_attempt_id);
            assert!(self.sealed_count.is_none(), "封口后不能再提交工具调用");
            assert!(
                !self.calls.contains_key(&call.slot),
                "同一 slot 不能重复提交"
            );
            self.calls.insert(call.slot, call);
            Ok(())
        })
    }

    fn calls_sealed<'a>(&'a mut self, call_count: u32) -> PortFuture<'a, Result<(), AgentError>> {
        Box::pin(async move {
            if self.sealed_count.is_some() {
                return Err(AgentError::new(
                    AgentErrorKind::InvalidToolBatch,
                    "smoke_batch_already_sealed",
                ));
            }
            let actual_count = u32::try_from(self.calls.len()).map_err(|_| {
                AgentError::new(
                    AgentErrorKind::InvalidToolBatch,
                    "smoke_call_count_exceeds_u32",
                )
            })?;
            let contiguous =
                (0..call_count).all(|slot| self.calls.contains_key(&ToolCallSlot::new(slot)));
            if actual_count != call_count || !contiguous {
                return Err(AgentError::new(
                    AgentErrorKind::InvalidToolBatch,
                    "smoke_incremental_slots_not_contiguous",
                ));
            }

            self.sealed_count = Some(call_count);
            self.runtime
                .incremental_seals
                .fetch_add(1, Ordering::SeqCst);
            self.runtime
                .started
                .send(call_count as usize)
                .map_err(|_| {
                    AgentError::new(AgentErrorKind::ToolDispatch, "smoke_receiver_closed")
                })?;
            Ok(())
        })
    }

    fn commit<'a>(self: Box<Self>) -> PortFuture<'a, Result<(), AgentError>>
    where
        Self: 'a,
    {
        Box::pin(async move {
            let expected_count = self.sealed_count.ok_or_else(|| {
                AgentError::new(
                    AgentErrorKind::InvalidToolBatch,
                    "smoke_incremental_batch_not_sealed",
                )
            })?;
            if self.calls.len() != expected_count as usize {
                return Err(AgentError::new(
                    AgentErrorKind::InvalidToolBatch,
                    "smoke_incremental_call_count_changed",
                ));
            }
            let _permit =
                self.runtime.gate.acquire().await.map_err(|_| {
                    AgentError::new(AgentErrorKind::ToolDispatch, "smoke_gate_closed")
                })?;

            // 现在才执行；每算完一个就上报一个。上报顺序不重要——内核按槽位排。
            for item in self.calls.into_values() {
                let result = add_result(item.call.id, &item.call.arguments);
                self.reporter.settled(item.slot, result).await?;
            }
            self.runtime
                .incremental_commits
                .fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
    }

    fn abort<'a>(
        self: Box<Self>,
        _reason: ToolBatchAbortReason,
    ) -> PortFuture<'a, Result<AbortClassification, AgentError>>
    where
        Self: 'a,
    {
        Box::pin(async move {
            // 示例在 commit 获得 gate 前完全不执行，因此每个已 submit 项都能确定为未开始。
            // 这些取消结论不会被伪装成普通 ToolResults，也无需作为副作用事实进入恢复回执。
            let _ = &self.batch;
            Ok(self
                .calls
                .into_keys()
                .fold(AbortClassification::new(), |report, slot| {
                    report.cancelled_before_start(slot)
                }))
        })
    }
}

fn add_result(call_id: ToolCallId, arguments: &Value) -> ToolResult {
    let values = arguments
        .get("left")
        .and_then(Value::as_i64)
        .zip(arguments.get("right").and_then(Value::as_i64));
    match values {
        Some((left, right)) => ToolResult::success_json(call_id, json!({"sum": left + right})),
        None => ToolResult::failure(call_id, "left/right 必须是整数"),
    }
}

struct NoCompaction;

impl Compaction for NoCompaction {
    fn compact<'a>(
        &'a self,
        conversation: &'a [TranscriptItem],
    ) -> PortFuture<'a, Result<Vec<TranscriptItem>, AgentError>> {
        Box::pin(async move { Ok(conversation.to_vec()) })
    }
}

struct PrintingObserver {
    protocol: SmokeProtocol,
}

impl Observer for PrintingObserver {
    fn observe(&self, event: &AgentEvent) {
        eprintln!(
            "[{}][{:?}] {event:?}",
            self.protocol.label(),
            event.metadata().level
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use midturn::{AbortedSlot, ToolBatchAttemptId, ToolCall};

    #[test]
    fn chat_options_forward_extensions_and_keep_a_fixed_small_token_limit() {
        let mut fields = JsonObject::new();
        fields.insert("thinking".to_owned(), json!({"type": "disabled"}));

        let legacy = openai_chat_options(&fields, ChatTokenField::MaxTokens);
        assert_eq!(legacy.max_tokens, Some(SMOKE_MAX_OUTPUT_TOKENS));
        assert_eq!(legacy.additional_fields, fields);

        let completion = openai_chat_options(&fields, ChatTokenField::MaxCompletionTokens);
        assert_eq!(completion.max_tokens, None);
        assert_eq!(completion.additional_fields["thinking"]["type"], "disabled");
        assert_eq!(
            completion.additional_fields["max_completion_tokens"],
            SMOKE_MAX_OUTPUT_TOKENS
        );
    }

    #[tokio::test]
    async fn text_probe_does_not_treat_an_empty_tool_seal_as_a_tool_call() {
        let mut sink = TextProbeSink::default();
        sink.emit(ModelStreamEvent::TextDelta {
            part_index: 0,
            delta: "ok".to_owned(),
        })
        .await
        .unwrap();
        sink.emit(ModelStreamEvent::ToolCallsSealed { call_count: 0 })
            .await
            .unwrap();
        assert_eq!(sink.text_events, 1);
        assert!(!sink.saw_tool_event);
    }

    #[test]
    fn text_probe_error_does_not_copy_provider_summary() {
        let sensitive = "provider-body-must-not-escape";
        let error = sanitized_probe_error(
            SmokeProtocol::OpenAiChat,
            "stream",
            AgentError::new(AgentErrorKind::Model, sensitive),
        );
        assert!(!error.to_string().contains(sensitive));
        assert!(error.to_string().contains("kind=Model"));
    }

    fn runtime() -> (SmokeTools, mpsc::UnboundedReceiver<usize>, Arc<Semaphore>) {
        let (started, receiver) = mpsc::unbounded_channel();
        let gate = Arc::new(Semaphore::new(0));
        (
            SmokeTools {
                started,
                gate: gate.clone(),
                incremental_begins: AtomicUsize::new(0),
                incremental_seals: AtomicUsize::new(0),
                incremental_commits: AtomicUsize::new(0),
                legacy_dispatches: AtomicUsize::new(0),
            },
            receiver,
            gate,
        )
    }

    fn start() -> ToolBatchStart {
        ToolBatchStart {
            run_id: RunId::new("smoke-run"),
            batch_attempt_id: ToolBatchAttemptId::new("smoke-run/model/1/tools"),
        }
    }

    /// 这些用例直接驱动 runtime，不经过 session：上报端只是把结算收进一个列表。
    #[derive(Default)]
    struct RecordingReporter {
        settled: std::sync::Mutex<Vec<(ToolCallSlot, ToolResult)>>,
    }

    impl ToolCallReporter for RecordingReporter {
        fn settled<'a>(
            &'a self,
            slot: ToolCallSlot,
            result: ToolResult,
        ) -> PortFuture<'a, Result<(), AgentError>> {
            Box::pin(async move {
                self.settled.lock().unwrap().push((slot, result));
                Ok(())
            })
        }
    }

    fn incremental_call(slot: u32, id: &str, left: i64, right: i64) -> IncrementalToolCall {
        let start = start();
        IncrementalToolCall {
            run_id: start.run_id,
            batch_attempt_id: start.batch_attempt_id,
            slot: ToolCallSlot::new(slot),
            call: ToolCall::new(id, "add", json!({"left": left, "right": right})),
        }
    }

    #[tokio::test]
    async fn incremental_runtime_seals_and_reports_every_slot_on_commit() {
        let (runtime, mut started, gate) = runtime();
        let reporter = Arc::new(RecordingReporter::default());
        let mut batch = runtime
            .begin_incremental(start(), reporter.clone())
            .await
            .unwrap()
            .unwrap();

        // 完成事件可以乱序到达；commit 时每个槽都会被上报一次。
        batch
            .submit(incremental_call(1, "second", 5, 7))
            .await
            .unwrap();
        batch
            .submit(incremental_call(0, "first", 2, 3))
            .await
            .unwrap();
        batch.calls_sealed(2).await.unwrap();
        assert_eq!(started.recv().await, Some(2));
        assert_eq!(runtime.incremental_seals.load(Ordering::SeqCst), 1);
        assert_eq!(runtime.legacy_dispatches.load(Ordering::SeqCst), 0);

        gate.add_permits(1);
        batch.commit().await.unwrap();
        let settled = reporter.settled.lock().unwrap();
        assert_eq!(settled.len(), 2);
        assert!(settled
            .iter()
            .any(|(slot, result)| slot.get() == 0 && result.call_id.as_str() == "first"));
        assert!(settled
            .iter()
            .any(|(slot, result)| slot.get() == 1 && result.call_id.as_str() == "second"));
        drop(settled);
        assert_eq!(runtime.incremental_begins.load(Ordering::SeqCst), 1);
        assert_eq!(runtime.incremental_commits.load(Ordering::SeqCst), 1);
        assert_eq!(runtime.legacy_dispatches.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn incremental_runtime_abort_reports_every_call_as_not_started() {
        let (runtime, _started, _gate) = runtime();
        let reporter = Arc::new(RecordingReporter::default());
        let mut batch = runtime
            .begin_incremental(start(), reporter.clone())
            .await
            .unwrap()
            .unwrap();
        batch
            .submit(incremental_call(0, "first", 2, 3))
            .await
            .unwrap();
        batch
            .submit(incremental_call(1, "second", 5, 7))
            .await
            .unwrap();

        let report = batch
            .abort(ToolBatchAbortReason::ModelStreamInterrupted)
            .await
            .unwrap();
        assert_eq!(report.slots.len(), 2);
        assert!(report
            .slots
            .values()
            .all(|outcome| matches!(outcome, AbortedSlot::CancelledBeforeStart)));
        assert!(reporter.settled.lock().unwrap().is_empty());
        assert_eq!(runtime.incremental_commits.load(Ordering::SeqCst), 0);
        assert_eq!(runtime.legacy_dispatches.load(Ordering::SeqCst), 0);
    }
}
