//! 无输入输出操作的运行状态机。
//!
//! 它向驱动器明确提出两类操作要求：接收零个或多个对话记录项的请求边界，以及一个完整的
//! 工具批次。状态机自身不执行输入输出或调度。

use std::collections::HashSet;

use crate::ports::ModelResponse;
use crate::types::{
    order_tool_results, validate_closed_transcript, AgentError, AgentErrorKind, InputMessage,
    ModelOutput, ModelUsage, RunId, ToolBatchAttemptId, ToolBatchId, ToolCallBatch, ToolCallId,
    ToolResultBatch, TranscriptItem,
};

#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestBoundaryKind {
    /// 初始请求边界，或完整工具结果批次之后的边界。
    BeforeModelRequest,
    /// 模型未生成本地调用，运行本应在此结束。
    BeforeCompletion,
}

/// 要求驱动器执行的下一项操作。
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum TurnStep {
    RequestBoundary {
        kind: RequestBoundaryKind,
    },
    CallModel {
        transcript: Vec<TranscriptItem>,
    },
    DispatchTools {
        batch: ToolCallBatch,
    },
    Done {
        output: ModelOutput,
        usage: Option<ModelUsage>,
    },
}

#[derive(Clone)]
enum Boundary {
    ModelRequest,
    Completion(ModelOutput),
}

enum TurnState {
    NeedBoundary(Boundary),
    WaitingBoundary(Boundary),
    NeedModel,
    WaitingModel,
    NeedTools(ToolCallBatch),
    WaitingTools(Vec<ToolCallId>),
    Done(ModelOutput),
    Failed,
}

/// 一次服务商无关的运行。每个调用仅认领一次，完整结果批次会经过校验，所有对话记录变更
/// 都发生在明确的状态转换处。
pub struct Turn {
    run_id: RunId,
    transcript: Vec<TranscriptItem>,
    committable_len: usize,
    tool_batch_seq: u64,
    usage: Option<ModelUsage>,
    state: TurnState,
}

impl Turn {
    /// 构造一个已经通过闭合性校验的运行。
    ///
    /// 非法初始记录属于此便利构造器的契约违例并会 panic。需要处理不受信记录的调用方应
    /// 使用 [`Turn::try_new`]，以便显式接收校验错误。
    pub fn new(run_id: RunId, transcript: Vec<TranscriptItem>) -> Self {
        Self::try_new(run_id, transcript)
            .unwrap_or_else(|error| panic!("初始对话记录不闭合: {error}"))
    }

    /// 校验初始记录中的每个工具轮都已闭合，再构造运行。
    ///
    /// 校验在 `Turn` 存在前完成，因此非法记录既不会进入模型请求，也不会被
    /// [`Turn::committable_transcript`] 暴露为安全前缀。
    pub fn try_new(run_id: RunId, transcript: Vec<TranscriptItem>) -> Result<Self, AgentError> {
        validate_closed_transcript(&transcript)?;
        let committable_len = transcript.len();
        Ok(Self {
            run_id,
            transcript,
            committable_len,
            tool_batch_seq: 0,
            usage: None,
            state: TurnState::NeedBoundary(Boundary::ModelRequest),
        })
    }

    /// 从持久 checkpoint 的“下一次模型请求前”边界恢复。
    pub(crate) fn restore_before_model(
        run_id: RunId,
        transcript: Vec<TranscriptItem>,
        tool_batch_sequence: u64,
        usage: Option<ModelUsage>,
    ) -> Result<Self, AgentError> {
        let mut turn = Self::try_new(run_id, transcript)?;
        turn.tool_batch_seq = tool_batch_sequence;
        turn.usage = usage;
        Ok(turn)
    }

    /// 从模型已成功结束、但完成边界尚未封口的 checkpoint 恢复。
    pub(crate) fn restore_before_completion(
        run_id: RunId,
        transcript: Vec<TranscriptItem>,
        output: ModelOutput,
        tool_batch_sequence: u64,
        usage: Option<ModelUsage>,
    ) -> Result<Self, AgentError> {
        if !output.tool_calls.is_empty() {
            return Err(Self::invalid_state(
                "completion_checkpoint_contains_tool_calls",
            ));
        }
        let mut turn = Self::try_new(run_id, transcript)?;
        turn.tool_batch_seq = tool_batch_sequence;
        turn.usage = usage;
        turn.state = TurnState::NeedBoundary(Boundary::Completion(output));
        Ok(turn)
    }

    /// 从权威模型输出已经形成、工具结果尚未全部写回的 checkpoint 恢复。
    pub(crate) fn restore_tool_batch(
        run_id: RunId,
        mut transcript: Vec<TranscriptItem>,
        output: ModelOutput,
        batch_id: ToolBatchId,
        tool_batch_sequence: u64,
        usage: Option<ModelUsage>,
    ) -> Result<Self, AgentError> {
        validate_closed_transcript(&transcript)?;
        if output.tool_calls.is_empty() {
            return Err(Self::invalid_state("tool_checkpoint_has_no_calls"));
        }
        let mut turn = Self::try_new(run_id.clone(), transcript.clone())?;
        turn.claim_call_ids(&output)?;
        let batch = ToolCallBatch {
            run_id,
            batch_id,
            calls: output.tool_calls.clone(),
        };
        transcript.push(TranscriptItem::ModelOutput(output));
        turn.transcript = transcript;
        turn.tool_batch_seq = tool_batch_sequence;
        turn.usage = usage;
        turn.state = TurnState::NeedTools(batch);
        Ok(turn)
    }

    pub(crate) const fn tool_batch_sequence(&self) -> u64 {
        self.tool_batch_seq
    }

    /// 停在完成边界时，那份等着提交的最终输出。持久化投影用它写 `BeforeCompletion`。
    pub(crate) fn pending_completion_output(&self) -> Option<&ModelOutput> {
        match &self.state {
            TurnState::NeedBoundary(Boundary::Completion(output))
            | TurnState::WaitingBoundary(Boundary::Completion(output))
            | TurnState::Done(output) => Some(output),
            _ => None,
        }
    }

    pub(crate) fn usage(&self) -> Option<&ModelUsage> {
        self.usage.as_ref()
    }

    pub fn transcript(&self) -> &[TranscriptItem] {
        &self.transcript
    }

    /// 返回当前可安全提交的对话记录前缀。带工具调用的模型输出只有在完整结果批次被接受后
    /// 才会与结果一起进入此前缀。
    pub fn committable_transcript(&self) -> &[TranscriptItem] {
        &self.transcript[..self.committable_len]
    }

    /// 在请求边界用新的闭合对话替换受保护前缀之后的可提交记录。
    ///
    /// 这是驱动器执行轮间压缩的提交点。校验全部完成后才修改状态，因此非法压缩结果不会
    /// 污染原记录；运行状态、usage 和工具批序号都保持不变。
    pub(crate) fn replace_committable_suffix(
        &mut self,
        protected_prefix_len: usize,
        replacement: Vec<TranscriptItem>,
    ) -> Result<(), AgentError> {
        if !matches!(self.state, TurnState::WaitingBoundary(_)) {
            return Err(Self::invalid_state("turn_not_waiting_at_request_boundary"));
        }
        if self.committable_len != self.transcript.len() {
            return Err(Self::invalid_state(
                "request_boundary_has_uncommittable_transcript",
            ));
        }
        if protected_prefix_len > self.committable_len {
            return Err(Self::invalid_state("protected_prefix_out_of_bounds"));
        }

        validate_closed_transcript(&replacement)?;
        let mut candidate = self.transcript[..protected_prefix_len].to_vec();
        candidate.extend(replacement);
        validate_closed_transcript(&candidate)?;

        self.transcript = candidate;
        self.commit_transcript();
        Ok(())
    }

    pub fn next_step(&mut self) -> Result<TurnStep, AgentError> {
        let state = std::mem::replace(&mut self.state, TurnState::Failed);
        match state {
            TurnState::NeedBoundary(boundary) => {
                let kind = match boundary {
                    Boundary::ModelRequest => RequestBoundaryKind::BeforeModelRequest,
                    Boundary::Completion(_) => RequestBoundaryKind::BeforeCompletion,
                };
                self.state = TurnState::WaitingBoundary(boundary);
                Ok(TurnStep::RequestBoundary { kind })
            }
            TurnState::NeedModel => {
                self.state = TurnState::WaitingModel;
                Ok(TurnStep::CallModel {
                    transcript: self.transcript.clone(),
                })
            }
            TurnState::NeedTools(batch) => {
                let pending = batch.calls.iter().map(|call| call.id.clone()).collect();
                self.state = TurnState::WaitingTools(pending);
                Ok(TurnStep::DispatchTools { batch })
            }
            TurnState::Done(output) => {
                self.state = TurnState::Done(output.clone());
                Ok(TurnStep::Done {
                    output,
                    usage: self.usage.clone(),
                })
            }
            TurnState::WaitingBoundary(boundary) => {
                self.state = TurnState::WaitingBoundary(boundary);
                Err(Self::invalid_state("turn_waiting_at_request_boundary"))
            }
            TurnState::WaitingModel => {
                self.state = TurnState::WaitingModel;
                Err(Self::invalid_state("turn_waiting_for_model"))
            }
            TurnState::WaitingTools(pending) => {
                self.state = TurnState::WaitingTools(pending);
                Err(Self::invalid_state("turn_waiting_for_tool_batch"))
            }
            TurnState::Failed => Err(Self::invalid_state("turn_failed")),
        }
    }

    /// 恢复一个明确的请求边界。空向量具有明确含义：在完成边界提交最终响应；在其他边界
    /// 推进到模型调用。
    pub fn resume_boundary(&mut self, items: Vec<TranscriptItem>) -> Result<(), AgentError> {
        let state = std::mem::replace(&mut self.state, TurnState::Failed);
        let TurnState::WaitingBoundary(boundary) = state else {
            self.state = state;
            return Err(Self::invalid_state("turn_not_waiting_at_request_boundary"));
        };
        if let Err(error) = validate_closed_transcript(&items) {
            return self.fail(error);
        }

        match boundary {
            Boundary::ModelRequest => {
                self.transcript.extend(items);
                self.commit_transcript();
                self.state = TurnState::NeedModel;
            }
            Boundary::Completion(output) if items.is_empty() => {
                self.state = TurnState::Done(output);
            }
            Boundary::Completion(_) => {
                self.transcript.extend(items);
                self.commit_transcript();
                self.state = TurnState::NeedModel;
            }
        }
        Ok(())
    }

    /// 提交规范化响应，并准备其完整本地调用批次，或进入完成边界。此处不解析服务商 JSON。
    pub fn model_response(&mut self, response: ModelResponse) -> Result<(), AgentError> {
        self.model_response_inner(response, None)
    }

    /// 提交一个已经分配候选工具批次 ID 的规范化响应。
    ///
    /// 流式驱动器在模型请求开始时就分配此 ID，使提前转发的单项调用、中断报告和最终完整
    /// 批次始终使用同一个关联值。没有工具调用时该 ID 不会进入对话记录。
    pub fn model_response_for_attempt(
        &mut self,
        response: ModelResponse,
        batch_attempt_id: ToolBatchAttemptId,
    ) -> Result<(), AgentError> {
        self.model_response_inner(response, Some(batch_attempt_id))
    }

    fn model_response_inner(
        &mut self,
        response: ModelResponse,
        batch_attempt_id: Option<ToolBatchAttemptId>,
    ) -> Result<(), AgentError> {
        if !matches!(self.state, TurnState::WaitingModel) {
            return Err(Self::invalid_state("turn_not_waiting_for_model"));
        }

        if let Some(usage) = response.usage {
            if let Some(total) = &mut self.usage {
                total.merge(&usage);
            } else {
                self.usage = Some(usage);
            }
        }

        let output = response.output;
        if output.tool_calls.is_empty() {
            self.transcript
                .push(TranscriptItem::ModelOutput(output.clone()));
            self.commit_transcript();
            self.state = TurnState::NeedBoundary(Boundary::Completion(output));
            return Ok(());
        }

        if let Err(error) = self.claim_call_ids(&output) {
            return self.fail(error);
        }
        let Some(next_tool_batch_sequence) = self.tool_batch_seq.checked_add(1) else {
            return self.fail(AgentError::new(
                AgentErrorKind::BudgetExceeded,
                "tool_batch_sequence_exhausted",
            ));
        };
        self.tool_batch_seq = next_tool_batch_sequence;
        let batch_id = batch_attempt_id.map(ToolBatchId::from).unwrap_or_else(|| {
            ToolBatchId::new(format!(
                "{}/tools/{}",
                self.run_id.as_str(),
                self.tool_batch_seq
            ))
        });
        let batch = ToolCallBatch {
            run_id: self.run_id.clone(),
            batch_id,
            calls: output.tool_calls.clone(),
        };
        self.transcript.push(TranscriptItem::ModelOutput(output));
        self.state = TurnState::NeedTools(batch);
        Ok(())
    }

    /// 丢弃一次未完成的模型草稿，并把已经发生的外部事实作为一条正常输入提交。
    ///
    /// 此方法仅能在等待模型时调用。即使流中已经完整解析了若干单项工具调用，只要没有得到
    /// 整个模型响应的成功终态，就不存在可提交的权威 `ModelOutput`。调用方不得截取调用前缀
    /// 再配上部分或孤立的 `ToolResults`，因为模型原本可能继续生成更多调用。
    ///
    /// 恢复回执使用开放 role 的 [`InputMessage`] 表达：它保留运行时确认的已发生或结果
    /// 未知事实，但不把未完成草稿伪造成一个正常 Assistant/工具轮。
    pub fn recover_model_attempt(&mut self, receipt: InputMessage) -> Result<(), AgentError> {
        if !matches!(
            self.state,
            TurnState::WaitingModel | TurnState::NeedTools(_) | TurnState::WaitingTools(_)
        ) {
            return Err(Self::invalid_state("turn_not_waiting_for_model_recovery"));
        }

        // 工具阶段的中断：模型说完了，但这一批工具没有拿到完整结果。带调用的 `ModelOutput`
        // 只有配上完整的 `ToolResults` 才可提交，所以它此刻仍是草稿——一起丢掉，
        // 用回执交代已经发生的事实。
        self.transcript.truncate(self.committable_len);
        self.transcript.push(TranscriptItem::Input(receipt));
        self.commit_transcript();
        // 恢复事实先写入；随后仍经过正常请求边界，使已在信箱中的输入排在回执之后。
        self.state = TurnState::NeedBoundary(Boundary::ModelRequest);
        Ok(())
    }

    /// 为每个待处理调用提交且仅提交一个已完成结果。运行时返回顺序无关紧要；此处会恢复
    /// 确定的模型调用生成顺序。
    pub fn tool_results(&mut self, batch: ToolResultBatch) -> Result<(), AgentError> {
        let TurnState::WaitingTools(pending) = &self.state else {
            return Err(Self::invalid_state("turn_not_waiting_for_tool_batch"));
        };

        let ordered = match order_tool_results(pending, batch.results) {
            Ok(ordered) => ordered,
            Err(error) => return self.fail(error),
        };
        self.transcript
            .push(TranscriptItem::ToolResults(ToolResultBatch {
                results: ordered,
            }));
        self.commit_transcript();
        self.state = TurnState::NeedBoundary(Boundary::ModelRequest);
        Ok(())
    }

    fn commit_transcript(&mut self) {
        self.committable_len = self.transcript.len();
    }

    fn claim_call_ids(&mut self, output: &ModelOutput) -> Result<(), AgentError> {
        let mut current = HashSet::with_capacity(output.tool_calls.len());
        for call in &output.tool_calls {
            if call.id.as_str().is_empty() {
                return Err(AgentError::new(
                    AgentErrorKind::InvalidToolBatch,
                    "empty_tool_call_id",
                ));
            }
            if !current.insert(call.id.clone()) {
                return Err(AgentError::new(
                    AgentErrorKind::InvalidToolBatch,
                    "duplicate_tool_call_id",
                ));
            }
        }
        Ok(())
    }

    fn fail<T>(&mut self, error: AgentError) -> Result<T, AgentError> {
        self.state = TurnState::Failed;
        Err(error)
    }

    fn invalid_state(summary: &'static str) -> AgentError {
        AgentError::new(AgentErrorKind::InvalidState, summary)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::types::{InputMessage, ToolCall, ToolResult};

    fn response(output: ModelOutput) -> ModelResponse {
        ModelResponse {
            output,
            ..ModelResponse::default()
        }
    }

    #[test]
    fn state_machine_exposes_batch_and_request_boundaries() {
        let mut turn = Turn::new(RunId::new("run-1"), Vec::new());
        assert_eq!(
            turn.next_step().unwrap(),
            TurnStep::RequestBoundary {
                kind: RequestBoundaryKind::BeforeModelRequest
            }
        );

        // 注入内容不局限于用户角色。
        turn.resume_boundary(vec![InputMessage::text("developer", "constraint").into()])
            .unwrap();
        assert!(matches!(
            turn.next_step().unwrap(),
            TurnStep::CallModel { .. }
        ));

        turn.model_response(response(ModelOutput::calls(vec![
            ToolCall::new("a", "read", json!({"path": "a"})),
            ToolCall::new("b", "read", json!({"path": "b"})),
        ])))
        .unwrap();
        let TurnStep::DispatchTools { batch } = turn.next_step().unwrap() else {
            panic!("expected one complete tool batch");
        };
        assert_eq!(batch.calls.len(), 2);

        turn.tool_results(ToolResultBatch {
            results: vec![
                ToolResult::success_json(ToolCallId::new("b"), json!(2)),
                ToolResult::success_json(ToolCallId::new("a"), json!(1)),
            ],
        })
        .unwrap();
        assert_eq!(
            turn.next_step().unwrap(),
            TurnStep::RequestBoundary {
                kind: RequestBoundaryKind::BeforeModelRequest
            }
        );

        let TranscriptItem::ToolResults(results) = turn.transcript().last().unwrap() else {
            panic!("expected committed tool results");
        };
        assert_eq!(results.results[0].call_id.as_str(), "a");
        assert_eq!(results.results[1].call_id.as_str(), "b");
    }

    #[test]
    fn completion_boundary_can_steer_or_finish() {
        let mut turn = Turn::new(RunId::new("run-1"), Vec::new());
        turn.next_step().unwrap();
        turn.resume_boundary(Vec::new()).unwrap();
        turn.next_step().unwrap();
        turn.model_response(response(ModelOutput::text("candidate")))
            .unwrap();

        assert_eq!(
            turn.next_step().unwrap(),
            TurnStep::RequestBoundary {
                kind: RequestBoundaryKind::BeforeCompletion
            }
        );
        turn.resume_boundary(vec![InputMessage::text("operator", "revise").into()])
            .unwrap();
        assert!(matches!(
            turn.next_step().unwrap(),
            TurnStep::CallModel { .. }
        ));
    }

    #[test]
    fn request_boundary_can_replace_only_the_committable_suffix() {
        let base: TranscriptItem = InputMessage::text("system", "base").into();
        let old: TranscriptItem = InputMessage::text("user", "old history").into();
        let summary: TranscriptItem = InputMessage::text("user", "summary").into();
        let mut turn = Turn::new(RunId::new("run-compact"), vec![base.clone(), old.clone()]);

        assert!(turn
            .replace_committable_suffix(1, vec![summary.clone()])
            .is_err());
        assert_eq!(turn.transcript(), &[base.clone(), old.clone()]);

        turn.next_step().unwrap();
        assert!(turn
            .replace_committable_suffix(3, vec![summary.clone()])
            .is_err());
        assert_eq!(turn.transcript(), &[base.clone(), old]);
        turn.replace_committable_suffix(1, vec![summary.clone()])
            .unwrap();
        assert_eq!(turn.transcript(), &[base.clone(), summary.clone()]);

        let dangling = TranscriptItem::ModelOutput(ModelOutput::calls(vec![ToolCall::new(
            "dangling",
            "read",
            json!({}),
        )]));
        assert!(turn.replace_committable_suffix(1, vec![dangling]).is_err());
        assert_eq!(turn.transcript(), &[base, summary]);

        turn.resume_boundary(Vec::new()).unwrap();
        assert!(matches!(
            turn.next_step().unwrap(),
            TurnStep::CallModel { .. }
        ));
    }

    #[test]
    fn model_output_rejects_empty_and_duplicate_tool_call_ids() {
        let mut empty = waiting_model_turn();
        let error = empty
            .model_response(response(ModelOutput::calls(vec![ToolCall::new(
                "",
                "read",
                json!({}),
            )])))
            .unwrap_err();
        assert_eq!(error.summary, "empty_tool_call_id");

        let mut duplicate = waiting_model_turn();
        let error = duplicate
            .model_response(response(ModelOutput::calls(vec![
                ToolCall::new("same", "read", json!({})),
                ToolCall::new("same", "write", json!({})),
            ])))
            .unwrap_err();
        assert_eq!(error.summary, "duplicate_tool_call_id");
    }

    #[test]
    fn committable_transcript_waits_for_complete_tool_results() {
        let initial: TranscriptItem = InputMessage::text("user", "start").into();
        let mut turn = Turn::new(RunId::new("run-commit"), vec![initial]);
        assert_eq!(turn.committable_transcript().len(), 1);

        turn.next_step().unwrap();
        turn.resume_boundary(vec![InputMessage::text("developer", "constraint").into()])
            .unwrap();
        assert_eq!(turn.committable_transcript().len(), 2);
        turn.next_step().unwrap();

        turn.model_response(response(ModelOutput::calls(vec![ToolCall::new(
            "call-1",
            "read",
            json!({"path": "a"}),
        )])))
        .unwrap();
        assert_eq!(turn.transcript().len(), 3);
        assert_eq!(turn.committable_transcript().len(), 2);

        turn.next_step().unwrap();
        turn.tool_results(ToolResultBatch {
            results: vec![ToolResult::success_json(
                ToolCallId::new("call-1"),
                json!(1),
            )],
        })
        .unwrap();

        assert_eq!(turn.committable_transcript(), turn.transcript());
        assert!(matches!(
            &turn.committable_transcript()[2],
            TranscriptItem::ModelOutput(output) if output.tool_calls.len() == 1
        ));
        assert!(matches!(
            &turn.committable_transcript()[3],
            TranscriptItem::ToolResults(results) if results.results.len() == 1
        ));

        turn.next_step().unwrap();
        turn.resume_boundary(Vec::new()).unwrap();
        turn.next_step().unwrap();
        turn.model_response(response(ModelOutput::calls(vec![ToolCall::new(
            "call-2",
            "read",
            json!({"path": "b"}),
        )])))
        .unwrap();
        assert_eq!(turn.transcript().len(), 5);
        assert_eq!(turn.committable_transcript().len(), 4);

        turn.next_step().unwrap();
        let error = turn.tool_results(ToolResultBatch::default()).unwrap_err();
        assert_eq!(error.summary, "tool_result_count_mismatch");
        assert_eq!(turn.committable_transcript().len(), 4);
    }

    #[test]
    fn model_output_without_tool_calls_is_immediately_committable() {
        let mut turn = waiting_model_turn();
        turn.model_response(response(ModelOutput::text("done")))
            .unwrap();

        assert_eq!(turn.committable_transcript(), turn.transcript());
        assert_eq!(turn.committable_transcript().len(), 1);
    }

    #[test]
    fn try_new_rejects_unclosed_initial_transcript_before_a_turn_exists() {
        let error = match Turn::try_new(
            RunId::new("run-invalid-initial"),
            vec![TranscriptItem::ModelOutput(ModelOutput::calls(vec![
                ToolCall::new("dangling", "read", json!({"path": "a"})),
            ]))],
        ) {
            Ok(_) => panic!("未闭合的初始工具调用不应构造出 Turn"),
            Err(error) => error,
        };

        assert_eq!(error.kind, AgentErrorKind::InvalidToolBatch);
        assert_eq!(error.summary, "dangling_tool_calls");
    }

    #[test]
    #[should_panic(expected = "初始对话记录不闭合")]
    fn compatible_new_fails_closed_for_invalid_initial_transcript() {
        let _ = Turn::new(
            RunId::new("run-invalid-initial"),
            vec![TranscriptItem::ModelOutput(ModelOutput::calls(vec![
                ToolCall::new("dangling", "read", json!({"path": "a"})),
            ]))],
        );
    }

    fn waiting_model_turn() -> Turn {
        let mut turn = Turn::new(RunId::new("run-test"), Vec::new());
        turn.next_step().unwrap();
        turn.resume_boundary(Vec::new()).unwrap();
        turn.next_step().unwrap();
        turn
    }
}
