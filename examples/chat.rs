//! 一个能真跑的最小 agent：接真实服务商，带一个会拖时间的工具。
//!
//! 它存在的理由是把内核的两条论点变成看得见的东西：
//!
//! - **出向**：工具调用一旦参数完整就交给运行时，不等模型把话说完。终端上能看到
//!   「工具已接管」出现在模型还在输出的时候。
//! - **入向**：跑的过程中直接打字，内容进信箱，在下一个边界进入模型；会话空闲时
//!   打字则直接开一轮——同一个入口，两种效果。
//!
//! 跑法：
//!
//! ```text
//! AGENT_API_KEY=sk-...  cargo run --example chat --features openai
//! ```
//!
//! 可选：`AGENT_MODEL`（默认 deepseek-v4-flash）、`AGENT_ENDPOINT`
//! （默认 https://api.deepseek.com/chat/completions）。

use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::time::Duration;

use midturn::adapters::http::{HttpModel, HttpModelConfig, Protocol};
use midturn::{
    AbortClassification, AgentError, AgentErrorKind, AgentSession, Compaction, Enqueued,
    IncrementalToolBatch, IncrementalToolCall, InputMessage, MailboxInput, ModelStreamEvent,
    ModelStreamObservation, ObservedModelStreamEvent, PortFuture, SessionConfig, StreamObserver,
    ToolBatchAbortReason, ToolBatchStart, ToolCall, ToolCallBatch, ToolCallReporter, ToolCallSlot,
    ToolDefinition, ToolResult, ToolResultBatch, ToolRuntime, TranscriptItem,
};
use serde_json::json;

/// 本轮开始以来的毫秒数。示例用它证明「接管早于响应聚合完成」，而不是靠断言。
///
/// 必须由主循环在每轮开始时重置：否则首次调用才初始化，第一条时间戳恒为 0，
/// 提前量就被测没了。
static TURN_START: StdMutex<Option<std::time::Instant>> = StdMutex::new(None);

fn mark_turn_start() {
    *TURN_START.lock().unwrap() = Some(std::time::Instant::now());
}

fn elapsed_ms() -> u128 {
    TURN_START
        .lock()
        .unwrap()
        .map(|start| start.elapsed().as_millis())
        .unwrap_or(0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("AGENT_API_KEY")
        .map_err(|_| "请设置 AGENT_API_KEY；这个示例会真的调用服务商并产生费用")?;
    let model_name =
        std::env::var("AGENT_MODEL").unwrap_or_else(|_| "deepseek-v4-flash".to_owned());
    let endpoint = std::env::var("AGENT_ENDPOINT")
        .unwrap_or_else(|_| "https://api.deepseek.com/chat/completions".to_owned());

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async move { run(api_key, model_name, endpoint).await })
}

async fn run(
    api_key: String,
    model_name: String,
    endpoint: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let model = HttpModel::new(
        HttpModelConfig::new(
            endpoint,
            api_key,
            model_name.clone(),
            Protocol::openai_chat(),
        )
        .with_timeout(Duration::from_secs(120)),
    )?;

    let session = Arc::new(
        AgentSession::new(
            Arc::new(SlowClock),
            Arc::new(SlowClock),
            Arc::new(KeepEverything),
            Arc::new(model),
            SessionConfig::default(),
        )
        .with_stream_observer(Arc::new(Terminal)),
    );

    println!("模型：{model_name}");
    println!("有一个 sleep 工具，可以让它「等 5 秒再回答」——工具在跑的时候直接打字，");
    println!("内容会进信箱并在下一个边界进入模型。空行退出。\n");

    // 标准输入是阻塞的，放到专用线程里，主线程留给 agent。
    let (line_tx, mut line_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    std::thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let Ok(line) = line else { break };
            if line_tx.send(line).is_err() {
                break;
            }
        }
    });

    // 读输入和等这一轮必须并发：串行地 join 会把「运行期间插话」变成不可能——
    // 键盘上敲的字要等这一轮结束才被读到，于是永远只能开新的一轮。
    let (done_tx, mut done_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut running = false;

    loop {
        if !running {
            print!("你> ");
            io::stdout().flush()?;
        }

        tokio::select! {
            line = line_rx.recv() => {
                let Some(line) = line else { break };
                if line.trim().is_empty() {
                    break;
                }
                let items = vec![TranscriptItem::from(InputMessage::text("user", line))];
                match session.enqueue(MailboxInput::next_model_request(items)).await {
                    // 空闲时投递把会话叫醒；句柄交给一个任务去等，主循环继续读输入。
                    Ok(Enqueued::Started(handle)) => {
                        mark_turn_start();
                        running = true;
                        let done_tx = done_tx.clone();
                        tokio::spawn(async move {
                            let _ = done_tx.send(handle.join().await);
                        });
                    }
                    // 已经有运行在跑：内容会在它的下一个边界进入，我们什么都不用做。
                    Ok(Enqueued::Pending) => {
                        println!("\n\x1b[35m[已插入当前这一轮，将在下一个边界进入模型]\x1b[0m");
                    }
                    Ok(Enqueued::Held(reason)) => println!("[已收下，但暂时没人来取：{reason:?}]"),
                    Ok(other) => println!("[已收下：{other:?}]"),
                    Err(rejected) => println!("[投递被拒：{:?}]", rejected.reason),
                }
            }
            Some(outcome) = done_rx.recv() => {
                running = false;
                match outcome {
                    midturn::TurnOutcome::Completed { usage, .. } => {
                        println!("\n\x1b[90m[本轮结束] {usage:?}\x1b[0m\n");
                    }
                    other => println!("\n[本轮结束] {other:?}\n"),
                }
            }
        }
    }

    // 优雅退出：先关掉自动启动，再等安静。
    //
    // 少了这两步，退出时还在跑的那一轮会被连进程一起丢掉——工具可能正做到一半。
    // 顺序不能反：先等安静的话，中间到达的输入会再开一轮，永远等不完。
    session.set_auto_start(false).await;
    if running {
        println!("\x1b[90m[等待当前这一轮收尾……]\x1b[0m");
    }
    session.wait_until_idle().await;
    Ok(())
}

/// 把流式事件打到终端。推理用灰色前缀区分，正文直接输出。
struct Terminal;

impl StreamObserver for Terminal {
    fn observe(&self, event: &ObservedModelStreamEvent) {
        let payload = &event.payload;
        if matches!(payload, ModelStreamObservation::AttemptCommitted) {
            let mut out = io::stdout().lock();
            let _ = writeln!(
                out,
                "\x1b[33m[+{:>5}ms 模型响应聚合完成]\x1b[0m",
                elapsed_ms()
            );
            let _ = out.flush();
            return;
        }
        let ModelStreamObservation::Delta(delta) = payload else {
            return;
        };
        let mut out = io::stdout().lock();
        match delta {
            ModelStreamEvent::ReasoningStart { .. } => {
                let _ = write!(out, "\n\x1b[90m（想）");
            }
            ModelStreamEvent::ReasoningDelta { delta, .. } => {
                let _ = write!(out, "{delta}");
            }
            ModelStreamEvent::ReasoningEnd { .. } => {
                let _ = writeln!(out, "\x1b[0m");
            }
            ModelStreamEvent::TextStart { .. } => {
                let _ = write!(out, "\nagent> ");
            }
            ModelStreamEvent::TextDelta { delta, .. } => {
                let _ = write!(out, "{delta}");
            }
            ModelStreamEvent::TextEnd { .. } => {
                let _ = writeln!(out);
            }
            // 这一行是「出向及时性」的证据：它出现在模型还没说完的时候。
            ModelStreamEvent::ToolCallReady { call, .. } => {
                let _ = writeln!(out, "\n\x1b[36m[工具已接管] {}\x1b[0m", call.name.as_str());
            }
            _ => {}
        }
        let _ = out.flush();
    }
}

/// 一个故意慢的工具：让「工具在跑的时候还能插话」变成可以亲手试的事。
struct SlowClock;

impl midturn::PromptSource for SlowClock {
    fn base_context(&self) -> Result<Vec<TranscriptItem>, AgentError> {
        Ok(vec![InputMessage::text(
            "system",
            "你是一个简洁的中文助手。需要等待或计时时使用 sleep 工具。",
        )
        .into()])
    }
}

impl ToolRuntime for SlowClock {
    fn definitions(&self) -> Vec<ToolDefinition> {
        let mut sleep = ToolDefinition::new(
            "sleep",
            json!({
                "type": "object",
                "properties": {"seconds": {"type": "integer", "description": "等待秒数"}},
                "required": ["seconds"],
                "additionalProperties": false
            }),
        );
        sleep.description = Some("等待指定秒数后返回，用于演示长工具".to_owned());
        vec![sleep]
    }

    /// 整批分发是必需的回退路径：`begin_incremental` 按批返回 `None` 时走这里。
    /// 本例总是选择增量接管，所以它实际不会被用到，但契约要求它存在。
    fn dispatch<'a>(
        &'a self,
        batch: ToolCallBatch,
    ) -> PortFuture<'a, Result<ToolResultBatch, AgentError>> {
        Box::pin(async move {
            let mut results = Vec::with_capacity(batch.calls.len());
            for call in batch.calls {
                results.push(run_sleep(call).await);
            }
            Ok(ToolResultBatch { results })
        })
    }

    /// 选择在模型还在生成时就逐项接管并开始执行调用。
    ///
    /// 信号层（在流里解析出完整调用）业界都有；把**执行**也下放到这一刻的很少见
    /// ——已核实的九个独立设计里只有 OpenCode 这么做。选择加入这里，就必须回答
    /// 「流断了、已经交出去的怎么办」，那正是 `abort` 存在的理由。
    ///
    /// `reporter` 是结果唯一的通道：每个工具一跑完就上报，不等整批。
    /// 这也是「批内先完成的那一项」在断流时不被判成未知的原因。
    fn begin_incremental<'a>(
        &'a self,
        _start: ToolBatchStart,
        reporter: Arc<dyn ToolCallReporter>,
    ) -> PortFuture<'a, Result<Option<Box<dyn IncrementalToolBatch + 'a>>, AgentError>> {
        Box::pin(async move {
            Ok(Some(Box::new(EagerBatch {
                reporter,
                started: Vec::new(),
            }) as Box<dyn IncrementalToolBatch + 'a>))
        })
    }
}

/// 一到手就开跑的候选批次。
struct EagerBatch {
    reporter: Arc<dyn ToolCallReporter>,
    started: Vec<StartedCall>,
}

struct StartedCall {
    slot: ToolCallSlot,
    /// 任务体 = 执行 + 上报。它结束了就意味着内核已经收到结算。
    task: tokio::task::JoinHandle<()>,
}

impl IncrementalToolBatch for EagerBatch {
    /// 单个调用参数完整了。这里**立刻起任务执行**，不等模型说完；跑完就上报。
    fn submit<'a>(
        &'a mut self,
        call: IncrementalToolCall,
    ) -> PortFuture<'a, Result<(), AgentError>> {
        Box::pin(async move {
            println!(
                "\x1b[32m[+{:>5}ms 运行时已接管并开始执行] {} slot={}\x1b[0m",
                elapsed_ms(),
                call.call.name.as_str(),
                call.slot.get()
            );
            let slot = call.slot;
            let reporter = Arc::clone(&self.reporter);
            let task = tokio::spawn(async move {
                let result = run_sleep(call.call).await;
                // 内核可能已经终止了这一批（比如流断了、批次已 abort）：那时它会拒收，
                // 而拒收本身就是答案——这个结果没人要了。
                let _ = reporter.settled(slot, result).await;
            });
            self.started.push(StartedCall { slot, task });
            Ok(())
        })
    }

    /// 数组封口只表示「不会再有新调用」，不表示模型响应已经成功。
    fn calls_sealed<'a>(&'a mut self, call_count: u32) -> PortFuture<'a, Result<(), AgentError>> {
        Box::pin(async move {
            println!(
                "\x1b[32m[+{:>5}ms 工具数组封口] 共 {call_count} 个\x1b[0m",
                elapsed_ms()
            );
            Ok(())
        })
    }

    /// 模型响应成功了。任务早就在跑、跑完自己上报，这里没有别的事要做：
    /// 结果不从 commit 返回，从来不。
    fn commit<'a>(self: Box<Self>) -> PortFuture<'a, Result<(), AgentError>>
    where
        Self: 'a,
    {
        Box::pin(async move {
            println!(
                "\x1b[32m[+{:>5}ms 模型响应完整] {} 个任务在跑，结果各自上报\x1b[0m",
                elapsed_ms(),
                self.started.len()
            );
            Ok(())
        })
    }

    /// 模型响应没有成功终态：冻结已经接管的调用，如实分类。
    ///
    /// 这是提前执行的代价，也是别人不做这件事的原因：已经跑起来的东西不能假装
    /// 没发生。跑完的已经自己上报过了，不进报告；还在跑的**不能**报「确定没开始」
    /// ——我们已经起过任务了，无法证明它没产生影响。
    fn abort<'a>(
        self: Box<Self>,
        reason: ToolBatchAbortReason,
    ) -> PortFuture<'a, Result<AbortClassification, AgentError>>
    where
        Self: 'a,
    {
        Box::pin(async move {
            println!(
                "\x1b[31m[+{:>5}ms 候选批次终止] {reason:?}\x1b[0m",
                elapsed_ms()
            );
            let mut report = AbortClassification::new();
            for started in self.started {
                if started.task.is_finished() {
                    // 任务体包含上报：结束了就是内核已经收到了。
                    continue;
                }
                started.task.abort();
                report = report
                    .outcome_unknown(started.slot, "已经开始执行，中止时无法确认是否产生影响");
            }
            Ok(report)
        })
    }
}

async fn run_sleep(call: ToolCall) -> ToolResult {
    // 参数校验归实现方：内核只转发，不解释。
    let seconds = call
        .arguments
        .get("seconds")
        .and_then(|value| value.as_u64())
        .unwrap_or(1)
        .min(30);
    tokio::time::sleep(Duration::from_secs(seconds)).await;
    println!(
        "\x1b[32m[+{:>5}ms 工具执行完毕] sleep {seconds}s\x1b[0m",
        elapsed_ms()
    );
    ToolResult::success_json(call.id, json!({"slept_seconds": seconds}))
}

/// 不压缩：示例不会长到需要它。真实实现应当在这里概括历史。
struct KeepEverything;

impl Compaction for KeepEverything {
    fn compact<'a>(
        &'a self,
        conversation: &'a [TranscriptItem],
    ) -> PortFuture<'a, Result<Vec<TranscriptItem>, AgentError>> {
        Box::pin(async move {
            if conversation.is_empty() {
                return Err(AgentError::new(
                    AgentErrorKind::InvalidState,
                    "nothing_to_compact",
                ));
            }
            Ok(conversation.to_vec())
        })
    }
}
