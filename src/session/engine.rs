//! 驱动循环：`Machine` 的唯一持有者。
//!
//! ```text
//! loop {
//!     let cmd = rx.recv().await;            // ← 循环只在这里等
//!     let effects = machine.step(cmd);      // 纯函数
//!     for e in effects { execute(e) }       // Persist 同步 await；其余 spawn 或立即返回
//! }
//! ```
//!
//! **循环不得 await 任何调进用户代码的 future。** 模型请求、`begin_incremental`、每一次
//! `submit / seal / commit / abort`、`dispatch`、`compact` 全部 spawn 出去，完成后以
//! `Command` 回流。理由：运行时在 `abort()` 里调 `settled()`（要发命令并等回复）时，
//! 若循环正阻塞在 `abort()` 上，就是死锁。

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, oneshot};
use tokio::task::{AbortHandle, JoinHandle};

use crate::persistence::{CheckpointRevision, CheckpointStore, PersistenceError};
use crate::ports::{
    Compaction, IncrementalToolBatch, Model, ModelStreamEvent, ModelStreamSink, PortFuture,
    ToolCallReporter, ToolRuntime,
};
use crate::types::{
    AgentError, AgentErrorKind, RunId, ToolBatchAttemptId, ToolBatchId, ToolBatchStart,
    ToolCallBatch, ToolCallSlot, ToolResult, TranscriptItem,
};

use super::machine::{Command, Effect, Machine, Observation, RuntimeCall, RuntimeOp};

/// 循环之外还活着的东西：模型任务、批次所有者、整批 dispatch、压缩。
/// 计时器不记——到点发的命令若已过期，`step` 自己会忽略。
pub(super) struct Engine {
    pub(super) machine: Machine,
    store: Arc<dyn CheckpointStore>,
    revision: Option<CheckpointRevision>,
    model: Arc<dyn Model>,
    tools: Arc<dyn ToolRuntime>,
    compaction: Arc<dyn Compaction>,
    tx: mpsc::UnboundedSender<Command>,
    /// 观察端 drainer 的入口；在 `run` 里才 spawn（观察端可能在那之前才装上）。
    observe: Option<mpsc::UnboundedSender<Observation>>,
    model_tasks: HashMap<ToolBatchAttemptId, AbortHandle>,
    /// 每个增量批次一个所有者任务：指令通道 + 撤销句柄。`Release` 两个一起收。
    batch_owners: HashMap<ToolBatchAttemptId, (mpsc::UnboundedSender<RuntimeCall>, AbortHandle)>,
    dispatch_tasks: HashMap<ToolBatchId, AbortHandle>,
    compact_tasks: HashMap<RunId, AbortHandle>,
}

impl Engine {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        machine: Machine,
        store: Arc<dyn CheckpointStore>,
        revision: Option<CheckpointRevision>,
        model: Arc<dyn Model>,
        tools: Arc<dyn ToolRuntime>,
        compaction: Arc<dyn Compaction>,
        tx: mpsc::UnboundedSender<Command>,
    ) -> Self {
        Self {
            machine,
            store,
            revision,
            model,
            tools,
            compaction,
            tx,
            observe: None,
            model_tasks: HashMap::new(),
            batch_owners: HashMap::new(),
            dispatch_tasks: HashMap::new(),
            compact_tasks: HashMap::new(),
        }
    }

    pub(super) async fn run(mut self, mut rx: mpsc::UnboundedReceiver<Command>) {
        self.observe = Some(spawn_observer_drainer(&self.machine));
        let boot = self.machine.step(Command::Boot);
        self.execute(boot).await;
        while let Some(command) = rx.recv().await {
            match command {
                Command::Inspect(look) => {
                    // 只读查看不进 step，也不产生 effect。
                    look(&self.machine);
                }
                Command::Shutdown => {
                    // 外壳没了。循环自己还持着一个 `tx`（给 spawn 出去的任务回流用），
                    // 所以不会因为发送端全部消失而自然结束——得有人说一声。
                    // 已落盘的状态留在 store 里，重开时以它为准；在途任务全部撤销。
                    self.shutdown();
                    return;
                }
                command => {
                    let effects = self.machine.step(command);
                    self.execute(effects).await;
                }
            }
        }
    }

    /// 撤销所有还在外面跑的任务。它们的回流命令再也不会被处理。
    fn shutdown(&mut self) {
        for (_, task) in self.model_tasks.drain() {
            task.abort();
        }
        for (_, (_, task)) in self.batch_owners.drain() {
            task.abort();
        }
        for (_, task) in self.dispatch_tasks.drain() {
            task.abort();
        }
        for (_, task) in self.compact_tasks.drain() {
            task.abort();
        }
    }

    async fn execute(&mut self, effects: Vec<Effect>) {
        let mut queue: VecDeque<Effect> = effects.into();
        while let Some(effect) = queue.pop_front() {
            match effect {
                Effect::Persist => {
                    if let Some(fence_effects) = self.persist().await {
                        // 剩余 effect 依赖的状态没落盘：全部丢弃，包括 Reply（调用方的
                        // oneshot 关闭，包装层按隔离原因映射）。只执行隔离本身产生的
                        // 通知：运行等待方、空闲等待方、收回外部任务。
                        queue = fence_effects.into();
                    }
                }
                Effect::Reply(reply) => reply(),
                Effect::CallModel { attempt, request } => self.spawn_model(attempt, request),
                Effect::AbortModel { attempt } => {
                    if let Some(task) = self.model_tasks.remove(&attempt) {
                        task.abort();
                    }
                }
                Effect::BeginIncremental { attempt, start } => {
                    self.spawn_batch_owner(attempt, start)
                }
                Effect::Runtime { attempt, call } => {
                    if let Some((owner, _)) = self.batch_owners.get(&attempt) {
                        // 所有者已退出（commit / abort 之后）就丢弃；它退出前发的最后一条回执
                        // （Commit 的）会替它把话说完。
                        let _ = owner.send(call);
                    }
                }
                Effect::Release { attempt } => {
                    // 关通道不够：关闭的 receiver 仍会把已缓冲的指令排空，正在 await 的
                    // 用户 future 也不会停。连任务一起撤：Box 随之 drop，运行时在 Drop 里收尾。
                    if let Some((_, task)) = self.batch_owners.remove(&attempt) {
                        task.abort();
                    }
                }
                Effect::Dispatch { batch_id, batch } => self.spawn_dispatch(batch_id, batch),
                Effect::AbortDispatch { batch_id } => {
                    if let Some(task) = self.dispatch_tasks.remove(&batch_id) {
                        task.abort();
                    }
                }
                Effect::Compact {
                    run_id,
                    conversation,
                } => self.spawn_compact(run_id, conversation),
                Effect::AbortCompact { run_id } => {
                    if let Some(task) = self.compact_tasks.remove(&run_id) {
                        task.abort();
                    }
                }
                Effect::ArmTimer { attempt, after } => {
                    let tx = self.tx.clone();
                    tokio::spawn(async move {
                        tokio::time::sleep(after).await;
                        let _ = tx.send(Command::BatchTimedOut { attempt });
                    });
                }
                Effect::Emit(observation) => {
                    if let Some(observe) = &self.observe {
                        let _ = observe.send(observation);
                    }
                }
                Effect::Wake => {
                    let _ = self.tx.send(Command::Kick);
                }
            }
        }
    }

    /// 全量 checkpoint CAS。成功返回 `None`；失败即隔离，返回隔离产生的 effect
    /// （里面没有 `Persist`），调用方用它们替换掉本该继续执行的其余 effect。
    async fn persist(&mut self) -> Option<Vec<Effect>> {
        let checkpoint = self.machine.checkpoint();
        // store 是用户代码：panic 不许带走循环。它 panic 时我们不知道 CAS 有没有提交——
        // 按「已提交但确认丢失」隔离，与 Backend 错误同一条路。
        let outcome =
            match guarded_op(|| self.store.compare_and_swap(self.revision, checkpoint)).await {
                Ok(outcome) => outcome,
                Err(()) => Err(PersistenceError::Backend {
                    summary: "checkpoint_store_panicked".to_owned(),
                }),
            };
        match outcome {
            Ok(stored) => {
                self.revision = Some(stored.revision);
                None
            }
            Err(error) => {
                // Backend 错误可能是「未提交」也可能是「已提交但确认丢失」；不能从当前内存
                // 镜像盲目重试，必须隔离 writer 并重开权威 store。
                let kind = if matches!(&error, PersistenceError::Backend { .. }) {
                    AgentErrorKind::PersistenceCommitUnknown
                } else {
                    AgentErrorKind::Persistence
                };
                let effects = self.machine.step(Command::PersistFailed {
                    error: AgentError::new(kind, format!("checkpoint_save_failed:{error}")),
                });
                // 隔离之后不再落盘。debug 下这是断言；release 下也 fail-closed：
                // 把任何漏进来的 Persist 丢掉，其余通知照发。
                debug_assert!(
                    !effects
                        .iter()
                        .any(|effect| matches!(effect, Effect::Persist)),
                    "隔离不再落盘"
                );
                let effects: Vec<Effect> = effects
                    .into_iter()
                    .filter(|effect| !matches!(effect, Effect::Persist))
                    .collect();
                Some(effects)
            }
        }
    }

    // ── spawn 出去的端口调用 ─────────────────────────────────────

    fn spawn_model(&mut self, attempt: ToolBatchAttemptId, request: crate::ports::ModelRequest) {
        let model = Arc::clone(&self.model);
        let tx = self.tx.clone();
        let timeout = self.machine.core.config.model_timeout;
        let attempt_for_task = attempt.clone();
        let work = tokio::spawn(async move {
            let mut sink = CommandSink {
                tx: tx.clone(),
                attempt: attempt_for_task.clone(),
            };
            let result = with_timeout(
                timeout,
                "model_request_timeout",
                model.complete_stream(request, &mut sink),
            )
            .await;
            let command = match result {
                Ok(response) => Command::ModelDone {
                    attempt: attempt_for_task,
                    response,
                },
                Err(error) => Command::ModelFailed {
                    attempt: attempt_for_task,
                    error,
                },
            };
            let _ = tx.send(command);
        });
        // 适配器 panic 不能让运行悬在 CallingModel：转成一次失败的模型请求。
        let abort = supervise(work, self.tx.clone(), {
            let attempt = attempt.clone();
            move || Command::ModelFailed {
                attempt,
                error: AgentError::new(AgentErrorKind::Model, "model_adapter_panicked"),
            }
        });
        self.model_tasks.insert(attempt, abort);
    }

    fn spawn_batch_owner(&mut self, attempt: ToolBatchAttemptId, start: ToolBatchStart) {
        let (instr_tx, mut instr_rx) = mpsc::unbounded_channel::<RuntimeCall>();
        let tools = Arc::clone(&self.tools);
        let tx = self.tx.clone();
        let attempt_for_task = attempt.clone();
        let work = tokio::spawn(async move {
            let attempt = attempt_for_task;
            let ack = |op: RuntimeOp, result: Result<(), AgentError>| {
                let _ = tx.send(Command::RuntimeAck {
                    attempt: attempt.clone(),
                    op,
                    result,
                });
            };
            let reporter: Arc<dyn ToolCallReporter> = Arc::new(CommandReporter {
                tx: tx.clone(),
                attempt: attempt.clone(),
            });
            // 每一次调进运行时都单独兜住 panic：panic 就是那一步的 Err，走它的正常回执路径。
            let mut receiver: Box<dyn IncrementalToolBatch + '_> =
                match guarded_op(|| tools.begin_incremental(start, reporter)).await {
                    Ok(Ok(Some(receiver))) => {
                        ack(RuntimeOp::Begin(true), Ok(()));
                        receiver
                    }
                    Ok(Ok(None)) => {
                        ack(RuntimeOp::Begin(false), Ok(()));
                        return;
                    }
                    Ok(Err(error)) => {
                        ack(RuntimeOp::Begin(true), Err(error));
                        return;
                    }
                    Err(()) => {
                        ack(RuntimeOp::Begin(true), Err(runtime_panicked()));
                        return;
                    }
                };
            // 顺序执行，保住 `&mut self` 语义。
            // 状态机可能在一个 step 里连发 Submit/Seal/Commit；一旦有一步失败，后面
            // 排着的都不再交给运行时——它的回执会让状态机转去 Abort，那才是下一句话。
            let mut broken = false;
            while let Some(call) = instr_rx.recv().await {
                match call {
                    RuntimeCall::Submit(call) => {
                        if broken {
                            continue;
                        }
                        let result = guarded_op(|| receiver.submit(call))
                            .await
                            .unwrap_or_else(|()| Err(runtime_panicked()));
                        broken = result.is_err();
                        ack(RuntimeOp::Submit, result);
                    }
                    RuntimeCall::Seal(count) => {
                        if broken {
                            continue;
                        }
                        let result = guarded_op(|| receiver.calls_sealed(count))
                            .await
                            .unwrap_or_else(|()| Err(runtime_panicked()));
                        broken = result.is_err();
                        ack(RuntimeOp::Seal, result);
                    }
                    RuntimeCall::Commit => {
                        if broken {
                            continue;
                        }
                        // commit 消费 Box；此后结果只走 settled，内核不再对它说话。
                        let result = guarded_op(|| receiver.commit())
                            .await
                            .unwrap_or_else(|()| Err(runtime_panicked()));
                        ack(RuntimeOp::Commit, result);
                        return;
                    }
                    RuntimeCall::Abort(reason) => {
                        let result = guarded_op(|| receiver.abort(reason))
                            .await
                            .unwrap_or_else(|()| Err(runtime_panicked()));
                        let _ = tx.send(Command::BatchAborted {
                            attempt: attempt.clone(),
                            result,
                        });
                        return;
                    }
                }
            }
        });
        // 每一步用户代码都已各自兜住；这里只剩内核胶水自己 panic 的情形——所有者没了，
        // 状态机在任何阶段收到这条 Err 都会按「运行时不可用」收口。
        let handle = supervise(work, self.tx.clone(), {
            let attempt = attempt.clone();
            move || Command::BatchAborted {
                attempt,
                result: Err(AgentError::new(
                    AgentErrorKind::ToolDispatch,
                    "batch_owner_panicked",
                )),
            }
        });
        self.batch_owners.insert(attempt, (instr_tx, handle));
    }

    fn spawn_dispatch(&mut self, batch_id: ToolBatchId, batch: ToolCallBatch) {
        let tools = Arc::clone(&self.tools);
        let tx = self.tx.clone();
        let batch_for_task = batch_id.clone();
        let work = tokio::spawn(async move {
            let result = tools.dispatch(batch).await;
            let _ = tx.send(Command::BatchDispatched {
                batch_id: batch_for_task,
                result,
            });
        });
        let abort = supervise(work, self.tx.clone(), {
            let batch_id = batch_id.clone();
            move || Command::BatchDispatched {
                batch_id,
                result: Err(AgentError::new(
                    AgentErrorKind::ToolDispatch,
                    "tool_runtime_panicked",
                )),
            }
        });
        self.dispatch_tasks.insert(batch_id, abort);
    }

    fn spawn_compact(&mut self, run_id: RunId, conversation: Vec<TranscriptItem>) {
        let compaction = Arc::clone(&self.compaction);
        let tx = self.tx.clone();
        let timeout = self.machine.core.config.compaction_timeout;
        let run_for_task = run_id.clone();
        let work = tokio::spawn(async move {
            let result = with_timeout(
                timeout,
                "compaction_timeout",
                compaction.compact(&conversation),
            )
            .await;
            let _ = tx.send(Command::Compacted {
                run_id: run_for_task,
                result,
            });
        });
        let abort = supervise(work, self.tx.clone(), {
            let run_id = run_id.clone();
            move || Command::Compacted {
                run_id,
                result: Err(AgentError::new(
                    AgentErrorKind::InvalidState,
                    "compaction_panicked",
                )),
            }
        });
        self.compact_tasks.insert(run_id, abort);
    }
}

/// 看住一个 spawn 出去的任务：它 panic 了就把 `on_panic()` 喂回循环。被主动 abort 的
/// 任务不算 panic——那是内核自己不再等它了。返回的句柄只用于 abort。
fn supervise(
    work: JoinHandle<()>,
    tx: mpsc::UnboundedSender<Command>,
    on_panic: impl FnOnce() -> Command + Send + 'static,
) -> AbortHandle {
    let abort = work.abort_handle();
    tokio::spawn(async move {
        if let Err(join) = work.await {
            if join.is_panic() {
                let _ = tx.send(on_panic());
            }
        }
    });
    abort
}

fn runtime_panicked() -> AgentError {
    AgentError::new(AgentErrorKind::ToolDispatch, "tool_runtime_panicked")
}

/// 用户端口的一次调用 = 一个同步调用（构造 future，可能直接 panic）+ 若干次 poll。
/// 两段都兜住：任一段 panic 都是 `Err(())`。
async fn guarded_op<F: std::future::Future>(make: impl FnOnce() -> F) -> Result<F::Output, ()> {
    let future = std::panic::catch_unwind(std::panic::AssertUnwindSafe(make)).map_err(|_| ())?;
    caught(future).await
}

/// 把 future 的每一次 poll 包进 `catch_unwind`：用户代码 panic 变成 `Err(())`，
/// 而不是带走所有者任务。不引入 `futures` crate，就这几行。
fn caught<F: std::future::Future>(future: F) -> Caught<F> {
    Caught {
        inner: Box::pin(future),
        done: false,
    }
}

struct Caught<F: std::future::Future> {
    inner: std::pin::Pin<Box<F>>,
    done: bool,
}

impl<F: std::future::Future> std::future::Future for Caught<F> {
    type Output = Result<F::Output, ()>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = self.get_mut();
        assert!(!this.done, "polled after completion");
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            this.inner.as_mut().poll(cx)
        })) {
            Ok(std::task::Poll::Pending) => std::task::Poll::Pending,
            Ok(std::task::Poll::Ready(value)) => {
                this.done = true;
                std::task::Poll::Ready(Ok(value))
            }
            Err(_) => {
                this.done = true;
                std::task::Poll::Ready(Err(()))
            }
        }
    }
}

/// 给用户代码的 future 加超时；`None` 表示由适配器自己控制。
async fn with_timeout<T>(
    timeout: Option<Duration>,
    summary: &'static str,
    future: impl std::future::Future<Output = Result<T, AgentError>>,
) -> Result<T, AgentError> {
    match timeout {
        Some(limit) => match tokio::time::timeout(limit, future).await {
            Ok(result) => result,
            Err(_) => Err(AgentError::new(AgentErrorKind::Timeout, summary)),
        },
        None => future.await,
    }
}

// ── 通往循环的两个适配器 ─────────────────────────────────────────

/// 适配器拿到的 sink：`emit` = 发命令、等回复。回复 `Err` = 本次尝试已失效。
struct CommandSink {
    tx: mpsc::UnboundedSender<Command>,
    attempt: ToolBatchAttemptId,
}

impl ModelStreamSink for CommandSink {
    fn emit<'a>(&'a mut self, event: ModelStreamEvent) -> PortFuture<'a, Result<(), AgentError>> {
        let tx = self.tx.clone();
        let attempt = self.attempt.clone();
        Box::pin(async move {
            let (reply_tx, reply_rx) = oneshot::channel();
            tx.send(Command::ModelEvent {
                attempt,
                event,
                reply: reply_tx,
            })
            .map_err(|_| loop_closed())?;
            match reply_rx.await {
                Ok(result) => result,
                Err(_) => Err(closed_reply(&tx).await),
            }
        })
    }
}

/// 运行时拿到的上报端：只是一个 `Sender`，不持有会话——引用环由此消失。
struct CommandReporter {
    tx: mpsc::UnboundedSender<Command>,
    attempt: ToolBatchAttemptId,
}

impl ToolCallReporter for CommandReporter {
    fn settled<'a>(
        &'a self,
        slot: ToolCallSlot,
        result: ToolResult,
    ) -> PortFuture<'a, Result<(), AgentError>> {
        let tx = self.tx.clone();
        let attempt = self.attempt.clone();
        Box::pin(async move {
            let (reply_tx, reply_rx) = oneshot::channel();
            tx.send(Command::ToolSettled {
                attempt,
                slot,
                result,
                reply: reply_tx,
            })
            .map_err(|_| loop_closed())?;
            match reply_rx.await {
                Ok(result) => result,
                Err(_) => Err(closed_reply(&tx).await),
            }
        })
    }
}

fn loop_closed() -> AgentError {
    AgentError::new(AgentErrorKind::InvalidState, "session_loop_closed")
}

/// 回复通道关闭有两种可能：本条命令的 `Persist` 失败把 Reply 一起丢了（去问循环隔离原因），
/// 或者循环本身没了。与公共包装层同一套映射，端口实现看到的错误种类前后一致。
async fn closed_reply(tx: &mpsc::UnboundedSender<Command>) -> AgentError {
    let (reply, rx) = oneshot::channel();
    let asked = tx
        .send(Command::Inspect(Box::new(move |machine| {
            let _ = reply.send(machine.core.fence);
        })))
        .is_ok();
    let fence = if asked { rx.await.ok().flatten() } else { None };
    match fence {
        Some(kind) => AgentError::new(
            kind,
            "session_writer_is_fenced;reopen_from_checkpoint_store",
        ),
        None => loop_closed(),
    }
}

// ── 观察端 drainer ───────────────────────────────────────────────

/// 单个任务顺序回调三个观察端。序号顺序 = 队列顺序 = 观察顺序，构造性成立。
/// 慢观察端只拖慢自己；观察端回调里调 `session.enqueue()` 也不会死锁——循环是空闲的。
fn spawn_observer_drainer(machine: &Machine) -> mpsc::UnboundedSender<Observation> {
    let (tx, mut rx) = mpsc::unbounded_channel::<Observation>();
    let observer = machine.core.observers.observer.clone();
    let content = machine.core.observers.content.clone();
    let stream = machine.core.observers.stream.clone();
    tokio::spawn(async move {
        // 三个端口共用一个计数器，在这里、按到达顺序取号：序号顺序就是观察顺序。
        // `enabled` 和 `observe` 都是用户代码，都在这个任务上跑、都各自兜 panic：
        // `enabled` panic 当作「关掉」（不取号）；`observe` panic 只丢它自己那一条。
        let mut sequence = 0u64;
        while let Some(observation) = rx.recv().await {
            match observation {
                Observation::Agent(mut event) => {
                    let Some(observer) = observer.as_ref() else {
                        continue;
                    };
                    let metadata = event.metadata();
                    if !guarded(|| observer.enabled(metadata)).unwrap_or(false) {
                        continue;
                    }
                    sequence += 1;
                    event.set_sequence(sequence);
                    let _ = guarded(|| observer.observe(&event));
                }
                Observation::Content(mut event) => {
                    let Some(content) = content.as_ref() else {
                        continue;
                    };
                    let kind = event.kind();
                    if !guarded(|| content.enabled(kind)).unwrap_or(false) {
                        continue;
                    }
                    sequence += 1;
                    event.set_sequence(sequence);
                    let _ = guarded(|| content.observe(&event));
                }
                Observation::Stream(mut event) => {
                    let Some(stream) = stream.as_ref() else {
                        continue;
                    };
                    if !guarded(|| stream.enabled()).unwrap_or(false) {
                        continue;
                    }
                    sequence += 1;
                    event.sequence = sequence;
                    let _ = guarded(|| stream.observe(&event));
                }
            }
        }
    });
    tx
}

/// 用户回调的 panic 不许带走投递任务。
fn guarded<T>(call: impl FnOnce() -> T) -> Result<T, ()> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(call)).map_err(|_| ())
}
