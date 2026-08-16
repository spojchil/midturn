//! 会话：单写者状态机的公共外壳。
//!
//! `AgentSession` 本身不持有任何可变状态——它只是一个 `Sender<Command>`。全部状态在
//! 驱动循环任务（`engine`）的局部变量里，改它的只有纯函数 `step`（`machine`）。
//! 每个公共方法都是「发一条命令、等回复」。
//!
//! 循环在第一次调用任何异步方法时惰性 spawn：`AgentSession::new` 可以在 runtime 之外
//! 构造，但真正干活必须在 tokio 里。

mod config;
mod engine;
mod machine;

#[cfg(test)]
mod tests;

use std::sync::{Arc, Mutex as StdMutex};

use tokio::sync::{mpsc, oneshot, OnceCell};

pub use config::{Consumption, ContextBudget, CumulativeBudget, SessionBudget, SessionConfig};

use crate::events::{ContentObserver, Observer, RunStage, StreamObserver};
use crate::mailbox::{Delivery, Mailbox, MailboxInput, MailboxRejected, MailboxRejectedReason};
use crate::persistence::{
    CheckpointRevision, CheckpointStore, EffectId, EffectLedger, EffectOutcome,
    EffectRecoveryAction, InMemoryCheckpointStore, PersistenceError, SessionCheckpoint, SessionId,
};
use crate::ports::{Compaction, Model, PromptSource, ToolRuntime};
use crate::types::{AgentError, AgentErrorKind, ModelOutput, ModelUsage, RunId, TranscriptItem};

use engine::Engine;
use machine::{Command, Machine, StartOutcome};

/// 一次运行的终态。
///
/// 尚未投递的信箱内容不在这里返回：输入属于会话而不属于某一次运行，
/// 收尾时它们留在信箱里，等下一次运行的第一个边界排空。
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum TurnOutcome {
    Completed {
        run_id: RunId,
        output: ModelOutput,
        usage: Option<ModelUsage>,
    },
    Stopped {
        run_id: RunId,
        usage: Option<ModelUsage>,
    },
    Failed {
        run_id: RunId,
        stage: RunStage,
        usage: Option<ModelUsage>,
        error: AgentError,
    },
}

/// 一次投递之后会发生什么。
///
/// 输入总是已经收下并写进 checkpoint；这里描述的是「谁会来取它」。
#[non_exhaustive]
pub enum Enqueued {
    /// 这次投递把空闲的会话叫醒了，可以等这一轮的结果。
    Started(RunHandle),
    /// 已经有运行在跑，内容会在它的下一个边界进入。
    Pending,
    /// 收下了，但暂时不会有人来取。
    Held(HoldReason),
}

impl std::fmt::Debug for Enqueued {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Started(handle) => f.debug_tuple("Started").field(handle.run_id()).finish(),
            Self::Pending => f.write_str("Pending"),
            Self::Held(reason) => f.debug_tuple("Held").field(reason).finish(),
        }
    }
}

/// [`AgentSession::reschedule`] 之后谁来取这些内容。
///
/// 与 [`Enqueued`] 多一个 `Unchanged`：那一类当时就是空的，什么也没发生——
/// 这不是「有运行在跑会来取」，也不是「收下了没人取」，是根本没动。
#[non_exhaustive]
pub enum RescheduleOutcome {
    Unchanged,
    Started(RunHandle),
    Pending,
    Held(HoldReason),
}

impl std::fmt::Debug for RescheduleOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unchanged => f.write_str("Unchanged"),
            Self::Started(handle) => f.debug_tuple("Started").field(handle.run_id()).finish(),
            Self::Pending => f.write_str("Pending"),
            Self::Held(reason) => f.debug_tuple("Held").field(reason).finish(),
        }
    }
}

/// [`AgentSession::reschedule`] 的结果。
#[non_exhaustive]
#[derive(Debug)]
pub struct Rescheduled {
    /// 实际被改变交付语义的信封数。
    pub envelopes: usize,
    pub outcome: RescheduleOutcome,
}

/// 收下了输入却没有开跑的原因。
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HoldReason {
    /// 这条投递本来就不打算叫醒模型。
    Passive,
    /// 自动启动被关掉了；重新打开时会立刻兑现。
    AutoStartDisabled,
    /// 有未完成的运行等待 `try_resume`。
    ResumeRequired,
    /// 有结果未知的副作用等待对账。
    ReconciliationRequired,
    /// 持久化 writer 已被隔离，必须重开会话。
    Fenced,
    /// 运行序号已经用尽。
    RunIdExhausted,
}

/// 已经取得运行所有权的句柄。丢弃句柄只会停止等待，不会遗弃运行。
pub struct RunHandle {
    run_id: RunId,
    tx: mpsc::UnboundedSender<Command>,
    completion: oneshot::Receiver<TurnOutcome>,
}

impl RunHandle {
    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    /// 请求取消当前运行。模型 / 压缩任务被丢弃；已交付的工具由运行时冻结分类。
    pub fn cancel(&self) {
        let (reply, _) = oneshot::channel();
        let _ = self.tx.send(Command::Cancel { reply });
    }

    pub async fn join(self) -> TurnOutcome {
        match self.completion.await {
            Ok(outcome) => outcome,
            // 通道关闭有两种可能：收尾那次 `Persist` 失败把 Reply 一起丢了（去问循环
            // 隔离原因），或者循环本身没了。两种都给一个稳定的终态。
            Err(_) => {
                let (reply, rx) = oneshot::channel();
                let asked = self
                    .tx
                    .send(Command::Inspect(Box::new(move |machine| {
                        let _ = reply.send(machine.core.fence);
                    })))
                    .is_ok();
                let kind = if asked { rx.await.ok().flatten() } else { None };
                let error = match kind {
                    Some(kind) => AgentError::new(
                        kind,
                        "session_writer_is_fenced;reopen_from_checkpoint_store",
                    ),
                    None => AgentError::new(AgentErrorKind::InvalidState, "session_loop_closed"),
                };
                TurnOutcome::Failed {
                    run_id: self.run_id,
                    stage: RunStage::Boundary,
                    usage: None,
                    error,
                }
            }
        }
    }
}

/// 尚未 spawn 的驱动循环。`with_*observer` 在这上面改；第一次异步调用时移交给任务。
struct Pending {
    engine: Engine,
    rx: mpsc::UnboundedReceiver<Command>,
}

/// 会话的公共外壳。它自身没有状态：状态在驱动循环里。
///
/// **丢弃它就是关掉会话**：驱动循环收到 `Shutdown` 后撤销在途任务并退出；已落盘的状态留在
/// store 里，下次 `open` 以它为准（半截运行 → `try_resume`）。还握着 [`RunHandle`] 的调用方
/// 会得到 `session_loop_closed`。想让会话活着，就把 `Arc<AgentSession>` 拿在手里。
pub struct AgentSession {
    session_id: SessionId,
    tx: mpsc::UnboundedSender<Command>,
    pending: StdMutex<Option<Pending>>,
    started: OnceCell<()>,
}

impl AgentSession {
    /// 构造内存会话（不落盘）。
    pub fn new(
        prompt: Arc<dyn PromptSource>,
        tools: Arc<dyn ToolRuntime>,
        compaction: Arc<dyn Compaction>,
        model: Arc<dyn Model>,
        config: SessionConfig,
    ) -> Self {
        Self::new_with_session_id(
            ephemeral_session_id(),
            prompt,
            tools,
            compaction,
            model,
            config,
        )
    }

    /// 使用调用方提供的稳定会话命名空间构造内存会话。
    pub fn new_with_session_id(
        session_id: SessionId,
        prompt: Arc<dyn PromptSource>,
        tools: Arc<dyn ToolRuntime>,
        compaction: Arc<dyn Compaction>,
        model: Arc<dyn Model>,
        config: SessionConfig,
    ) -> Self {
        let machine = Machine::new(
            session_id.clone(),
            config,
            prompt,
            Arc::clone(&tools),
            Arc::new(Vec::new()),
            Mailbox::default(),
            EffectLedger::default(),
            Consumption::default(),
            0,
            None,
        );
        Self::assemble(
            session_id,
            machine,
            Arc::new(InMemoryCheckpointStore::new()),
            None,
            model,
            tools,
            compaction,
        )
    }

    /// 打开或创建一个由外部 store 持久化的 session。
    pub async fn open(
        session_id: SessionId,
        checkpoint_store: Arc<dyn CheckpointStore>,
        prompt: Arc<dyn PromptSource>,
        tools: Arc<dyn ToolRuntime>,
        compaction: Arc<dyn Compaction>,
        model: Arc<dyn Model>,
        config: SessionConfig,
    ) -> Result<Self, PersistenceError> {
        let stored = match checkpoint_store.load(&session_id).await? {
            Some(stored) => stored,
            None => {
                checkpoint_store
                    .compare_and_swap(None, SessionCheckpoint::empty(session_id.clone()))
                    .await?
            }
        };
        stored.checkpoint.validate()?;
        if stored.checkpoint.session_id != session_id {
            return Err(PersistenceError::InvalidCheckpoint {
                summary: "checkpoint_session_id_mismatch".to_owned(),
            });
        }
        let checkpoint = stored.checkpoint;
        let machine = Machine::new(
            session_id.clone(),
            config,
            prompt,
            Arc::clone(&tools),
            checkpoint.conversation,
            Mailbox::from_inputs(checkpoint.mailbox),
            checkpoint.effects,
            checkpoint.consumption,
            checkpoint.run_sequence,
            checkpoint.active_run,
        );
        Ok(Self::assemble(
            session_id,
            machine,
            checkpoint_store,
            Some(stored.revision),
            model,
            tools,
            compaction,
        ))
    }

    fn assemble(
        session_id: SessionId,
        machine: Machine,
        store: Arc<dyn CheckpointStore>,
        revision: Option<CheckpointRevision>,
        model: Arc<dyn Model>,
        tools: Arc<dyn ToolRuntime>,
        compaction: Arc<dyn Compaction>,
    ) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let engine = Engine::new(
            machine,
            store,
            revision,
            model,
            tools,
            compaction,
            tx.clone(),
        );
        Self {
            session_id,
            tx,
            pending: StdMutex::new(Some(Pending { engine, rx })),
            started: OnceCell::new(),
        }
    }

    pub fn session_id(&self) -> &SessionId {
        &self.session_id
    }

    /// 安装结构化事件接收器。只收不带正文的元数据。
    ///
    /// 三个 `with_*observer` 都必须在**第一次异步调用之前**（`enqueue`、`conversation`、
    /// `set_auto_start`……任何一个都算）：那一次调用会 spawn 驱动循环，之后再装会 panic。
    /// 它们消费 `self`，所以自然的写法就是构造 → 装观察端 → 包进 `Arc`。
    pub fn with_observer(self, observer: Arc<dyn Observer>) -> Self {
        self.with_engine(|engine| engine.machine.core.observers.observer = Some(observer));
        self
    }

    /// 安装内容事件接收器：完整对话、模型输出、工具结果，未经脱敏。
    /// 不装就收不到，也不会构造——安装动作本身就是选择加入。
    pub fn with_content_observer(self, observer: Arc<dyn ContentObserver>) -> Self {
        self.with_engine(|engine| engine.machine.core.observers.content = Some(observer));
        self
    }

    /// 安装可能接收正文和完整工具参数的高频流观察器。
    pub fn with_stream_observer(self, observer: Arc<dyn StreamObserver>) -> Self {
        self.with_engine(|engine| engine.machine.core.observers.stream = Some(observer));
        self
    }

    fn with_engine(&self, edit: impl FnOnce(&mut Engine)) {
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        match pending.as_mut() {
            Some(pending) => edit(&mut pending.engine),
            None => panic!("观察端必须在会话开始处理命令之前安装"),
        }
    }

    /// 第一次调用时把驱动循环 spawn 出去。之后每次只是取 `tx`。
    async fn ensure_started(&self) -> Result<(), MailboxRejectedReason> {
        if tokio::runtime::Handle::try_current().is_err() {
            return Err(MailboxRejectedReason::RuntimeUnavailable);
        }
        self.started
            .get_or_init(|| async {
                let pending = self
                    .pending
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .take();
                if let Some(Pending { engine, rx }) = pending {
                    // 循环在自己的任务里 spawn 观测投递任务；这里只 spawn 循环本身。
                    tokio::spawn(engine.run(rx));
                }
            })
            .await;
        Ok(())
    }

    // ── 投递 ─────────────────────────────────────────────────────

    /// 把输入交给会话。<b>任何时候都收</b>：拒绝理由只有记录段不闭合、容量超限、
    /// 持久化写入被隔离、以及当前线程不在 Tokio runtime 里四种（[`Self::enqueue_for`]
    /// 另有 `WrongRun`）。「正在跑」和「空着」都不是拒绝的理由——空闲不是拒绝的原因，
    /// 而是开跑的原因。
    pub async fn enqueue(
        self: &Arc<Self>,
        input: MailboxInput,
    ) -> Result<Enqueued, MailboxRejected> {
        self.deliver(input, true, None).await
    }

    /// 只向指定运行投递；目标运行已经结束时返回 `WrongRun`，不会静默落入下一轮。
    /// 它永远不会开启新的运行：这句话只对当前这一轮有意义。
    pub async fn enqueue_for(
        self: &Arc<Self>,
        run_id: &RunId,
        input: MailboxInput,
    ) -> Result<(), MailboxRejected> {
        self.deliver(input, false, Some(run_id.clone()))
            .await
            .map(|_| ())
    }

    async fn deliver(
        self: &Arc<Self>,
        input: MailboxInput,
        may_start: bool,
        expected_run: Option<RunId>,
    ) -> Result<Enqueued, MailboxRejected> {
        if let Err(reason) = self.ensure_started().await {
            return Err(MailboxRejected { reason, input });
        }
        let (reply, rx) = oneshot::channel();
        let fallback = input.clone();
        if self
            .tx
            .send(Command::Enqueue {
                input,
                may_start,
                expected_run,
                reply,
            })
            .is_err()
        {
            return Err(MailboxRejected {
                reason: MailboxRejectedReason::SessionClosed,
                input: fallback,
            });
        }
        match rx.await {
            Ok(Ok(outcome)) => Ok(self.wrap(outcome)),
            Ok(Err(rejected)) => Err(rejected),
            // Persist 失败会丢弃 Reply：这就是那条 Fenced。原因去问循环；问不到就是循环没了。
            Err(_) => Err(MailboxRejected {
                reason: match self.persistence_fence_reason().await {
                    Some(AgentErrorKind::PersistenceCommitUnknown) => {
                        MailboxRejectedReason::PersistenceCommitUnknown
                    }
                    Some(_) => MailboxRejectedReason::Persistence,
                    None => MailboxRejectedReason::SessionClosed,
                },
                input: fallback,
            }),
        }
    }

    /// 改变**还在排队**的内容的交付语义。
    ///
    /// 人是会改主意的：一句话按 [`Delivery::NextModelRequest`] 发出去之后，用户可能立刻想
    /// 「先别急，等它跑完再说」。内容已经收下了，重发一遍模型会看到两次，撤回又会丢掉它
    /// ——真正想要的是把它挪到另一个边界。只影响调用这一刻仍在队列里的信封。
    ///
    /// 不逐条改：信箱的概念是**投递边界**，不是待办清单。
    pub async fn reschedule(
        self: &Arc<Self>,
        from: Delivery,
        to: Delivery,
    ) -> Result<Rescheduled, AgentError> {
        self.ensure_started().await.map_err(runtime_error)?;
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Command::Reschedule { from, to, reply })
            .map_err(|_| loop_closed())?;
        let reply = match rx.await {
            Ok(reply) => reply?,
            Err(_) => return Err(self.fenced().await),
        };
        let outcome = if reply.envelopes == 0 {
            RescheduleOutcome::Unchanged
        } else {
            match self.wrap(reply.outcome) {
                Enqueued::Started(handle) => RescheduleOutcome::Started(handle),
                Enqueued::Pending => RescheduleOutcome::Pending,
                Enqueued::Held(reason) => RescheduleOutcome::Held(reason),
            }
        };
        Ok(Rescheduled {
            envelopes: reply.envelopes,
            outcome,
        })
    }

    /// `Persist` 失败会丢弃 `Reply`，调用方看到的就是通道关闭；原因去问循环。
    async fn fenced(&self) -> AgentError {
        match self.persistence_fence_reason().await {
            Some(kind) => AgentError::new(
                kind,
                "session_writer_is_fenced;reopen_from_checkpoint_store",
            ),
            None => loop_closed(),
        }
    }

    fn wrap(&self, outcome: StartOutcome) -> Enqueued {
        match outcome {
            StartOutcome::Started { run_id, completion } => Enqueued::Started(RunHandle {
                run_id,
                tx: self.tx.clone(),
                completion,
            }),
            StartOutcome::Pending => Enqueued::Pending,
            StartOutcome::Held(reason) => Enqueued::Held(reason),
        }
    }

    // ── 生命周期 ─────────────────────────────────────────────────

    /// 打开或关闭「空闲时信箱有触发内容就自动开跑」。
    ///
    /// 纯运行时开关，不落盘。关掉它不拒收任何输入，也不打断进行中的运行——那是
    /// [`Self::cancel_run`] 的职责。重新打开时若信箱里已经攒了内容，立刻兑现。
    pub async fn set_auto_start(self: &Arc<Self>, enabled: bool) {
        if self.ensure_started().await.is_err() {
            return;
        }
        let _ = self.tx.send(Command::SetAutoStart(enabled));
    }

    /// 等到没有运行在跑。不阻止新的开始——想要真的安静，先关自动启动。
    pub async fn wait_until_idle(&self) {
        if self.ensure_started().await.is_err() {
            return;
        }
        let (reply, rx) = oneshot::channel();
        if self.tx.send(Command::WaitIdle { reply }).is_err() {
            return;
        }
        let _ = rx.await;
    }

    /// 取消当前运行。返回被取消的运行 ID；没有运行在跑就是 `None`。
    pub async fn cancel_run(&self) -> Option<RunId> {
        self.ensure_started().await.ok()?;
        let (reply, rx) = oneshot::channel();
        self.tx.send(Command::Cancel { reply }).ok()?;
        rx.await.ok().flatten()
    }

    /// 显式续跑重开后留在盘上的半截运行。`Ok(None)` = 没有这样的运行。
    /// 它是显式命令，不看 `auto_start`：用户点名要续的运行就续。
    pub async fn try_resume(self: &Arc<Self>) -> Result<Option<RunHandle>, AgentError> {
        self.ensure_started().await.map_err(runtime_error)?;
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Command::Resume { reply })
            .map_err(|_| loop_closed())?;
        let outcome = match rx.await {
            Ok(outcome) => outcome?,
            Err(_) => return Err(self.fenced().await),
        };
        Ok(outcome.and_then(|outcome| match self.wrap(outcome) {
            Enqueued::Started(handle) => Some(handle),
            _ => None,
        }))
    }

    // ── 查看 ─────────────────────────────────────────────────────

    async fn inspect<T: Send + 'static>(
        &self,
        look: impl FnOnce(&Machine) -> T + Send + 'static,
    ) -> Option<T> {
        self.ensure_started().await.ok()?;
        let (reply, rx) = oneshot::channel::<T>();
        self.tx
            .send(Command::Inspect(Box::new(move |machine| {
                let _ = reply.send(look(machine));
            })))
            .ok()?;
        rx.await.ok()
    }

    pub async fn conversation(&self) -> Vec<TranscriptItem> {
        self.inspect(|machine| machine.core.conversation.as_ref().clone())
            .await
            .unwrap_or_default()
    }

    /// 返回当前 writer 被持久化层隔离的原因；出现值后必须丢弃本实例并重新 `open`。
    pub async fn persistence_fence_reason(&self) -> Option<AgentErrorKind> {
        self.inspect(|machine| machine.core.fence).await.flatten()
    }

    /// 返回当前必须对账、不能自动重投的副作用。
    pub async fn pending_reconciliation(&self) -> Vec<EffectRecoveryAction> {
        self.inspect(|machine| machine.pending_reconciliation())
            .await
            .unwrap_or_default()
    }

    /// 用外部查询、幂等存储或人工确认得到的权威结果解决一个未知 effect。
    pub async fn reconcile_effect(
        &self,
        effect_id: &EffectId,
        outcome: EffectOutcome,
    ) -> Result<(), AgentError> {
        self.ensure_started().await.map_err(runtime_error)?;
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Command::Reconcile {
                effect_id: effect_id.clone(),
                outcome,
                reply,
            })
            .map_err(|_| loop_closed())?;
        match rx.await {
            Ok(result) => result,
            Err(_) => Err(self.fenced().await),
        }
    }
}

/// 测试用的状态窥视：走同一条 `Inspect` 命令，不绕过单写者。
#[cfg(test)]
impl AgentSession {
    pub(crate) async fn mailbox_snapshot(&self) -> Vec<MailboxInput> {
        self.inspect(|machine| machine.core.mailbox.snapshot())
            .await
            .unwrap_or_default()
    }

    pub(crate) async fn effect_ledger(&self) -> EffectLedger {
        self.inspect(|machine| machine.core.effects.clone())
            .await
            .unwrap_or_default()
    }

    pub(crate) async fn has_active_run(&self) -> bool {
        self.inspect(|machine| machine.run.is_some())
            .await
            .unwrap_or(false)
    }

    pub(crate) async fn run_sequence(&self) -> u64 {
        self.inspect(|machine| machine.core.run_seq)
            .await
            .unwrap_or(0)
    }
}

impl Drop for AgentSession {
    fn drop(&mut self) {
        // 循环没启动过（`pending` 还在）就什么都不用做；启动过就叫它收尾。
        let _ = self.tx.send(Command::Shutdown);
    }
}

fn runtime_error(_: MailboxRejectedReason) -> AgentError {
    AgentError::new(
        AgentErrorKind::InvalidState,
        "tokio_runtime_required_for_session_driver",
    )
}

fn loop_closed() -> AgentError {
    AgentError::new(AgentErrorKind::InvalidState, "session_loop_closed")
}

fn ephemeral_session_id() -> SessionId {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
    SessionId::new(format!("ephemeral-{timestamp}-{sequence}"))
        .expect("generated session id is non-empty")
}
