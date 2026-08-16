//! 会话状态机：一份状态、一个纯函数 `step`、一个写者。
//!
//! 这里没有 `async`、没有锁、没有 I/O。世界的一切以 [`Command`] 进来，`step` 改
//! [`Machine`] 并产出有序的 [`Effect`]，由驱动循环（`session::engine`）执行。持久化与观测
//! 不是独立写者：它们只是 `step` 产出的两种 effect。
//!
//! # 结构
//!
//! [`Machine`] = [`Core`]（除运行之外的一切）+ `run: Option<Run>`。运行内的逻辑写成
//! `impl Core { fn xxx(&mut self, run: &mut Run, out) -> Next }`——两个借用不相交，
//! 内层函数从不收尾，只回答 [`Next`]；由 `Machine` 统一执行 `Pump` / `Finish`。
//!
//! 唯一的两处「不纯」是两个同步端口调用：`PromptSource::base_context()`（建运行时）和
//! `ToolRuntime::definitions()`（建模型请求时）。两者都不做 I/O，都在 `catch_unwind` 里。

mod command;
mod effect;
mod mailbox;
mod model;
mod phase;
mod recovery;
mod run;
mod snapshot;
mod tools;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use tokio::sync::oneshot;

pub(crate) use command::{Command, RescheduleReply, RuntimeOp, StartOutcome};
pub(crate) use effect::{Effect, Observation, RuntimeCall};
pub(crate) use phase::{
    AfterCompaction, Attempt, AttemptMode, CoreOutcome, Next, Phase, Run, RunStats,
};

use crate::events::{
    AgentEvent, ContentEvent, ContentEventKind, ContentObserver, EventKind, ModelStreamObservation,
    ObservedModelStreamEvent, Observer, RunStage, StreamObserver,
};
use crate::mailbox::{Mailbox, MailboxRejected, MailboxRejectedReason};
use crate::persistence::{EffectLedger, RunSnapshot, SessionId};
use crate::ports::{PromptSource, ToolRuntime};
use crate::types::{AgentError, AgentErrorKind, RunId, ToolBatchId, TranscriptItem};

use super::{Consumption, SessionConfig, TurnOutcome};

/// 三个观察端。`step` **不调它们的任何方法**——用户代码一律在循环外的投递任务上跑
/// （`enabled` 过滤 + 取号 + `observe`）。`step` 只看「装没装」决定要不要构造事件：
/// 没装就连克隆都省了。
#[derive(Default)]
pub(crate) struct Observers {
    pub observer: Option<Arc<dyn Observer>>,
    pub content: Option<Arc<dyn ContentObserver>>,
    pub stream: Option<Arc<dyn StreamObserver>>,
}

/// 会话状态里除「当前运行」之外的一切。
pub(crate) struct Core {
    pub(crate) session_id: SessionId,
    pub(crate) config: SessionConfig,
    pub(crate) prompt: Arc<dyn PromptSource>,
    pub(crate) tools: Arc<dyn ToolRuntime>,
    pub(crate) observers: Observers,

    // ── 持久状态（进 checkpoint）──
    pub(crate) conversation: Arc<Vec<TranscriptItem>>,
    pub(crate) mailbox: Mailbox,
    pub(crate) effects: EffectLedger,
    pub(crate) consumption: Consumption,
    pub(crate) run_seq: u64,
    /// 重开时盘上带来的半截运行。`Resume` 之前不动它；`Enqueue` 只会得到 `Held(ResumeRequired)`。
    pub(crate) recovered: Option<RunSnapshot>,

    // ── 进程内状态（不进 checkpoint）──
    pub(crate) auto_start: bool,
    /// 持久化 writer 被隔离的原因。一旦置位，所有变更命令一律拒绝。
    pub(crate) fence: Option<AgentErrorKind>,
    idle_waiters: Vec<oneshot::Sender<()>>,
    /// 本次 `step` 是否改了持久状态。是则 `Persist` 成为效果列表的第一项。
    dirty: bool,
}

/// 会话的全部状态。它是驱动循环任务的局部变量，没有第二个持有者。
pub(crate) struct Machine {
    pub(crate) core: Core,
    pub(crate) run: Option<Run>,
}

impl Machine {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        session_id: SessionId,
        config: SessionConfig,
        prompt: Arc<dyn PromptSource>,
        tools: Arc<dyn ToolRuntime>,
        conversation: Arc<Vec<TranscriptItem>>,
        mailbox: Mailbox,
        effects: EffectLedger,
        consumption: Consumption,
        run_seq: u64,
        recovered: Option<RunSnapshot>,
    ) -> Self {
        let auto_start = config.auto_start;
        Self {
            core: Core {
                session_id,
                config,
                prompt,
                tools,
                observers: Observers::default(),
                conversation,
                mailbox,
                effects,
                consumption,
                run_seq,
                recovered,
                auto_start,
                fence: None,
                idle_waiters: Vec::new(),
                dirty: false,
            },
            run: None,
        }
    }

    /// 处理一条命令。返回的 effect 有序；若本次改了持久状态，第一项是 `Persist`。
    pub(crate) fn step(&mut self, command: Command) -> Vec<Effect> {
        let mut out = Vec::new();
        self.core.dirty = false;

        if let Some(kind) = self.core.fence {
            self.reject_fenced(kind, command, &mut out);
            return out;
        }

        match command {
            Command::Enqueue {
                input,
                may_start,
                expected_run,
                reply,
            } => self.on_enqueue(input, may_start, expected_run, reply, &mut out),
            Command::Reschedule { from, to, reply } => {
                self.on_reschedule(from, to, reply, &mut out)
            }
            Command::SetAutoStart(enabled) => self.on_set_auto_start(enabled, &mut out),
            Command::Cancel { reply } => self.on_cancel(reply, &mut out),
            Command::Resume { reply } => self.on_resume(reply, &mut out),
            Command::WaitIdle { reply } => {
                if self.run.is_none() {
                    reply_to(&mut out, reply, ());
                } else {
                    self.core.idle_waiters.push(reply);
                }
            }
            Command::Reconcile {
                effect_id,
                outcome,
                reply,
            } => self.on_reconcile(effect_id, outcome, reply, &mut out),
            Command::Boot => self.on_boot(&mut out),
            Command::Kick => {
                let _ = self.try_start(&mut out);
            }
            Command::Shutdown => {}
            Command::Inspect(look) => look(self),

            Command::ModelEvent {
                attempt,
                event,
                reply,
            } => self.on_model_event(attempt, event, reply, &mut out),
            Command::ModelDone { attempt, response } => {
                self.on_model_done(attempt, response, &mut out)
            }
            Command::ModelFailed { attempt, error } => {
                self.on_model_failed(attempt, error, &mut out)
            }

            Command::ToolSettled {
                attempt,
                slot,
                result,
                reply,
            } => self.on_tool_settled(attempt, slot, result, reply, &mut out),
            Command::RuntimeAck {
                attempt,
                op,
                result,
            } => self.on_runtime_ack(attempt, op, result, &mut out),
            Command::BatchAborted { attempt, result } => {
                self.on_batch_aborted(attempt, result, &mut out)
            }
            Command::BatchDispatched { batch_id, result } => {
                self.on_batch_dispatched(batch_id, result, &mut out)
            }
            Command::BatchTimedOut { attempt } => self.on_batch_timed_out(attempt, &mut out),

            Command::Compacted { run_id, result } => self.on_compacted(run_id, result, &mut out),

            Command::PersistFailed { error } => self.fence(error, &mut out),
        }

        if self.core.dirty {
            out.insert(0, Effect::Persist);
        }
        out
    }

    /// 持久化失败：内存里的状态从此与盘上分叉；不回滚，只隔离。重开以盘上为准。
    ///
    /// 活跃运行就地宣告失败——它的等待方拿到的是「writer 被隔离」，不是通道关闭。
    /// 这里**不能** `touch`：隔离之后不再有 `Persist`。
    fn fence(&mut self, error: AgentError, out: &mut Vec<Effect>) {
        self.core.fence = Some(error.kind);
        let Some(run) = self.run.take() else {
            self.core.wake_idle_waiters(out);
            return;
        };
        let run_id = run.id.clone();
        // 收回还在外面跑的任务：它们的回执进来也只会被 reject_fenced 丢掉。
        if let Some(attempt) = run.phase.attempt() {
            out.push(Effect::AbortModel {
                attempt: attempt.id.clone(),
            });
            out.push(Effect::Release {
                attempt: attempt.id.clone(),
            });
        }
        if let Phase::Dispatching { attempt } = &run.phase {
            out.push(Effect::AbortDispatch {
                batch_id: ToolBatchId::from(&attempt.id),
            });
        }
        if matches!(run.phase, Phase::Compacting { .. }) {
            out.push(Effect::AbortCompact {
                run_id: run_id.clone(),
            });
        }
        let error_kind = error.kind;
        self.core.emit(out, EventKind::RunFailed, |sequence| {
            AgentEvent::RunFailed {
                sequence,
                run_id: run_id.clone(),
                stage: RunStage::Boundary,
                error_kind,
            }
        });
        let outcome = TurnOutcome::Failed {
            run_id,
            stage: RunStage::Boundary,
            usage: run.turn.usage().cloned(),
            error,
        };
        for waiter in run.waiters {
            reply_to(out, waiter, outcome.clone());
        }
        self.core.wake_idle_waiters(out);
    }

    /// 被隔离后：带回复的命令回 `Err`，其余丢弃。
    fn reject_fenced(&mut self, kind: AgentErrorKind, command: Command, out: &mut Vec<Effect>) {
        let error = || {
            AgentError::new(
                kind,
                "session_writer_is_fenced;reopen_from_checkpoint_store",
            )
        };
        match command {
            Command::Enqueue { input, reply, .. } => {
                let reason = if kind == AgentErrorKind::PersistenceCommitUnknown {
                    MailboxRejectedReason::PersistenceCommitUnknown
                } else {
                    MailboxRejectedReason::Persistence
                };
                reply_to(out, reply, Err(MailboxRejected { reason, input }));
            }
            Command::Reschedule { reply, .. } => reply_to(out, reply, Err(error())),
            Command::Cancel { reply } => reply_to(out, reply, None),
            Command::Resume { reply } => reply_to(out, reply, Err(error())),
            Command::WaitIdle { reply } => reply_to(out, reply, ()),
            Command::Reconcile { reply, .. } => reply_to(out, reply, Err(error())),
            Command::ModelEvent { reply, .. } => reply_to(out, reply, Err(error())),
            Command::ToolSettled { reply, .. } => reply_to(out, reply, Err(error())),
            Command::Inspect(look) => look(self),
            Command::SetAutoStart(_)
            | Command::Boot
            | Command::Kick
            | Command::Shutdown
            | Command::ModelDone { .. }
            | Command::ModelFailed { .. }
            | Command::RuntimeAck { .. }
            | Command::BatchAborted { .. }
            | Command::BatchDispatched { .. }
            | Command::BatchTimedOut { .. }
            | Command::Compacted { .. }
            | Command::PersistFailed { .. } => {}
        }
    }

    /// 执行内层函数给出的结论。所有「运行内处理」都以这个收口。
    pub(crate) fn settle(&mut self, next: Next, out: &mut Vec<Effect>) {
        match next {
            Next::Park => {}
            Next::Pump => self.pump(out),
            Next::Finish(outcome) => self.finish_run(outcome, out),
        }
    }
}

impl Core {
    /// 标记「本次 step 改了持久状态」。所有改 conversation / mailbox / effects /
    /// consumption / run_seq / run 的地方都必须调它。
    pub(crate) fn touch(&mut self) {
        self.dirty = true;
    }

    /// 运行结束时唤醒 `wait_until_idle` 的等待方。
    pub(crate) fn wake_idle_waiters(&mut self, out: &mut Vec<Effect>) {
        for waiter in self.idle_waiters.drain(..) {
            reply_to(out, waiter, ());
        }
    }

    // ── 观测 ─────────────────────────────────────────────────────
    //
    // 序号占位为 0：投递任务在 `enabled` 通过之后才取号填入。这里不调用任何用户代码。

    pub(crate) fn emit(
        &mut self,
        out: &mut Vec<Effect>,
        kind: EventKind,
        build: impl FnOnce(u64) -> AgentEvent,
    ) {
        if self.observers.observer.is_none() {
            return;
        }
        let event = build(0);
        debug_assert_eq!(event.metadata(), kind.metadata());
        out.push(Effect::Emit(Observation::Agent(event)));
    }

    pub(crate) fn emit_content(
        &mut self,
        out: &mut Vec<Effect>,
        kind: ContentEventKind,
        build: impl FnOnce(u64) -> ContentEvent,
    ) {
        if self.observers.content.is_none() {
            return;
        }
        let event = build(0);
        debug_assert_eq!(event.kind(), kind);
        out.push(Effect::Emit(Observation::Content(event)));
    }

    pub(crate) fn emit_stream(
        &mut self,
        out: &mut Vec<Effect>,
        run_id: &RunId,
        request_index: u64,
        build: impl FnOnce() -> ModelStreamObservation,
    ) {
        if self.observers.stream.is_none() {
            return;
        }
        out.push(Effect::Emit(Observation::Stream(
            ObservedModelStreamEvent {
                sequence: 0,
                run_id: run_id.clone(),
                request_index,
                payload: build(),
            },
        )));
    }
}

/// 把一次 oneshot 回复包成 effect。放进列表的位置就是它相对 `Persist` 的位置。
pub(crate) fn reply_to<T: Send + 'static>(
    out: &mut Vec<Effect>,
    reply: oneshot::Sender<T>,
    value: T,
) {
    out.push(Effect::Reply(Box::new(move || {
        let _ = reply.send(value);
    })));
}
