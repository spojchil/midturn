//! 进入状态机的全部消息。
//!
//! 只有一个 channel、一个消费者。**轮外**来源（信箱内容与重新评估的触发）能从
//! `run == None` 造出一个 `Run`；**轮内**来源（模型流、工具运行时、压缩、计时器）带着
//! `attempt` 或 `run_id`，没有匹配的活动就被忽略或拒绝。机制上两者没有区别。

use tokio::sync::oneshot;

use crate::mailbox::{Delivery, MailboxInput, MailboxRejected};
use crate::persistence::{EffectId, EffectOutcome};
use crate::ports::{ModelResponse, ModelStreamEvent};
use crate::types::{
    AbortClassification, AgentError, RunId, ToolBatchAttemptId, ToolBatchId, ToolCallSlot,
    ToolResult, ToolResultBatch, TranscriptItem,
};

use super::super::{HoldReason, TurnOutcome};
use super::Machine;

/// 一次投递 / 改期 / 恢复之后会发生什么。公共 `Enqueued` 是它加上 `RunHandle` 的包装。
pub(crate) enum StartOutcome {
    /// 这次调用把空闲的会话叫醒了。`completion` 在运行结束时收到终态。
    Started {
        run_id: RunId,
        completion: oneshot::Receiver<TurnOutcome>,
    },
    /// 已经有运行在跑，内容会在它的下一个边界进入。
    Pending,
    /// 收下了，但暂时不会有人来取。
    Held(HoldReason),
}

pub(crate) struct RescheduleReply {
    pub envelopes: usize,
    pub outcome: StartOutcome,
}

impl std::fmt::Debug for StartOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Started { run_id, .. } => f.debug_tuple("Started").field(run_id).finish(),
            Self::Pending => f.write_str("Pending"),
            Self::Held(reason) => f.debug_tuple("Held").field(reason).finish(),
        }
    }
}

/// 批次所有者任务回报的一次运行时调用完成。
#[derive(Debug)]
pub(crate) enum RuntimeOp {
    /// `begin_incremental` 返回了；`true` = 运行时接管。
    Begin(bool),
    Submit,
    Seal,
    Commit,
}

pub(crate) enum Command {
    // ── 轮外 ──
    Enqueue {
        input: MailboxInput,
        /// `enqueue_for` 传 `false`：只针对当前这一轮，永不开新运行。
        may_start: bool,
        /// `enqueue_for` 的目标运行；不匹配则 `WrongRun`。
        expected_run: Option<RunId>,
        reply: oneshot::Sender<Result<StartOutcome, MailboxRejected>>,
    },
    Reschedule {
        from: Delivery,
        to: Delivery,
        reply: oneshot::Sender<Result<RescheduleReply, AgentError>>,
    },
    SetAutoStart(bool),
    /// 取消当前运行（若有）。回复被取消的运行 ID。
    Cancel {
        reply: oneshot::Sender<Option<RunId>>,
    },
    /// 显式续跑重开后的运行。`Ok(None)` = 没有等待恢复的运行。
    Resume {
        reply: oneshot::Sender<Result<Option<StartOutcome>, AgentError>>,
    },
    WaitIdle {
        reply: oneshot::Sender<()>,
    },
    Reconcile {
        effect_id: EffectId,
        outcome: EffectOutcome,
        reply: oneshot::Sender<Result<(), AgentError>>,
    },
    /// 循环启动时喂一次：重开后按 phase 决定；空闲则看信箱。
    Boot,
    /// `Effect::Wake` 的回声：空闲就看信箱。
    Kick,
    /// 会话外壳被丢弃：循环收尾退出。不进 `step`——循环直接处理。
    Shutdown,
    /// 只读查看。不是状态转移，循环直接执行，不进 `step`。
    Inspect(Box<dyn FnOnce(&Machine) + Send>),

    // ── 轮内：模型适配器 ──
    ModelEvent {
        attempt: ToolBatchAttemptId,
        event: ModelStreamEvent,
        /// 适配器 await 它：`Err` = 本次尝试已失效，停止发布。这也是背压。
        reply: oneshot::Sender<Result<(), AgentError>>,
    },
    ModelDone {
        attempt: ToolBatchAttemptId,
        response: ModelResponse,
    },
    ModelFailed {
        attempt: ToolBatchAttemptId,
        error: AgentError,
    },

    // ── 轮内：工具运行时 ──
    ToolSettled {
        attempt: ToolBatchAttemptId,
        slot: ToolCallSlot,
        result: ToolResult,
        /// `Ok` 在落盘之后才回。
        reply: oneshot::Sender<Result<(), AgentError>>,
    },
    RuntimeAck {
        attempt: ToolBatchAttemptId,
        op: RuntimeOp,
        result: Result<(), AgentError>,
    },
    BatchAborted {
        attempt: ToolBatchAttemptId,
        result: Result<AbortClassification, AgentError>,
    },
    BatchDispatched {
        batch_id: ToolBatchId,
        result: Result<ToolResultBatch, AgentError>,
    },
    BatchTimedOut {
        attempt: ToolBatchAttemptId,
    },

    // ── 轮内：压缩 ──
    Compacted {
        run_id: RunId,
        result: Result<Vec<TranscriptItem>, AgentError>,
    },

    // ── 持久化 ──
    PersistFailed {
        error: AgentError,
    },
}
