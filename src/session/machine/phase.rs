//! 一次运行的阶段与一次模型请求的工具账。
//!
//! `Phase` 描述的是**在等世界的哪个回应**；对话记录走到哪一步由 `Turn` 子状态机管，
//! 两者正交。`Attempt` 是一次模型请求期间的工具账，但**账本本身在 `EffectLedger`**：
//! 这里只记「哪些槽已完整解析」和「模式定了没有」，已交付 / 已结算全部去账本里查。
//! 没有第二份 `submitted` / `settled`：两份真相迟早分叉。

use std::collections::BTreeMap;
use std::time::Instant;

use tokio::sync::oneshot;

use crate::events::RunStage;
use crate::run::{RequestBoundaryKind, Turn};
use crate::types::{
    AgentError, ModelOutput, ModelUsage, RunId, ToolBatchAbortReason, ToolBatchAttemptId,
    ToolBatchId, ToolCall, ToolCallBatch, ToolCallSlot,
};

use super::super::TurnOutcome;

/// 增量模式的决定时机是惰性的：第一个 `ToolCallReady` 到达时才问运行时。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AttemptMode {
    /// 还没问过运行时。这期间的 `ready` 只在内存，不交付也不落盘。
    Undecided,
    /// 运行时接管了；批次所有者任务已启动。
    Incremental,
    /// 运行时不接管；模型说完后整批 dispatch。
    Batch,
}

/// 一次模型请求的全部工具账（账本之外的那部分）。
pub(crate) struct Attempt {
    pub id: ToolBatchAttemptId,
    pub request_index: u64,
    pub started_at: Instant,
    /// 进入工具阶段的时刻；`ToolBatchFinished.duration_ms` 从这里算。
    pub tools_started_at: Option<Instant>,
    /// 已完整解析的调用，按槽序。
    pub ready: BTreeMap<ToolCallSlot, ToolCall>,
    /// 数组封口了吗。Chat 协议没有原生封口信号，`ModelDone` 时补判。
    pub sealed: Option<u32>,
    pub mode: AttemptMode,
    /// 运行时已经确认 `commit`：批次对象被它拿走在跑，所有者任务已退出，之后没有
    /// `abort` 可问。只在 `RuntimeAck(Commit, Ok)` 时置起——发出去不算，运行时可能因为
    /// 更早一步失败而根本没收到。
    pub committed: bool,
    /// 批次计时器已经武装过：一个 attempt 只有一个计时器（从进入等待阶段起算），
    /// 转入 `Aborting` 时不再来一个。
    pub timer_armed: bool,
    /// 模型成功说完后的规范输出；`Persist` 投影与整批 dispatch 都要用。
    pub output: Option<ModelOutput>,
    /// 流的首次失败。粘滞：一旦置位，后续 `ModelEvent` 一律拒收。
    pub failure: Option<AgentError>,
}

impl Attempt {
    pub(crate) fn new(run_id: &RunId, request_index: u64) -> Self {
        Self {
            id: ToolBatchAttemptId::new(format!("{}/model/{request_index}/tools", run_id.as_str())),
            request_index,
            started_at: Instant::now(),
            tools_started_at: None,
            ready: BTreeMap::new(),
            sealed: None,
            mode: AttemptMode::Undecided,
            committed: false,
            timer_armed: false,
            output: None,
            failure: None,
        }
    }

    /// 整批模式的批次可以从输出重建；`Turn` 给这一批用的 ID 就是 attempt ID。
    pub(crate) fn batch(&self, run_id: &RunId) -> Option<ToolCallBatch> {
        let output = self.output.as_ref()?;
        Some(ToolCallBatch {
            run_id: run_id.clone(),
            batch_id: ToolBatchId::from(&self.id),
            calls: output.tool_calls.clone(),
        })
    }
}

/// 压缩结束后回到哪里。
pub(crate) enum AfterCompaction {
    /// 回到这个边界继续：排空信箱、叫模型。`Turn` 此时停在 WaitingBoundary。
    Drain(RequestBoundaryKind),
    /// 收尾。压缩结果直接成为最终 conversation。
    Finish(CoreOutcome),
}

/// 在等世界的哪个回应。
pub(crate) enum Phase {
    /// 枢纽（瞬时）：`step` 内直接推进，不会停在这里被别的命令看到。
    ReadyToCall,
    /// 模型请求在飞。
    CallingModel(Attempt),
    /// 模型已成功说完，还有已交付的槽在账本里 InFlight（或运行时尚未回答要不要接管）。
    WaitingTools(Attempt),
    /// 断流 / 取消 / 运行时拒绝；已发 `Runtime::Abort`，等冻结报告。
    Aborting {
        attempt: Attempt,
        reason: ToolBatchAbortReason,
        error: AgentError,
    },
    /// 整批模式：`dispatch` 在飞。
    Dispatching { attempt: Attempt },
    /// 压缩在飞。
    Compacting {
        then: AfterCompaction,
        started_at: Instant,
    },
}

impl Phase {
    pub(crate) fn attempt(&self) -> Option<&Attempt> {
        match self {
            Self::CallingModel(attempt)
            | Self::WaitingTools(attempt)
            | Self::Aborting { attempt, .. }
            | Self::Dispatching { attempt } => Some(attempt),
            Self::ReadyToCall | Self::Compacting { .. } => None,
        }
    }

    pub(crate) fn attempt_mut(&mut self) -> Option<&mut Attempt> {
        match self {
            Self::CallingModel(attempt)
            | Self::WaitingTools(attempt)
            | Self::Aborting { attempt, .. }
            | Self::Dispatching { attempt } => Some(attempt),
            Self::ReadyToCall | Self::Compacting { .. } => None,
        }
    }

    /// 把 attempt 从阶段里拿出来，阶段暂时变成枢纽。调用方负责设回一个真实阶段。
    pub(crate) fn take_attempt(&mut self) -> Option<Attempt> {
        match std::mem::replace(self, Self::ReadyToCall) {
            Self::CallingModel(attempt)
            | Self::WaitingTools(attempt)
            | Self::Aborting { attempt, .. }
            | Self::Dispatching { attempt } => Some(attempt),
            other => {
                *self = other;
                None
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RunStats {
    pub model_requests: u64,
    pub tool_batches: u64,
    pub interrupted_tool_recoveries: u64,
}

pub(crate) enum CoreOutcome {
    Completed {
        output: ModelOutput,
        usage: Option<ModelUsage>,
    },
    Stopped,
    Failed {
        error: AgentError,
        stage: RunStage,
    },
}

/// 一次运行。
pub(crate) struct Run {
    pub id: RunId,
    pub turn: Turn,
    /// `PromptSource::base_context` 占了 turn 记录前多少项；这部分不进 conversation。
    pub protected_prefix_len: usize,
    pub stats: RunStats,
    pub phase: Phase,
    /// `join()` 的等待方。不持久化：重开后没人在等。
    pub waiters: Vec<oneshot::Sender<TurnOutcome>>,
    /// 最近一次压缩检查时的可提交长度；没变就不再压一次。
    pub last_compaction_checked_len: Option<usize>,
}

/// 运行内一次处理的结论。内层函数从不收尾，只回答「接下来怎样」；由外层统一执行。
pub(crate) enum Next {
    /// 阶段已设好，等世界。
    Park,
    /// 阶段是 `ReadyToCall`，从枢纽继续推进。
    Pump,
    /// 以这个结论收尾。
    Finish(CoreOutcome),
}
