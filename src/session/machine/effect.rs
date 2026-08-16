//! `step` 产出、驱动循环执行的动作。`step` 从不执行它们。
//!
//! 顺序是契约的一部分：`Persist` 排在依赖它的 `Runtime::Submit` / `Reply` 之前；
//! `Persist` 失败则整批剩余 effect 丢弃。

use std::time::Duration;

use crate::events::{AgentEvent, ContentEvent, ObservedModelStreamEvent};
use crate::ports::ModelRequest;
use crate::types::{
    IncrementalToolCall, RunId, ToolBatchAbortReason, ToolBatchAttemptId, ToolBatchId,
    ToolBatchStart, ToolCallBatch, TranscriptItem,
};

/// 回复外部调用 / 适配器 / 运行时。类型擦除成一个闭包：具体的 oneshot 类型各不相同，
/// 而循环只需要「执行它」。
pub(crate) type ReplyFn = Box<dyn FnOnce() + Send>;

/// 发给批次所有者任务的一次运行时调用。它顺序执行，保住 `&mut self` 语义。
pub(crate) enum RuntimeCall {
    Submit(IncrementalToolCall),
    Seal(u32),
    Commit,
    Abort(ToolBatchAbortReason),
}

pub(crate) enum Observation {
    Agent(AgentEvent),
    Content(ContentEvent),
    Stream(ObservedModelStreamEvent),
}

pub(crate) enum Effect {
    /// 快照由循环在 `step` 返回后从 `&Machine` 投影，不由 `step` 构造。
    Persist,
    Reply(ReplyFn),
    /// 「这一步结束后再来看一眼信箱」。收尾和开始是两次转移、两次落盘：盘上不该出现
    /// 「上一轮刚结束、下一轮已经开跑」这种合在一起的中间态。循环把它变成一条 `Kick`。
    Wake,
    /// spawn 模型任务。
    CallModel {
        attempt: ToolBatchAttemptId,
        request: ModelRequest,
    },
    /// 丢掉模型任务的 future。适配器下一次 `emit` 会拿到 `Err`；这条只是不再等它。
    AbortModel {
        attempt: ToolBatchAttemptId,
    },
    /// spawn 批次所有者任务；它先调 `begin_incremental`，回报 `RuntimeAck(Begin)`。
    BeginIncremental {
        attempt: ToolBatchAttemptId,
        start: ToolBatchStart,
    },
    Runtime {
        attempt: ToolBatchAttemptId,
        call: RuntimeCall,
    },
    /// 让批次所有者任务丢掉 `Box<dyn IncrementalToolBatch>`。这是给运行时的「内核不再等你」
    /// 信号；它在 Drop 里想取消什么自己取消。
    Release {
        attempt: ToolBatchAttemptId,
    },
    /// spawn 整批 dispatch。
    Dispatch {
        batch_id: ToolBatchId,
        batch: ToolCallBatch,
    },
    AbortDispatch {
        batch_id: ToolBatchId,
    },
    Compact {
        run_id: RunId,
        conversation: Vec<TranscriptItem>,
    },
    AbortCompact {
        run_id: RunId,
    },
    /// spawn 一个 sleep，到点发 `BatchTimedOut`。attempt 已关则被忽略。
    ArmTimer {
        attempt: ToolBatchAttemptId,
        after: Duration,
    },
    /// 序号在这里是占位 0：投递任务在 `enabled` 通过之后才取号填入。`step` 不调观察端。
    Emit(Observation),
}
