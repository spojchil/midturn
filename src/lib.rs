//! 一个小型、提供方中立的智能体内核。
//!
//! # 工具调用的两种交付模式
//!
//! 两种模式互斥，但共享同一批次的关联和结果完整性约束：
//!
//! - **默认整批模式**：模型响应成功结束后，内核才把完整、有序的调用数组通过
//!   [`ToolRuntime::dispatch`] 一次性交给运行时。
//! - **增量接管模式**：运行时通过 [`ToolRuntime::begin_incremental`] 显式选择加入。每个
//!   已经完整解析的单项调用按 `(batch_attempt_id, slot)` 交给
//!   [`IncrementalToolBatch::submit`]；这不表示整个调用数组已经完整。只有独立的
//!   [`IncrementalToolBatch::calls_sealed`] 才封闭数组，而整个模型响应仍须成功结束后才会
//!   调用 [`IncrementalToolBatch::commit`]。
//!
//! 因而“逐项接管”与“完整批次提交”不是同一个边界。增量运行时可以在 `submit` 时立即执行，
//! 也可以只缓存调用并等到 `commit` 再执行；内核不会替应用选择调度策略。每个调用的结果都通过
//! [`ToolCallReporter::settled`] 逐槽上报（上报返回 `Ok` 即已落盘）；结果齐了、运行时也确认了
//! commit，内核才把它们按模型发出顺序拼成一轮写进对话。
//!
//! # 模型流中断与已经发生的工具事实
//!
//! 假设调用 A 已完整解析并被增量运行时接管，而调用 B 仍在传输时模型流断开：
//!
//! - 未成功结束的模型输出只是草稿，不会写入规范对话，也不能与普通 `ToolResults` 拼成一个
//!   看似完整的工具轮；
//! - session 会 `abort` 候选批次。运行时必须冻结已经传给 `submit` 的调用，并把 A 分类为
//!   已完成、确定未开始或结果未知；尚未完整解析的 B 从未提交给运行时；
//! - 已完成或结果未知的真实事实会作为一条普通恢复输入提交，而不是伪造残缺 Assistant
//!   输出或工具结果。恢复事实属于受保护的耐久化记录，[`Compaction`] 不得删除或改写；
//!   确定未开始的调用没有副作用，因此无需写入恢复回执。
//!
//! 回执中的已完成结果来自工具运行时；内核不会把未知结果改写成成功。每个外部 effect 会按
//! `Prepared -> InFlight -> Outcome` 写入 checkpoint；在 `InFlight` 后失去结果的 effect
//! 必须由应用显式对账，内核不会自动重投。SQLite 后端可通过 `sqlite` feature 启用。若服务商
//! 能够以稳定序号恢复同一个 response，adapter 可以在当前模型尝试内续流；发起一个新的模型
//! 响应不等同于续流。
//!
//! 内核还保证运行时输入仅在显式的请求或完成边界排空。提供方 wire 格式位于可选的
//! `adapters` 模块，不进入核心状态机。`openai` 与 `anthropic` feature 分别启用相应协议，
//! 二者共享 `http` transport。

#[cfg(any(feature = "openai", feature = "anthropic"))]
pub mod adapters;
mod events;
mod mailbox;
pub mod persistence;
mod ports;
mod run;
mod session;
mod types;

pub use events::{
    AgentEvent, ContentEvent, ContentEventKind, ContentObserver, EventCategory, EventKind,
    EventLevel, EventMetadata, FilteredObserver, LevelFilter, ModelStreamObservation,
    NoopContentObserver, NoopObserver, NoopStreamObserver, ObservedModelStreamEvent, Observer,
    RunStage, StreamObserver, ToolCallSummary,
};
pub use mailbox::{Delivery, MailboxInput, MailboxRejected, MailboxRejectedReason};
pub use ports::{
    Compaction, IncrementalToolBatch, Model, ModelRequest, ModelResponse, ModelStreamEvent,
    ModelStreamSink, PortFuture, PromptSource, ToolCallReporter, ToolRuntime,
};
pub use run::{RequestBoundaryKind, Turn, TurnStep};
pub use session::{
    AgentSession, Consumption, ContextBudget, CumulativeBudget, Enqueued, HoldReason,
    RescheduleOutcome, Rescheduled, RunHandle, SessionBudget, SessionConfig, TurnOutcome,
};
pub use types::{
    durable_fact_kind, validate_transcript, AbortClassification, AbortedSlot, AgentError,
    AgentErrorKind, ContentPart, DurableFactKind, IncrementalToolCall, InputMessage,
    InterruptedToolBatchReceipt, InterruptedToolCallOutcome, InterruptedToolCallReceipt,
    JsonObject, ModelOutput, ModelUsage, RunId, ToolBatchAbortReason, ToolBatchAttemptId,
    ToolBatchId, ToolBatchStart, ToolCall, ToolCallBatch, ToolCallId, ToolCallSlot, ToolDefinition,
    ToolName, ToolResult, ToolResultBatch, ToolResultStatus, TranscriptItem,
};
