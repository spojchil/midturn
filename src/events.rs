//! 生命周期事件与内容事件，以及把它们串成一条有序流的发布器。
//!
//! # 两类数据，两个端口
//!
//! **数据分类不用日志等级表示。** 等级回答的是「这条有多啰嗦」，不是「这条里有没有
//! 提示词和凭据」；用 `Trace` 当安全边界，任何自己实现 [`Observer`] 而不套
//! [`FilteredObserver`] 的人都会拿到本不该拿到的东西。所以按数据分类拆成两个端口：
//!
//! - [`Observer`] 收 [`AgentEvent`]：只有数量、标识符、名称、边界和错误类别，
//!   **永远不携带正文**。它可以直接进普通日志。
//! - [`ContentObserver`] 收 [`ContentEvent`]：完整提示词、模型输出、工具结果，
//!   以及未经脱敏的 [`crate::ToolResult::metadata`]。它默认是
//!   [`NoopContentObserver`]，**装上它本身就是选择加入**，不需要再靠等级约定。
//!
//! [`StreamObserver`] 同属内容一类：它收模型正文增量和完整工具参数，默认也是 Noop。
//!
//! # 序号
//!
//! 三个端口共用一个序号计数器，由**唯一的观测投递任务**在 `enabled` 通过之后、`observe`
//! 之前按到达顺序分配，因此**序号顺序就是观察顺序**，三条流可以直接按序号归并成一条。
//! 过滤不造洞；只有 `observe` 自己 panic 会留下一个洞。详见 [`AgentEvent::sequence`]。

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use crate::mailbox::Delivery;
use crate::ports::ModelStreamEvent;
use crate::run::RequestBoundaryKind;
use crate::types::{
    AgentErrorKind, ModelOutput, ModelUsage, RunId, ToolBatchAttemptId, ToolBatchId, ToolCallId,
    ToolCallSlot, ToolDefinition, ToolName, ToolResultBatch, TranscriptItem,
};

#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum EventLevel {
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4,
    Trace = 5,
}

/// `Off` 既是总开关，其余值表示允许的最大详细程度。
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum LevelFilter {
    Off = 0,
    Error = 1,
    Warn = 2,
    #[default]
    Info = 3,
    Debug = 4,
    Trace = 5,
}

impl LevelFilter {
    pub const fn allows(self, level: EventLevel) -> bool {
        level as u8 <= self as u8
    }

    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Error,
            2 => Self::Warn,
            3 => Self::Info,
            4 => Self::Debug,
            5 => Self::Trace,
            _ => Self::Off,
        }
    }
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum EventCategory {
    Lifecycle = 0,
    Mailbox = 1,
    Boundary = 2,
    Model = 3,
    Tools = 4,
    Compaction = 5,
}

impl EventCategory {
    const ALL_MASK: u8 = (1 << 6) - 1;

    const fn mask(self) -> u8 {
        1 << self as u8
    }
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventKind {
    RunStarted,
    MailboxEnqueued,
    /// 排队中的一类内容被整体改了交付语义（`reschedule`）。
    MailboxRescheduled,
    BoundaryDrained,
    CompletionSealed,
    ModelRequestStarted,
    ModelRequestFinished,
    ToolCallsSealed,
    /// 运行时在批次提交前主动上报了一个单项调用已经结算。
    ToolCallSettled,
    ToolBatchAborted,
    ToolBatchStarted,
    ToolBatchFinished,
    CompactionStarted,
    CompactionFinished,
    RunCompleted,
    RunStopped,
    RunFailed,
}

impl EventKind {
    pub const fn metadata(self) -> EventMetadata {
        let (level, category) = match self {
            Self::RunFailed => (EventLevel::Error, EventCategory::Lifecycle),
            Self::RunStarted | Self::RunCompleted | Self::RunStopped => {
                (EventLevel::Info, EventCategory::Lifecycle)
            }
            Self::ModelRequestStarted | Self::ModelRequestFinished => {
                (EventLevel::Debug, EventCategory::Model)
            }
            Self::ToolCallsSealed
            | Self::ToolCallSettled
            | Self::ToolBatchAborted
            | Self::ToolBatchStarted
            | Self::ToolBatchFinished => (EventLevel::Debug, EventCategory::Tools),
            Self::CompactionStarted | Self::CompactionFinished => {
                (EventLevel::Debug, EventCategory::Compaction)
            }
            Self::MailboxEnqueued | Self::MailboxRescheduled => {
                (EventLevel::Trace, EventCategory::Mailbox)
            }
            Self::BoundaryDrained | Self::CompletionSealed => {
                (EventLevel::Trace, EventCategory::Boundary)
            }
        };
        EventMetadata {
            kind: self,
            level,
            category,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EventMetadata {
    pub kind: EventKind,
    pub level: EventLevel,
    pub category: EventCategory,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunStage {
    Boundary,
    Model,
    Tools,
    Compaction,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolCallSummary {
    pub id: ToolCallId,
    pub name: ToolName,
}

#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum AgentEvent {
    RunStarted {
        sequence: u64,
        run_id: RunId,
        prior_transcript_items: usize,
    },
    MailboxEnqueued {
        sequence: u64,
        /// 投递时若没有活跃运行则为 `None`——空闲收件是正常路径。
        run_id: Option<RunId>,
        delivery: Delivery,
        item_count: usize,
    },
    /// `reschedule` 改变了 `envelopes` 个信封的交付语义。没有内容，只有意图的变化：
    /// 光看 `MailboxEnqueued` 已经推不出当前队列的意图了，这条把它补齐。只在落盘之后发出。
    MailboxRescheduled {
        sequence: u64,
        run_id: Option<RunId>,
        from: Delivery,
        to: Delivery,
        envelopes: usize,
    },
    BoundaryDrained {
        sequence: u64,
        run_id: RunId,
        kind: RequestBoundaryKind,
        item_count: usize,
    },
    CompletionSealed {
        sequence: u64,
        run_id: RunId,
    },
    ModelRequestStarted {
        sequence: u64,
        run_id: RunId,
        request_index: u64,
        transcript_items: usize,
        function_tools: usize,
    },
    ModelRequestFinished {
        sequence: u64,
        run_id: RunId,
        request_index: u64,
        local_tool_calls: usize,
        duration_ms: u64,
        usage: Option<ModelUsage>,
    },
    ToolCallsSealed {
        sequence: u64,
        run_id: RunId,
        batch_attempt_id: ToolBatchAttemptId,
        call_count: u32,
    },
    /// 运行时在 `commit` 之前主动上报了单项结算，内核已将其持久化。
    ///
    /// 这条事件出现之后，该 slot 不会再被判为 `OutcomeUnknown`。
    ToolCallSettled {
        sequence: u64,
        run_id: RunId,
        batch_attempt_id: ToolBatchAttemptId,
        slot: ToolCallSlot,
        call_id: ToolCallId,
    },
    ToolBatchAborted {
        sequence: u64,
        run_id: RunId,
        batch_attempt_id: ToolBatchAttemptId,
        settled_count: usize,
        cancelled_count: usize,
        unknown_count: usize,
    },
    ToolBatchStarted {
        sequence: u64,
        run_id: RunId,
        batch_id: ToolBatchId,
        calls: Vec<ToolCallSummary>,
    },
    ToolBatchFinished {
        sequence: u64,
        run_id: RunId,
        batch_id: ToolBatchId,
        result_count: usize,
        error_count: usize,
        duration_ms: u64,
    },
    CompactionStarted {
        sequence: u64,
        run_id: RunId,
        transcript_items: usize,
        estimated_bytes: usize,
    },
    CompactionFinished {
        sequence: u64,
        run_id: RunId,
        transcript_items: usize,
        duration_ms: u64,
    },
    RunCompleted {
        sequence: u64,
        run_id: RunId,
        model_requests: u64,
        tool_batches: u64,
        interrupted_tool_recoveries: u64,
        usage: Option<ModelUsage>,
    },
    RunStopped {
        sequence: u64,
        run_id: RunId,
    },
    RunFailed {
        sequence: u64,
        run_id: RunId,
        stage: RunStage,
        /// 完整错误摘要可能含服务商正文，因此事件只携带稳定错误类别。
        error_kind: AgentErrorKind,
    },
}

impl AgentEvent {
    /// 单调序号，在事件真正投递给观察端的那一刻分配。
    ///
    /// **保证（对三个端口全部成立，含 [`AgentEvent::MailboxEnqueued`] 与模型流增量）**：
    /// 三个端口共用**一个投递任务**，它按状态机产出的顺序逐条 `enabled` → 取号 → `observe`，
    /// 中间没有第二个写者，因此
    ///
    /// - 序号严格递增；
    /// - **序号顺序就是观察顺序**——不存在「先看到 N+1 再看到 N」；
    /// - 被过滤掉的事件不消耗序号：**过滤不造洞**（过滤的线性化点就是投递任务里那次
    ///   `enabled` 调用，不是事件产生的那一刻）；**唯一会造洞的是 `observe` 自己 panic**——
    ///   号已经取了，那条没投出去。
    ///
    /// 于是 [`Observer`]、[`ContentObserver`] 和 [`StreamObserver`] 三条流可以直接按
    /// 序号归并成一条完整的有序流，不需要消费方自己重排，也不需要内核提供一个合并端口。
    ///
    /// **顺序是真实发生顺序，不是「整齐」顺序。** 增量工具运行时可以在模型还在生成时
    /// 就跑完一个调用，因此
    /// [`ContentEvent::ToolBatchResults`] 或 [`AgentEvent::ToolCallSettled`] 出现在
    /// [`AgentEvent::ToolCallsSealed`] **之前**是正常的——那批调用当时确实还没封口。
    /// 有些实现会刻意缓冲以避免这个形状；本内核不缓冲，因为缓冲等于报告一个没发生过的顺序。
    ///
    /// **代价，实现观察端时必须知道**：
    ///
    /// - 三个端口的回调是**互相串行**的，跑在投递任务上、不在会话的驱动循环上：慢回调
    ///   只会让观测落后，不会拖慢会话本身——但队列是无界的，落后太多就是内存。
    /// - `enabled` 和 `observe` 都是同步回调；不要在里面阻塞等待本会话的异步 API
    ///   （`current_thread` runtime 上 `block_on` 会 panic，多线程 runtime 上只是把投递
    ///   任务卡住）。
    /// - 观察端 panic 会被隔离，但那一条事件已经消耗掉的序号不会归还——**空洞只可能
    ///   由 `observe` 自己的 panic 造成**（`enabled` panic 按「关掉」处理，不取号）。
    pub const fn sequence(&self) -> u64 {
        match self {
            Self::RunStarted { sequence, .. }
            | Self::MailboxEnqueued { sequence, .. }
            | Self::MailboxRescheduled { sequence, .. }
            | Self::BoundaryDrained { sequence, .. }
            | Self::CompletionSealed { sequence, .. }
            | Self::ModelRequestStarted { sequence, .. }
            | Self::ModelRequestFinished { sequence, .. }
            | Self::ToolCallsSealed { sequence, .. }
            | Self::ToolCallSettled { sequence, .. }
            | Self::ToolBatchAborted { sequence, .. }
            | Self::ToolBatchStarted { sequence, .. }
            | Self::ToolBatchFinished { sequence, .. }
            | Self::CompactionStarted { sequence, .. }
            | Self::CompactionFinished { sequence, .. }
            | Self::RunCompleted { sequence, .. }
            | Self::RunStopped { sequence, .. }
            | Self::RunFailed { sequence, .. } => *sequence,
        }
    }

    pub const fn kind(&self) -> EventKind {
        match self {
            Self::RunStarted { .. } => EventKind::RunStarted,
            Self::MailboxEnqueued { .. } => EventKind::MailboxEnqueued,
            Self::MailboxRescheduled { .. } => EventKind::MailboxRescheduled,
            Self::BoundaryDrained { .. } => EventKind::BoundaryDrained,
            Self::CompletionSealed { .. } => EventKind::CompletionSealed,
            Self::ModelRequestStarted { .. } => EventKind::ModelRequestStarted,
            Self::ModelRequestFinished { .. } => EventKind::ModelRequestFinished,
            Self::ToolCallsSealed { .. } => EventKind::ToolCallsSealed,
            Self::ToolCallSettled { .. } => EventKind::ToolCallSettled,
            Self::ToolBatchAborted { .. } => EventKind::ToolBatchAborted,
            Self::ToolBatchStarted { .. } => EventKind::ToolBatchStarted,
            Self::ToolBatchFinished { .. } => EventKind::ToolBatchFinished,
            Self::CompactionStarted { .. } => EventKind::CompactionStarted,
            Self::CompactionFinished { .. } => EventKind::CompactionFinished,
            Self::RunCompleted { .. } => EventKind::RunCompleted,
            Self::RunStopped { .. } => EventKind::RunStopped,
            Self::RunFailed { .. } => EventKind::RunFailed,
        }
    }

    pub const fn metadata(&self) -> EventMetadata {
        self.kind().metadata()
    }

    /// 序号由投递任务在过滤之后、投递之前填入。
    pub(crate) fn set_sequence(&mut self, value: u64) {
        match self {
            Self::RunStarted { sequence, .. }
            | Self::MailboxEnqueued { sequence, .. }
            | Self::MailboxRescheduled { sequence, .. }
            | Self::BoundaryDrained { sequence, .. }
            | Self::CompletionSealed { sequence, .. }
            | Self::ModelRequestStarted { sequence, .. }
            | Self::ModelRequestFinished { sequence, .. }
            | Self::ToolCallsSealed { sequence, .. }
            | Self::ToolCallSettled { sequence, .. }
            | Self::ToolBatchAborted { sequence, .. }
            | Self::ToolBatchStarted { sequence, .. }
            | Self::ToolBatchFinished { sequence, .. }
            | Self::CompactionStarted { sequence, .. }
            | Self::CompactionFinished { sequence, .. }
            | Self::RunCompleted { sequence, .. }
            | Self::RunStopped { sequence, .. }
            | Self::RunFailed { sequence, .. } => *sequence = value,
        }
    }
}

/// [`AgentEvent`] 的接收端：只收不带正文的元数据。
pub trait Observer: Send + Sync {
    /// 在投递任务上、投递每一条之前调用，可用于级别、类别和动态开关过滤。
    /// 返回 `false` 的事件不取号、不投递。
    fn enabled(&self, _metadata: EventMetadata) -> bool {
        true
    }

    fn observe(&self, event: &AgentEvent);
}

/// 携带正文的事件种类。
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContentEventKind {
    /// 发给模型的完整对话与工具声明。
    ModelRequestTranscript,
    /// 模型这一次返回的完整规范输出。
    ModelResponseOutput,
    /// 一个工具批次的完整结果内容。
    ToolBatchResults,
}

/// 携带正文的事件。
///
/// 每一条都和 [`AgentEvent`] 里的一条元数据事件成对：元数据那条便宜、可安全进日志，
/// 这条带真实内容。两者用 `run_id` 加 `request_index` 或 `batch_id` 关联，序号相邻。
///
/// **这些内容未经脱敏**，包括 [`crate::ToolResult::metadata`]——中断回执会有意清空
/// 它，这里不会，因为诊断出口的语义就是「原样」。落盘或外传前请自行脱敏。
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum ContentEvent {
    /// 与 [`AgentEvent::ModelRequestStarted`] 配对的完整请求内容。
    ///
    /// `ModelRequestStarted` 只说“这一次请求带了多少条记录”，回答不了
    /// “到底发出去了什么”——而那正是调试 agent 时最常要看的东西。
    ModelRequestTranscript {
        sequence: u64,
        run_id: RunId,
        request_index: u64,
        transcript: Vec<TranscriptItem>,
        function_tools: Vec<ToolDefinition>,
    },
    /// 与 [`AgentEvent::ModelRequestFinished`] 配对的完整模型输出。
    ///
    /// 自动开始的运行没有等待方，`TurnOutcome` 也只给最后一轮；中间每一轮说了什么，
    /// 只能从这里拿到。
    ModelResponseOutput {
        sequence: u64,
        run_id: RunId,
        request_index: u64,
        output: ModelOutput,
    },
    /// 与 [`AgentEvent::ToolBatchFinished`] 配对的完整结果。
    ToolBatchResults {
        sequence: u64,
        run_id: RunId,
        batch_id: ToolBatchId,
        results: ToolResultBatch,
    },
}

impl ContentEvent {
    /// 与 [`AgentEvent::sequence`] 同一个计数器、同一套保证。
    pub const fn sequence(&self) -> u64 {
        match self {
            Self::ModelRequestTranscript { sequence, .. }
            | Self::ModelResponseOutput { sequence, .. }
            | Self::ToolBatchResults { sequence, .. } => *sequence,
        }
    }

    pub const fn kind(&self) -> ContentEventKind {
        match self {
            Self::ModelRequestTranscript { .. } => ContentEventKind::ModelRequestTranscript,
            Self::ModelResponseOutput { .. } => ContentEventKind::ModelResponseOutput,
            Self::ToolBatchResults { .. } => ContentEventKind::ToolBatchResults,
        }
    }

    pub(crate) fn set_sequence(&mut self, value: u64) {
        match self {
            Self::ModelRequestTranscript { sequence, .. }
            | Self::ModelResponseOutput { sequence, .. }
            | Self::ToolBatchResults { sequence, .. } => *sequence = value,
        }
    }
}

/// [`ContentEvent`] 的接收端。
///
/// 不安装就**既不发出也不构造**——内核只在装了内容观察端时才克隆对话、输出与结果。
/// 装上自己的实现就是选择加入——所以这里的 `enabled` 默认返回 `true`：既然你装了，
/// 就说明你要。
pub trait ContentObserver: Send + Sync {
    /// 在投递任务上、投递每一条之前调用。返回 `false` 的事件不取号、不投递
    /// （内容已经克隆过了——按种类细分的过滤省不下那次克隆，只省下投递）。
    fn enabled(&self, _kind: ContentEventKind) -> bool {
        true
    }

    fn observe(&self, event: &ContentEvent);
}

#[derive(Default)]
pub struct NoopContentObserver;

impl ContentObserver for NoopContentObserver {
    fn enabled(&self, _kind: ContentEventKind) -> bool {
        false
    }

    fn observe(&self, _event: &ContentEvent) {}
}

/// 带运行关联信息的高频模型流事件。
///
/// 与 [`AgentEvent`] 不同，此结构可能包含模型正文、完整工具参数和服务商扩展字段，不应
/// 默认写入普通日志。需要持久化它的应用负责显式启用、限长和脱敏。
#[derive(Clone, Debug, PartialEq)]
pub struct ObservedModelStreamEvent {
    pub sequence: u64,
    pub run_id: RunId,
    pub request_index: u64,
    pub payload: ModelStreamObservation,
}

/// 流观察层的一次增量或明确终态。
#[derive(Clone, Debug, PartialEq)]
pub enum ModelStreamObservation {
    /// 在模型请求成功前都只是暂存内容。
    Delta(ModelStreamEvent),
    /// 模型适配器已经返回一份通过内核校验的完整响应。
    AttemptCommitted,
    /// 本次流草稿已经丢弃；同一运行可能随后通过恢复回执继续请求模型。
    AttemptAborted { error_kind: AgentErrorKind },
}

/// 模型正文和工具调用增量的同步观察端。
///
/// 它在观测投递任务上调用（不在网络流读取路径上），接收端应快速返回，通常只把事件转发
/// 到自己的队列。投递任务会隔离 `enabled` 和 `observe` 的 panic；观察端是否启用或失败都
/// 不影响可靠的工具转发。不安装就不构造增量事件。
pub trait StreamObserver: Send + Sync {
    fn enabled(&self) -> bool {
        true
    }

    fn observe(&self, event: &ObservedModelStreamEvent);
}

#[derive(Default)]
pub struct NoopStreamObserver;

impl StreamObserver for NoopStreamObserver {
    fn enabled(&self) -> bool {
        false
    }

    fn observe(&self, _event: &ObservedModelStreamEvent) {}
}

/// 用原子级别和类别掩码包装任意 Observer，可在运行期间无锁切换。
pub struct FilteredObserver {
    inner: Arc<dyn Observer>,
    level: AtomicU8,
    category_mask: AtomicU8,
}

impl FilteredObserver {
    pub fn new(inner: Arc<dyn Observer>, level: LevelFilter) -> Self {
        Self {
            inner,
            level: AtomicU8::new(level as u8),
            category_mask: AtomicU8::new(EventCategory::ALL_MASK),
        }
    }

    /// 生效点是投递任务处理**下一条**事件时的 `enabled` 判定，不是调用这里的那一刻：
    /// 已经排队、还没投递的事件也服从新级别。级别与类别是两个独立的原子值，同时改二者
    /// 不构成一份成对快照。
    pub fn set_level(&self, level: LevelFilter) {
        self.level.store(level as u8, Ordering::Relaxed);
    }

    /// 空切片表示关闭全部类别。
    pub fn set_categories(&self, categories: &[EventCategory]) {
        let mask = categories
            .iter()
            .fold(0, |mask, category| mask | category.mask());
        self.category_mask.store(mask, Ordering::Relaxed);
    }
}

impl Observer for FilteredObserver {
    fn enabled(&self, metadata: EventMetadata) -> bool {
        let level = LevelFilter::from_u8(self.level.load(Ordering::Relaxed));
        let categories = self.category_mask.load(Ordering::Relaxed);
        level.allows(metadata.level)
            && categories & metadata.category.mask() != 0
            && self.inner.enabled(metadata)
    }

    fn observe(&self, event: &AgentEvent) {
        self.inner.observe(event);
    }
}

#[derive(Default)]
pub struct NoopObserver;

impl Observer for NoopObserver {
    fn enabled(&self, _metadata: EventMetadata) -> bool {
        false
    }

    fn observe(&self, _event: &AgentEvent) {}
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    #[derive(Default)]
    struct RecordingObserver {
        kinds: Mutex<Vec<EventKind>>,
    }

    impl Observer for RecordingObserver {
        fn observe(&self, event: &AgentEvent) {
            self.kinds.lock().unwrap().push(event.kind());
        }
    }

    #[test]
    fn level_filter_orders_severity_and_supports_off() {
        assert!(!LevelFilter::Off.allows(EventLevel::Error));
        assert!(LevelFilter::Info.allows(EventLevel::Error));
        assert!(LevelFilter::Info.allows(EventLevel::Info));
        assert!(!LevelFilter::Info.allows(EventLevel::Debug));
        assert!(LevelFilter::Trace.allows(EventLevel::Trace));
    }

    #[test]
    fn filtered_observer_switches_level_and_categories_dynamically() {
        let recording = Arc::new(RecordingObserver::default());
        let filtered = FilteredObserver::new(recording, LevelFilter::Off);
        let model = EventKind::ModelRequestStarted.metadata();
        assert!(!filtered.enabled(model));

        filtered.set_level(LevelFilter::Debug);
        assert!(filtered.enabled(model));
        assert!(!filtered.enabled(EventKind::BoundaryDrained.metadata()));

        filtered.set_categories(&[EventCategory::Tools]);
        assert!(!filtered.enabled(model));
        assert!(filtered.enabled(EventKind::ToolBatchStarted.metadata()));
    }
}
