//! 会话驱动器所需的输入输出端口。

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use crate::persistence::SessionId;
use crate::types::{
    AbortClassification, AgentError, IncrementalToolCall, ModelOutput, ModelUsage, RunId,
    ToolBatchAbortReason, ToolBatchStart, ToolCall, ToolCallBatch, ToolCallSlot, ToolDefinition,
    ToolResult, ToolResultBatch, TranscriptItem,
};

/// 装箱后的端口异步返回值使本包无需依赖 `async-trait` 宏。
pub type PortFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// 可压缩对话之外的受保护上下文。
///
/// [`PromptSource::base_context`] 在每次驱动器启动时重新读取（包括从 checkpoint 恢复同一
/// 运行），并置于模型请求最前面；它不会写入持久对话，也不会交给 [`Compaction`]。实现
/// 可以按启动动态生成全部受保护内容；返回的记录本身必须满足工具轮闭合约束。
///
/// # 这里放动态内容会打断前缀缓存
///
/// 本方法的返回值位于每次请求的**最前面**。服务商的前缀缓存按最长公共前缀命中，
/// 所以只要这里逐轮变化（当前时间、环境快照、剩余配额……），缓存就在位置 0 断掉，
/// 整个上下文每轮全额重算——长会话上这是数量级的成本差异。
///
/// 运行期间才产生的动态内容应当走信箱：它追加在对话末尾，前面的前缀保持不动。
pub trait PromptSource: Send + Sync {
    /// 默认返回空上下文。空系统提示词是一个正常选择，因此这里的默认实现表达的是
    /// “我确实不需要受保护前缀”，而不是“你忘了实现”。
    fn base_context(&self) -> Result<Vec<TranscriptItem>, AgentError> {
        Ok(Vec::new())
    }
}

/// 运行时在候选批次尚未 `commit` 时，主动上报单项调用的最终结果。
///
/// # 它解决什么
///
/// `commit` 一次性返回整批结果，所以在它返回之前，内核对批内每一项的进展一无所知。
/// 一批里工具 A 第 1 秒完成、B 第 60 秒完成时，这中间的 59 秒里发生断流或崩溃，
/// A 只能被归为 `OutcomeUnknown`——**而它本来是可知的**。上报把这段窗口关掉，
/// 直接减少未知态的数量。
///
/// # 契约
///
/// [`Self::settled`] 返回 `Ok` 的含义是**内核已经把这个结算持久化了**：此后无论断流、
/// 崩溃还是进程重开，该 slot 都不会再被判为结果未知。这也意味着它是一条不可撤回的
/// 终态事实——
///
/// - [`IncrementalToolBatch::abort`] 的报告里该 slot 无论写什么，内核都按已上报的
///   结算处理；
/// - [`IncrementalToolBatch::commit`] 返回的整批结果仍然是模型看到的内容，但不会
///   覆盖账本里已经上报的结算。
///
/// 上报是**可选**的。只在 `commit` 时才执行的运行时不需要它；只有提前执行、且批内
/// 单项完成时间差得开的运行时，才值得为此多付一次持久化写入。
pub trait ToolCallReporter: Send + Sync {
    /// 上报一个已经产生最终结果的单项调用。
    ///
    /// `slot` 必须是本批次中已经通过 [`IncrementalToolBatch::submit`] 交付过的槽位，
    /// `result.call_id` 必须与该槽位的调用一致，且同一槽位只能上报一次。
    fn settled<'a>(
        &'a self,
        slot: ToolCallSlot,
        result: ToolResult,
    ) -> PortFuture<'a, Result<(), AgentError>>;
}

/// 一个由增量工具运行时接管的候选批次。
///
/// “增量”描述调用信息的交付时机，不表示部分结果可以作为正常工具轮提交。一个候选批次有
/// 三个彼此独立的边界：
///
/// 1. [`Self::submit`] 逐项交付已经完整解析的单个调用；
/// 2. [`Self::calls_sealed`] 声明完整调用数组到此结束；
/// 3. [`Self::commit`] 仅在整个模型响应成功结束后发生，并返回完整结果批次。
///
/// `submit` 和 `calls_sealed` 的 future 成功只表示运行时已经可靠接管相应信息，不表示工具
/// 已经执行完成。立即执行、只缓存、并发、依赖排序和限流都由实现决定。`commit` 也只是模型
/// 尝试的协议提交点，不承诺 ACID 事务或自动回滚。
///
/// 如果模型响应没有成功结束，[`Self::abort`] 会取代 `commit`。即使 `submit` 返回 `Err`，
/// `abort` 的报告仍必须覆盖本次传入的调用，因为远程确认失败不能证明请求从未到达。实现
/// 必须停止确定尚未开始的工作、冻结已经接管的调用，并在 `abort` 返回后保证不再产生迟到
/// 执行或结果。只有提前执行远程或不可逆副作用的实现才通常需要持久幂等键、执行账本或状态
/// 查询；只在 `commit` 执行的本地实现可以简单缓存候选调用。
pub trait IncrementalToolBatch: Send {
    /// 接管一个已经完整解析的单项调用。
    ///
    /// 该调用之后是否还有其他调用尚未知；数组完整性只能由 [`Self::calls_sealed`] 声明。
    /// 返回 `Ok` 仅确认可靠接管，是否立刻开始执行由实现决定。
    fn submit<'a>(
        &'a mut self,
        call: IncrementalToolCall,
    ) -> PortFuture<'a, Result<(), AgentError>>;

    /// 声明工具调用数组已经传输完整。此操作可以早于整个模型响应完成，因此封口后仍不能
    /// 单凭此信号把候选批次当作一次成功的模型输出。
    fn calls_sealed<'a>(&'a mut self, call_count: u32) -> PortFuture<'a, Result<(), AgentError>>;

    /// 整个模型响应已经成功结束：可以跑了。
    ///
    /// 这只是一个信号。**结果不从这里返回**——每个槽的结果一律通过
    /// [`ToolCallReporter::settled`] 上报，那是结果唯一的通道。只在 commit 时才执行的
    /// 运行时在这里开始执行、逐槽调 `settled`；提前执行的运行时把它当 no-op。
    ///
    /// 返回 `Err` 表示运行时无法开始执行；驱动器会终止运行，不会把它伪装成可恢复的中断。
    fn commit<'a>(self: Box<Self>) -> PortFuture<'a, Result<(), AgentError>>
    where
        Self: 'a;

    /// 整个模型响应未能成功结束；停止未开始的工作并对剩下的槽分类。
    ///
    /// 报告只需覆盖**已经通过 [`Self::submit`] 交付、且尚未 `settled` 的槽**：每个是
    /// `CancelledBeforeStart`（确定没开始）还是 `OutcomeUnknown`（不知道）。已经知道结果
    /// 的槽在返回前用 [`ToolCallReporter::settled`] 上报——不要写进报告。返回后不得再执行
    /// 该候选批次中的工作；迟到的 `settled` 会撞上终态得到错误。
    ///
    /// # 这个 future 可能被强制取消
    ///
    /// 批次超时或会话被持久化隔离时，内核**不等它返回**：所有者任务被撤销，这个 future 和
    /// 持有它的 `Box<Self>` 一起被 drop（`submit` / `calls_sealed` / `commit` 同理）。所以：
    ///
    /// - 撤销远端工作的动作要么在 `Drop` 里完成/触发，要么做成一步不可分割的操作——
    ///   分多次 poll 才能完成的撤销可能在半途被丢下；
    /// - 被 drop 后仍在跑的后台工作是实现方自己的责任，内核已经按「结果未知」收口。
    fn abort<'a>(
        self: Box<Self>,
        reason: ToolBatchAbortReason,
    ) -> PortFuture<'a, Result<AbortClassification, AgentError>>
    where
        Self: 'a;
}

/// 可移植的工具目录，以及两种互斥的工具交付模式。
///
/// 默认模式在模型响应成功后把完整数组一次性交给 [`Self::dispatch`]。增量模式由
/// [`Self::begin_incremental`] 显式选择，随后把已经完整解析的单项调用及独立封口信号交给
/// [`IncrementalToolBatch`]；同一候选批次不会再通过 `dispatch` 重复下发。
///
/// 内核不执行调用，也不公开并行标志。工具查找、参数校验、审批、依赖排序、加锁、并发、
/// 限流、重试和远程转发均由实现负责。
pub trait ToolRuntime: Send + Sync {
    fn definitions(&self) -> Vec<ToolDefinition>;

    /// 整批分发是必需路径：即使实现选择了增量接管，`begin_incremental` 也可以按批返回
    /// `None` 退回这里，因此它不能缺席。缺少实现属于编译期错误，而不是运行期错误。
    fn dispatch<'a>(
        &'a self,
        batch: ToolCallBatch,
    ) -> PortFuture<'a, Result<ToolResultBatch, AgentError>>;

    /// 可选地接管一个增量候选批次。
    ///
    /// 默认返回 `None`，驱动器会缓存调用，并在模型响应成功后沿用
    /// [`ToolRuntime::dispatch`] 一次转发完整数组。返回 `Some` 后，该接收器负责本次候选
    /// 批次的逐项接管、封口以及最终 `commit` 或 `abort`。
    ///
    /// `reporter` 可以被克隆到运行时自己的任务里，用来在 `commit` 之前主动上报单项调用
    /// 已经结算；不需要这个能力就直接丢掉它。详见 [`ToolCallReporter`]。
    fn begin_incremental<'a>(
        &'a self,
        _batch: ToolBatchStart,
        _reporter: Arc<dyn ToolCallReporter>,
    ) -> PortFuture<'a, Result<Option<Box<dyn IncrementalToolBatch + 'a>>, AgentError>> {
        Box::pin(async { Ok(None) })
    }
}

/// 对话压缩策略。
///
/// 驱动器会在下一次模型请求前的安全边界以及运行收尾时按阈值调用它，同一运行中可能调用
/// 多次。此端口只接收可持久化对话，不接收 [`PromptSource::base_context`]；闭合的返回值会
/// 整体替换传入的对话，非法返回值则被丢弃并沿用原记录。需要调用模型的实现自行持有该依赖。
///
/// # 哪些记录不许动
///
/// effect 对账与中断恢复回执是**耐久事实**，必须逐项原样保留——不能概括、不能重排、
/// 不能折叠重复项、不能删除。用 [`crate::durable_fact_kind`] 把它们筛出来，
/// 不要自己去认 `kind` 字符串：内核加一种回执时，那样的代码不会收到任何提示。
///
/// 内核会在返回后独立校验这些记录的序列完全一致，不一致就丢弃整个压缩结果。
/// 所以这不是一条可以「尽力而为」的约定。
///
/// 另外注意压缩本身有代价：它改写对话，因此**服务商的前缀缓存会整体失效**，
/// 下一次请求全额重算。压缩省下的上下文要值回这笔钱。
pub trait Compaction: Send + Sync {
    fn compact<'a>(
        &'a self,
        conversation: &'a [TranscriptItem],
    ) -> PortFuture<'a, Result<Vec<TranscriptItem>, AgentError>>;
}

/// 传给服务商适配器的规范请求。
#[derive(Clone, Debug, PartialEq)]
pub struct ModelRequest {
    /// 稳定会话命名空间，可用于追踪、缓存隔离和 provider 幂等键。
    pub session_id: SessionId,
    /// 当前逻辑运行 ID。
    pub run_id: RunId,
    /// 当前运行中从 1 开始的模型请求序号。
    pub request_index: u64,
    pub transcript: Vec<TranscriptItem>,
    /// 仅包含可移植的本地分发函数。托管工具和服务商原生工具在模型适配器中配置，
    /// 不会进入 `ToolRuntime::dispatch`。
    pub function_tools: Vec<ToolDefinition>,
}

impl ModelRequest {
    pub fn new(
        session_id: SessionId,
        run_id: RunId,
        request_index: u64,
        transcript: Vec<TranscriptItem>,
        function_tools: Vec<ToolDefinition>,
    ) -> Self {
        Self {
            session_id,
            run_id,
            request_index,
            transcript,
            function_tools,
        }
    }

    #[cfg(all(test, any(feature = "openai", feature = "anthropic")))]
    pub(crate) fn test(
        transcript: Vec<TranscriptItem>,
        function_tools: Vec<ToolDefinition>,
    ) -> Self {
        Self::new(
            SessionId::new("adapter-test-session").expect("static test ID"),
            RunId::new("adapter-test-session/run/1"),
            1,
            transcript,
            function_tools,
        )
    }
}

/// 经模型适配器规范化的服务商响应。
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelResponse {
    pub output: ModelOutput,
    /// 仅用于诊断的服务商值；内核不会据此分支。
    pub finish_reason: Option<Value>,
    pub usage: Option<ModelUsage>,
}

/// 模型适配器在聚合一份最终响应期间发出的增量事件。
///
/// 正文与推理都是「开始 / 增量 / 结束」三段。三段式是必要的：只有增量的话，
/// 消费方永远不知道一段什么时候写完，只能等整个响应结束——那正是流式要避免的。
/// `part_index` 是**同一次响应内标识一段内容的稳定值**，由适配器分配；
/// 它只保证同段相等、异段不等，不保证连续，也不跨响应有意义。
///
/// 工具流同样区分三个边界：
///
/// - `ToolCallReady`：一个单项调用已经完整解析，可以按 slot 接管；它不表示这是末项。
/// - `ToolCallsSealed`：工具数组已经完整；它不表示正文、推理块或整个模型响应已经成功。
/// - [`Model::complete_stream`] 返回 `Ok`：最终规范响应已经成功聚合，此时增量批次才可
///   `commit`。
///
/// 例如 A 已发出 `ToolCallReady`、B 尚在传输时断流，运行时只会冻结 A；残缺的 B 不会被
/// 当作调用，未完成的模型草稿也不会成为正常对话轮次。
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum ModelStreamEvent {
    /// 一段正文开始。
    TextStart {
        part_index: u32,
    },
    TextDelta {
        part_index: u32,
        delta: String,
    },
    /// 这一段正文到此为止，不会再有增量。
    TextEnd {
        part_index: u32,
    },
    /// 一段推理开始。
    ///
    /// 推理内容和正文走同样的三段式，但**分开成对**：不少服务商会先推理再回答，
    /// 消费方需要能把两者区分开来渲染。内核不解释推理内容，也不把它写进规范对话——
    /// 精确回放所需的签名等信息由适配器保留在 `provider_data` 里。
    ReasoningStart {
        part_index: u32,
    },
    ReasoningDelta {
        part_index: u32,
        delta: String,
    },
    ReasoningEnd {
        part_index: u32,
    },
    ToolCallReady {
        slot: ToolCallSlot,
        call: ToolCall,
    },
    ToolCallsSealed {
        call_count: u32,
    },
}

/// 模型适配器向驱动器发送增量事件的异步接收端。
pub trait ModelStreamSink: Send {
    /// 返回 `Err` 表示当前模型 attempt 已经失效。adapter 应立即停止读取/发布并把该错误
    /// 从 `complete_stream` 返回；核心仍会粘滞记录首次失败，防止错误实现吞错后提交。
    fn emit<'a>(&'a mut self, event: ModelStreamEvent) -> PortFuture<'a, Result<(), AgentError>>;
}

/// 一次模型请求只对应一次语义完整的响应。重试、超时、流式聚合和服务商传输格式转换
/// 均由适配器负责。
pub trait Model: Send + Sync {
    /// 一次性生成完整响应，是必需路径：`complete_stream` 的默认实现直接委托它，
    /// 因此二者不能同时缺席。缺少实现属于编译期错误，而不是编译通过后在首次请求时才炸。
    fn complete<'a>(
        &'a self,
        request: ModelRequest,
    ) -> PortFuture<'a, Result<ModelResponse, AgentError>>;

    /// 流式生成并最终返回同一份完整响应。默认实现直接委托 [`Self::complete`] 且不产生
    /// 增量事件；支持流式协议的适配器覆盖此方法，在内部完成传输解析和最终聚合。
    ///
    /// 一旦向 `sink` 发布任何事件，实现不得把一个新的模型响应静默当作原响应重试并拼接；
    /// 否则正文草稿会重复，而 `ToolCallReady` 还可能已经触发外部影响。只有服务商明确支持
    /// 恢复同一个 response，且适配器能按稳定序号去重时，才可在当前 attempt 内续流；其他
    /// 中断必须返回 `Err`，由 session 冻结增量工具批。
    fn complete_stream<'a>(
        &'a self,
        request: ModelRequest,
        _sink: &'a mut dyn ModelStreamSink,
    ) -> PortFuture<'a, Result<ModelResponse, AgentError>> {
        self.complete(request)
    }
}
