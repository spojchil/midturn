//! 会话配置：预算与截止时间。

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// 并发会话驱动器的资源预算与截止时间。
/// 预算按**上下文**计，不按运行计。
///
/// 运行可以自动开始——信箱来件而当前空闲就跑一轮——所以「每轮多少次」框不住任何
/// 东西：每开一轮就重新计数。真正需要封口的是整个会话消耗了多少。
///
/// 两类度量必须分开看，因为它们对压缩的反应相反：
///
/// - [`ContextBudget`] 描述**当前上下文有多大**。压缩能把它降下来，所以它有一个软阈值
///   （触发压缩）和一个硬上限（压完还超就停）。
/// - [`CumulativeBudget`] 描述**一共发生了多少**。压缩救不了它，所以只有硬上限。
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct SessionBudget {
    pub context: ContextBudget,
    pub cumulative: CumulativeBudget,
}

/// 当前上下文有多大——压缩能降下来的那一类。
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ContextBudget {
    /// 序列化字节超过此值时，在下一个安全边界压缩。
    pub compact_above_bytes: usize,
    /// 压缩之后仍然超过此值就停止运行。`None` 表示不设硬上限。
    pub max_bytes: Option<usize>,
    /// 服务商上次报告的输入 token 超过此值时压缩。
    ///
    /// 比字节估算准得多，但只有发过至少一次请求、且服务商报了 usage 才有值；
    /// 没有时回退到字节估算。
    pub compact_above_tokens: Option<u64>,
    /// 压缩之后仍然超过此值就停止运行。
    pub max_tokens: Option<u64>,
}

impl Default for ContextBudget {
    fn default() -> Self {
        Self {
            compact_above_bytes: 256 * 1024,
            max_bytes: None,
            compact_above_tokens: None,
            max_tokens: None,
        }
    }
}

/// 一共发生了多少——压缩救不了的那一类。
///
/// 计数属于**整个会话**，跨运行累加，并且写进 checkpoint：否则重开一次进程就能把
/// 限额清零，这个预算也就形同虚设。
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct CumulativeBudget {
    /// 整个会话允许发起的模型请求数。`None` 表示不限。
    pub max_model_requests: Option<u64>,
    /// 整个会话允许执行的工具调用数。
    pub max_tool_calls: Option<u64>,
}

/// 会话累计消耗。随 checkpoint 持久化。
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Consumption {
    /// 已经发起的模型请求数。
    pub model_requests: u64,
    /// 已经分发的工具调用数（按调用计，不按批次计）。
    pub tool_calls: u64,
    /// 服务商上次报告的输入 token，用作「当前上下文有多大」的主度量。
    pub last_input_tokens: Option<u64>,
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct SessionConfig {
    /// 上下文与累计两类预算。
    pub budget: SessionBudget,
    /// 中断工具批次的执行事实在下一次模型请求中使用的普通输入角色。
    ///
    /// 默认值为 `user`；内核不赋予该字符串固定语义，应用可按适配器能力改为
    /// `developer`、`operator` 或自定义角色。
    pub interrupted_tool_receipt_role: String,
    /// 单次运行允许从“已有工具执行事实的模型流中断”自动继续的最大次数。
    pub max_interrupted_tool_recoveries: u32,
    /// 等待投递的 mailbox 记录项上限。
    pub max_mailbox_items: usize,
    /// 等待投递的 mailbox JSON 估算字节上限。
    pub max_mailbox_bytes: usize,
    /// 单次模型请求的截止时间；`None` 表示由 adapter 自己控制。
    pub model_timeout: Option<Duration>,
    /// 单次工具批的截止时间。超时会作为结果未知进入 effect ledger，而不是安全重试。
    pub tool_timeout: Option<Duration>,
    /// 单次压缩的截止时间。
    pub compaction_timeout: Option<Duration>,
    /// 初始的自动开始开关（[`crate::AgentSession::set_auto_start`] 之后可以改）。
    ///
    /// 默认打开：空闲、信箱有触发内容就开跑——包括 `open` 之后的第一步（`Boot`）。
    /// 只想打开 checkpoint 看看、迁移、审计的程序把它关掉，否则第一次异步调用就可能把
    /// 模型请求发出去。它不落盘：表达的是「当前这个进程想不想干活」。
    pub auto_start: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            budget: SessionBudget::default(),
            interrupted_tool_receipt_role: "user".to_owned(),
            max_interrupted_tool_recoveries: 3,
            max_mailbox_items: 1_024,
            max_mailbox_bytes: 2 * 1024 * 1024,
            model_timeout: Some(Duration::from_secs(120)),
            tool_timeout: Some(Duration::from_secs(300)),
            compaction_timeout: Some(Duration::from_secs(30)),
            auto_start: true,
        }
    }
}
