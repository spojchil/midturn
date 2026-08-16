//! 持久化会话检查点与权威工具副作用账本。
//!
//! 持久化特意抽象为一个小型端口，而不与具体数据库绑定。调用方通过乐观并发的
//! 比较并交换（CAS）写入完整的 [`SessionCheckpoint`]；该端口既可由数据库事务、
//! 对象存储实现，也可使用本包提供的 [`InMemoryCheckpointStore`]。
//!
//! # 副作用投递协议
//!
//! 不可逆的工具调用必须按以下顺序跨越持久化边界：
//!
//! 1. 以 [`EffectState::Prepared`] 状态加入一个 [`EffectIntent`]，并持久化保存检查点；
//! 2. 将其转换为 [`EffectState::InFlight`]，再次持久化保存；
//! 3. 只有此时才调用外部工具；
//! 4. 记录返回的 [`EffectOutcome`]，并第三次持久化保存。
//!
//! 恢复出的 `Prepared` 副作用可以安全投递，因为账本表明该调用从未获得授权。
//! 恢复出的 `InFlight` 副作用或显式的 `OutcomeUnknown` 绝不能自动重复投递：必须通过
//! 工具运行时、幂等存储或人工操作员进行核对。`Settled` 与 `CancelledBeforeStart` 为终态。

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::mailbox::MailboxInput;
use crate::types::{
    validate_closed_transcript, ModelOutput, ModelUsage, RunId, ToolBatchAttemptId, ToolBatchId,
    ToolCall, ToolCallSlot, ToolResult, TranscriptItem,
};

#[cfg(feature = "sqlite")]
mod sqlite;
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteCheckpointStore;
mod store;
pub use store::{CheckpointStore, InMemoryCheckpointStore, PersistenceFuture};

/// 当前 crate 版本可识别的模式版本。
pub const CHECKPOINT_SCHEMA_VERSION: u32 = 3;

/// 由调用方持有的持久命名空间，用于运行、工具副作用、事件与检查点。
///
/// 应用重新打开会话时必须复用此值。它应由抗碰撞的持久 ID 生成器生成；内核刻意不使用
/// 进程内计数器自行构造此值。
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct SessionId(String);

impl SessionId {
    pub fn new(value: impl Into<String>) -> Result<Self, PersistenceError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(PersistenceError::invalid("empty_session_id"));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_inner(self) -> String {
        self.0
    }
}

impl AsRef<str> for SessionId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// 单调递增的检查点修订号，用作 CAS 令牌。
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct CheckpointRevision(u64);

impl CheckpointRevision {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// 独立于提供商所发调用 ID，用于标识一个副作用。
///
/// 其确定性表示采用长度定界，因此任意会话、运行或批次字符串都无法通过插入分隔符产生冲突。
/// 来源标签还会将已完成批次 ID 与流式尝试 ID 隔离在不同命名空间中。
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct EffectId(String);

impl EffectId {
    pub fn for_tool_call(
        session_id: &SessionId,
        run_id: &RunId,
        origin: &EffectOrigin,
        slot: ToolCallSlot,
    ) -> Self {
        let (origin_tag, origin_value) = match origin {
            EffectOrigin::CompletedBatch { batch_id } => ('b', batch_id.as_str()),
            EffectOrigin::StreamingAttempt { batch_attempt_id } => ('a', batch_attempt_id.as_str()),
        };
        let value = format!(
            "s{}:{}r{}:{}{}{}:{}i{}",
            session_id.as_str().len(),
            session_id.as_str(),
            run_id.as_str().len(),
            run_id.as_str(),
            origin_tag,
            origin_value.len(),
            origin_value,
            slot.get(),
        );
        Self(value)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for EffectId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for EffectId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// 副作用来自完整模型响应，还是在流式传输期间被接受。
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EffectOrigin {
    CompletedBatch {
        batch_id: ToolBatchId,
    },
    StreamingAttempt {
        batch_attempt_id: ToolBatchAttemptId,
    },
}

impl EffectOrigin {
    fn belongs_to_resume_point(&self, resume_from: &RunResumePoint) -> bool {
        match (self, resume_from) {
            (
                Self::CompletedBatch { batch_id },
                RunResumePoint::ToolBatch {
                    batch_id: expected, ..
                },
            ) => batch_id == expected,
            (
                Self::StreamingAttempt { batch_attempt_id },
                RunResumePoint::ToolBatch {
                    batch_id: expected, ..
                },
            ) => batch_attempt_id.as_str() == expected.as_str(),
            (
                Self::StreamingAttempt { batch_attempt_id },
                RunResumePoint::ModelAttempt {
                    batch_attempt_id: expected,
                    ..
                },
            ) => batch_attempt_id == expected,
            _ => false,
        }
    }
}

/// 一个预期工具副作用的持久、不可变描述。
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EffectIntent {
    pub id: EffectId,
    pub session_id: SessionId,
    pub run_id: RunId,
    pub origin: EffectOrigin,
    pub slot: ToolCallSlot,
    pub call: ToolCall,
}

impl EffectIntent {
    pub fn new(
        session_id: SessionId,
        run_id: RunId,
        origin: EffectOrigin,
        slot: ToolCallSlot,
        call: ToolCall,
    ) -> Result<Self, EffectLedgerError> {
        if run_id.as_str().is_empty() {
            return Err(EffectLedgerError::InvalidIntent {
                summary: "empty_run_id".to_owned(),
            });
        }
        if call.id.as_str().is_empty() {
            return Err(EffectLedgerError::InvalidIntent {
                summary: "empty_tool_call_id".to_owned(),
            });
        }
        let id = EffectId::for_tool_call(&session_id, &run_id, &origin, slot);
        Ok(Self {
            id,
            session_id,
            run_id,
            origin,
            slot,
            call,
        })
    }

    fn validate(&self) -> Result<(), EffectLedgerError> {
        if self.session_id.as_str().trim().is_empty() {
            return Err(EffectLedgerError::InvalidIntent {
                summary: "empty_session_id".to_owned(),
            });
        }
        if self.run_id.as_str().is_empty() {
            return Err(EffectLedgerError::InvalidIntent {
                summary: "empty_run_id".to_owned(),
            });
        }
        if self.call.id.as_str().is_empty() {
            return Err(EffectLedgerError::InvalidIntent {
                summary: "empty_tool_call_id".to_owned(),
            });
        }
        let expected =
            EffectId::for_tool_call(&self.session_id, &self.run_id, &self.origin, self.slot);
        if self.id != expected {
            return Err(EffectLedgerError::InvalidIntent {
                summary: "effect_id_does_not_match_intent".to_owned(),
            });
        }
        Ok(())
    }
}

/// 运行时对一次调用作出的权威结论。
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "status", content = "value", rename_all = "snake_case")]
pub enum EffectOutcome {
    Settled(ToolResult),
    CancelledBeforeStart,
    OutcomeUnknown { summary: String },
}

impl EffectOutcome {
    pub fn requires_reconciliation(&self) -> bool {
        matches!(self, Self::OutcomeUnknown { .. })
    }

    pub fn is_terminal(&self) -> bool {
        !self.requires_reconciliation()
    }
}

/// 副作用的持久化投递状态。
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum EffectState {
    /// 意图已持久化，且尚未跨越外部调用边界。
    Prepared,
    /// 调用已获授权。在此状态下崩溃会导致现实世界中的执行结果不明确。
    InFlight { delivery_attempt: u32 },
    /// 运行时已提供结果；`OutcomeUnknown` 仍需核对。
    Outcome { outcome: EffectOutcome },
}

impl EffectState {
    pub fn requires_reconciliation(&self) -> bool {
        matches!(
            self,
            Self::InFlight { .. }
                | Self::Outcome {
                    outcome: EffectOutcome::OutcomeUnknown { .. }
                }
        )
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Outcome {
                outcome: EffectOutcome::Settled(_) | EffectOutcome::CancelledBeforeStart
            }
        )
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EffectRecord {
    pub intent: EffectIntent,
    pub state: EffectState,
}

/// 从账本推导出的恢复决策，绝不会静默重试结果不明确的副作用。
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum EffectRecoveryAction {
    /// 从未跨越调用边界，因此驱动器可以投递此意图。
    DeliverPrepared { id: EffectId },
    /// 可以应用已记录的结果，无需再次调用工具。
    ApplySettled { id: EffectId, result: ToolResult },
    /// 运行时已确认调用从未开始。
    IgnoreCancelled { id: EffectId },
    /// 查询幂等存储或运行时，或请求操作员介入；绝不自动重复投递。
    Reconcile {
        id: EffectId,
        /// 对账端口需要的完整、稳定调用身份；其中不含 Authorization 等传输凭据。
        intent: EffectIntent,
        summary: String,
    },
}

/// 副作用意图与结果的权威集合。
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct EffectLedger {
    #[serde(default)]
    records: BTreeMap<EffectId, EffectRecord>,
}

impl EffectLedger {
    pub fn get(&self, id: &EffectId) -> Option<&EffectRecord> {
        self.records.get(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&EffectId, &EffectRecord)> {
        self.records.iter()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    // 账本的状态推进只能由内核驱动：把这些转换收成 crate 私有，
    // 才能让「结果未知的副作用不会被自动重做」成为类型系统保证，
    // 而不是一条使用者可以绕开的约定。
    /// 幂等地记录相同意图；拒绝将同一 ID 复用于不同工作。
    pub(crate) fn record_intent(&mut self, intent: EffectIntent) -> Result<(), EffectLedgerError> {
        intent.validate()?;
        match self.records.get(&intent.id) {
            Some(existing) if existing.intent == intent => Ok(()),
            Some(_) => Err(EffectLedgerError::IntentConflict {
                id: intent.id.clone(),
            }),
            None => {
                self.records.insert(
                    intent.id.clone(),
                    EffectRecord {
                        intent,
                        state: EffectState::Prepared,
                    },
                );
                Ok(())
            }
        }
    }

    /// 将意图标记为已获投递授权。
    ///
    /// 开始外部调用前，必须通过 CAS 持久化保存包含该意图的检查点。
    pub(crate) fn begin_delivery(&mut self, id: &EffectId) -> Result<u32, EffectLedgerError> {
        let record = self.record_mut(id)?;
        match &record.state {
            EffectState::Prepared => {
                record.state = EffectState::InFlight {
                    delivery_attempt: 1,
                };
                Ok(1)
            }
            _ => Err(Self::invalid_transition(
                id,
                &record.state,
                "begin_delivery",
            )),
        }
    }

    /// 取消已知尚未跨越调用边界的工作。
    pub(crate) fn cancel_prepared(&mut self, id: &EffectId) -> Result<(), EffectLedgerError> {
        let record = self.record_mut(id)?;
        match &record.state {
            EffectState::Prepared => {
                record.state = EffectState::Outcome {
                    outcome: EffectOutcome::CancelledBeforeStart,
                };
                Ok(())
            }
            _ => Err(Self::invalid_transition(
                id,
                &record.state,
                "cancel_prepared",
            )),
        }
    }

    /// 记录授权调用后返回的首个结果。
    pub(crate) fn resolve(
        &mut self,
        id: &EffectId,
        outcome: EffectOutcome,
    ) -> Result<(), EffectLedgerError> {
        let record = self.record_mut(id)?;
        Self::validate_outcome(id, &record.intent, &outcome)?;
        match &record.state {
            EffectState::InFlight { .. } => {
                record.state = EffectState::Outcome { outcome };
                Ok(())
            }
            _ => Err(Self::invalid_transition(id, &record.state, "resolve")),
        }
    }

    /// 在崩溃或出现 `OutcomeUnknown` 后，应用权威核对结果。
    ///
    /// 核对结果可以仍为未知，也可以结算或取消该副作用。已经进入终态的事实不可变，不能覆盖。
    pub(crate) fn reconcile(
        &mut self,
        id: &EffectId,
        outcome: EffectOutcome,
    ) -> Result<(), EffectLedgerError> {
        let record = self.record_mut(id)?;
        Self::validate_outcome(id, &record.intent, &outcome)?;
        let reconcilable = matches!(
            &record.state,
            EffectState::InFlight { .. }
                | EffectState::Outcome {
                    outcome: EffectOutcome::OutcomeUnknown { .. }
                }
        );
        if !reconcilable {
            return Err(Self::invalid_transition(id, &record.state, "reconcile"));
        }
        record.state = EffectState::Outcome { outcome };
        Ok(())
    }

    pub fn recovery_actions_for_run(&self, run_id: &RunId) -> Vec<EffectRecoveryAction> {
        self.records
            .values()
            .filter(|record| &record.intent.run_id == run_id)
            .map(|record| match &record.state {
                EffectState::Prepared => EffectRecoveryAction::DeliverPrepared {
                    id: record.intent.id.clone(),
                },
                EffectState::InFlight { delivery_attempt } => EffectRecoveryAction::Reconcile {
                    id: record.intent.id.clone(),
                    intent: record.intent.clone(),
                    summary: format!(
                        "effect_was_in_flight_at_recovery:delivery_attempt={delivery_attempt}"
                    ),
                },
                EffectState::Outcome {
                    outcome: EffectOutcome::Settled(result),
                } => EffectRecoveryAction::ApplySettled {
                    id: record.intent.id.clone(),
                    result: result.clone(),
                },
                EffectState::Outcome {
                    outcome: EffectOutcome::CancelledBeforeStart,
                } => EffectRecoveryAction::IgnoreCancelled {
                    id: record.intent.id.clone(),
                },
                EffectState::Outcome {
                    outcome: EffectOutcome::OutcomeUnknown { summary },
                } => EffectRecoveryAction::Reconcile {
                    id: record.intent.id.clone(),
                    intent: record.intent.clone(),
                    summary: summary.clone(),
                },
            })
            .collect()
    }

    fn validate_for_session(&self, session_id: &SessionId) -> Result<(), EffectLedgerError> {
        for (key, record) in &self.records {
            record.intent.validate()?;
            if key != &record.intent.id {
                return Err(EffectLedgerError::InvalidIntent {
                    summary: "effect_ledger_key_mismatch".to_owned(),
                });
            }
            if &record.intent.session_id != session_id {
                return Err(EffectLedgerError::InvalidIntent {
                    summary: "effect_belongs_to_another_session".to_owned(),
                });
            }
            if let EffectState::InFlight { delivery_attempt } = &record.state {
                if *delivery_attempt == 0 {
                    return Err(EffectLedgerError::InvalidIntent {
                        summary: "zero_delivery_attempt".to_owned(),
                    });
                }
            }
            if let EffectState::Outcome { outcome } = &record.state {
                Self::validate_outcome(key, &record.intent, outcome)?;
            }
        }
        Ok(())
    }

    fn validate_outcome(
        id: &EffectId,
        intent: &EffectIntent,
        outcome: &EffectOutcome,
    ) -> Result<(), EffectLedgerError> {
        match outcome {
            EffectOutcome::Settled(result) if result.call_id != intent.call.id => {
                Err(EffectLedgerError::InvalidOutcome {
                    id: id.clone(),
                    summary: "settled_result_call_id_mismatch".to_owned(),
                })
            }
            EffectOutcome::OutcomeUnknown { summary } if summary.trim().is_empty() => {
                Err(EffectLedgerError::InvalidOutcome {
                    id: id.clone(),
                    summary: "empty_unknown_outcome_summary".to_owned(),
                })
            }
            _ => Ok(()),
        }
    }

    fn record_mut(&mut self, id: &EffectId) -> Result<&mut EffectRecord, EffectLedgerError> {
        self.records
            .get_mut(id)
            .ok_or_else(|| EffectLedgerError::MissingEffect { id: id.clone() })
    }

    fn invalid_transition(
        id: &EffectId,
        state: &EffectState,
        operation: &'static str,
    ) -> EffectLedgerError {
        EffectLedgerError::InvalidTransition {
            id: id.clone(),
            state: format!("{state:?}"),
            operation: operation.to_owned(),
        }
    }
}

#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EffectLedgerError {
    InvalidIntent {
        summary: String,
    },
    IntentConflict {
        id: EffectId,
    },
    MissingEffect {
        id: EffectId,
    },
    InvalidOutcome {
        id: EffectId,
        summary: String,
    },
    InvalidTransition {
        id: EffectId,
        state: String,
        operation: String,
    },
}

impl std::fmt::Display for EffectLedgerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidIntent { summary } => write!(f, "invalid effect intent: {summary}"),
            Self::IntentConflict { id } => write!(f, "effect intent conflicts with {id}"),
            Self::MissingEffect { id } => write!(f, "effect does not exist: {id}"),
            Self::InvalidOutcome { id, summary } => {
                write!(f, "invalid effect outcome for {id}: {summary}")
            }
            Self::InvalidTransition {
                id,
                state,
                operation,
            } => write!(
                f,
                "invalid effect transition for {id}: {operation} from {state}"
            ),
        }
    }
}

impl std::error::Error for EffectLedgerError {}

/// 活跃运行中可重新启动的语义位置。
///
/// 检查点写在这些语义边界上，而不是通过序列化任意 Future 或栈帧来生成。
/// 驱动器根据此值重建其瞬时状态机。
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "point", rename_all = "snake_case")]
pub enum RunResumePoint {
    /// 已提交的对话记录可以用于发送新的模型请求。
    BeforeModelRequest,
    /// 模型流曾处于活跃状态，并可能已发出增量工具副作用。必须丢弃其未完成草稿；
    /// 首先从账本恢复引用的副作用。
    ModelAttempt {
        request_index: u64,
        batch_attempt_id: ToolBatchAttemptId,
        effect_ids: Vec<EffectId>,
    },
    /// 成功的模型输出声明了一个完整工具批次。待每个副作用结算后，
    /// 有序的副作用记录足以重建其结果批次。
    ToolBatch {
        batch_id: ToolBatchId,
        output: ModelOutput,
        effect_ids: Vec<EffectId>,
    },
    /// 最终模型输出已提交，但完成边界上的邮箱输入仍可引导后续流程。
    BeforeCompletion { output: ModelOutput },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RunSnapshot {
    pub run_id: RunId,
    pub resume_from: RunResumePoint,
    pub model_request_sequence: u64,
    pub tool_batch_sequence: u64,
    pub interrupted_tool_recoveries: u32,
    pub usage: Option<ModelUsage>,
}

/// 会话在某个语义检查点上的完整持久化映像。
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SessionCheckpoint {
    pub schema_version: u32,
    pub session_id: SessionId,
    pub run_sequence: u64,
    /// 始终是闭合且可提交的规范对话记录。未完成的模型或工具草稿仅存在于
    /// `active_run.resume_from` 与副作用账本中。
    /// 与内存会话状态共享同一份数据；构建 checkpoint 时只做引用计数。
    pub conversation: Arc<Vec<TranscriptItem>>,
    #[serde(default)]
    pub mailbox: Vec<MailboxInput>,
    /// 会话累计消耗。不持久化的话，重开一次进程就能把限额清零。
    #[serde(default)]
    pub consumption: crate::session::Consumption,
    pub active_run: Option<RunSnapshot>,
    #[serde(default)]
    pub effects: EffectLedger,
}

impl SessionCheckpoint {
    pub fn empty(session_id: SessionId) -> Self {
        Self {
            schema_version: CHECKPOINT_SCHEMA_VERSION,
            session_id,
            run_sequence: 0,
            conversation: Arc::new(Vec::new()),
            mailbox: Vec::new(),
            consumption: crate::session::Consumption::default(),
            active_run: None,
            effects: EffectLedger::default(),
        }
    }

    pub fn validate(&self) -> Result<(), PersistenceError> {
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION {
            return Err(PersistenceError::UnsupportedSchema {
                found: self.schema_version,
                supported: CHECKPOINT_SCHEMA_VERSION,
            });
        }
        if self.session_id.as_str().trim().is_empty() {
            return Err(PersistenceError::invalid("empty_session_id"));
        }
        validate_closed_transcript(&self.conversation)
            .map_err(|error| PersistenceError::invalid(error.summary))?;
        for input in &self.mailbox {
            validate_closed_transcript(&input.items)
                .map_err(|error| PersistenceError::invalid(error.summary))?;
        }
        self.effects
            .validate_for_session(&self.session_id)
            .map_err(|error| PersistenceError::invalid(error.to_string()))?;

        if let Some(run) = &self.active_run {
            if run.run_id.as_str().is_empty() {
                return Err(PersistenceError::invalid("empty_active_run_id"));
            }
            if self.run_sequence == 0 {
                return Err(PersistenceError::invalid(
                    "active_run_requires_nonzero_run_sequence",
                ));
            }
            let expected_run_id = format!("{}/run/{}", self.session_id, self.run_sequence);
            if run.run_id.as_str() != expected_run_id {
                return Err(PersistenceError::invalid(
                    "active_run_id_does_not_match_session_sequence",
                ));
            }
            self.validate_resume_point(run)?;
        }
        Ok(())
    }

    fn validate_resume_point(&self, run: &RunSnapshot) -> Result<(), PersistenceError> {
        if run.tool_batch_sequence > run.model_request_sequence {
            return Err(PersistenceError::invalid(
                "tool_sequence_exceeds_model_sequence",
            ));
        }
        if run.model_request_sequence == 0 && (run.tool_batch_sequence != 0 || run.usage.is_some())
        {
            return Err(PersistenceError::invalid("unused_run_contains_progress"));
        }
        match &run.resume_from {
            RunResumePoint::BeforeModelRequest => {}
            RunResumePoint::BeforeCompletion { output } => {
                if run.model_request_sequence == 0 {
                    return Err(PersistenceError::invalid(
                        "completion_without_model_request",
                    ));
                }
                if !output.tool_calls.is_empty() {
                    return Err(PersistenceError::invalid("completion_contains_tool_calls"));
                }
                if self.conversation.last() != Some(&TranscriptItem::ModelOutput(output.clone())) {
                    return Err(PersistenceError::invalid(
                        "completion_output_not_at_conversation_tail",
                    ));
                }
            }
            RunResumePoint::ModelAttempt {
                request_index,
                batch_attempt_id,
                ..
            } => {
                if *request_index == 0 || *request_index != run.model_request_sequence {
                    return Err(PersistenceError::invalid(
                        "model_attempt_request_sequence_mismatch",
                    ));
                }
                if batch_attempt_id.as_str().is_empty() {
                    return Err(PersistenceError::invalid("empty_batch_attempt_id"));
                }
            }
            RunResumePoint::ToolBatch {
                batch_id, output, ..
            } => {
                if run.model_request_sequence == 0 || run.tool_batch_sequence == 0 {
                    return Err(PersistenceError::invalid(
                        "tool_batch_requires_nonzero_sequences",
                    ));
                }
                if batch_id.as_str().is_empty() || output.tool_calls.is_empty() {
                    return Err(PersistenceError::invalid("empty_tool_batch"));
                }
                let mut call_ids = BTreeSet::new();
                if output
                    .tool_calls
                    .iter()
                    .any(|call| call.id.as_str().is_empty() || !call_ids.insert(&call.id))
                {
                    return Err(PersistenceError::invalid(
                        "invalid_or_duplicate_tool_call_id",
                    ));
                }
            }
        }

        let effect_ids = match &run.resume_from {
            RunResumePoint::BeforeModelRequest | RunResumePoint::BeforeCompletion { .. } => &[][..],
            RunResumePoint::ModelAttempt { effect_ids, .. }
            | RunResumePoint::ToolBatch { effect_ids, .. } => effect_ids,
        };

        let mut seen = BTreeSet::new();
        for id in effect_ids {
            if !seen.insert(id) {
                return Err(PersistenceError::invalid(
                    "duplicate_effect_in_resume_point",
                ));
            }
            let record = self
                .effects
                .get(id)
                .ok_or_else(|| PersistenceError::invalid("resume_effect_not_in_ledger"))?;
            if record.intent.run_id != run.run_id {
                return Err(PersistenceError::invalid(
                    "resume_effect_belongs_to_another_run",
                ));
            }
            if !record
                .intent
                .origin
                .belongs_to_resume_point(&run.resume_from)
            {
                return Err(PersistenceError::invalid("resume_effect_origin_mismatch"));
            }
        }

        for (id, record) in self.effects.iter() {
            if record.intent.run_id == run.run_id
                && matches!(record.state, EffectState::Prepared)
                && !seen.contains(id)
            {
                return Err(PersistenceError::invalid("unreferenced_nonterminal_effect"));
            }
        }

        if let RunResumePoint::ToolBatch {
            output, effect_ids, ..
        } = &run.resume_from
        {
            if output.tool_calls.len() != effect_ids.len() {
                return Err(PersistenceError::invalid(
                    "tool_batch_effect_count_mismatch",
                ));
            }
            for (slot, (call, effect_id)) in output.tool_calls.iter().zip(effect_ids).enumerate() {
                let record = self.effects.get(effect_id).expect("checked above");
                if record.intent.slot.get() as usize != slot || record.intent.call != *call {
                    return Err(PersistenceError::invalid(
                        "tool_batch_effect_order_mismatch",
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StoredCheckpoint {
    pub revision: CheckpointRevision,
    pub checkpoint: SessionCheckpoint,
}

#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PersistenceError {
    InvalidCheckpoint {
        summary: String,
    },
    UnsupportedSchema {
        found: u32,
        supported: u32,
    },
    Conflict {
        session_id: SessionId,
        expected: Option<CheckpointRevision>,
        actual: Option<CheckpointRevision>,
    },
    RevisionExhausted {
        session_id: SessionId,
    },
    Backend {
        summary: String,
    },
}

impl PersistenceError {
    fn invalid(summary: impl Into<String>) -> Self {
        Self::InvalidCheckpoint {
            summary: summary.into(),
        }
    }
}

impl std::fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCheckpoint { summary } => write!(f, "invalid checkpoint: {summary}"),
            Self::UnsupportedSchema { found, supported } => write!(
                f,
                "unsupported checkpoint schema {found}; this build supports {supported}"
            ),
            Self::Conflict {
                session_id,
                expected,
                actual,
            } => write!(
                f,
                "checkpoint CAS conflict for {session_id}: expected {expected:?}, actual {actual:?}"
            ),
            Self::RevisionExhausted { session_id } => {
                write!(f, "checkpoint revision exhausted for {session_id}")
            }
            Self::Backend { summary } => write!(f, "checkpoint backend failed: {summary}"),
        }
    }
}

impl std::error::Error for PersistenceError {}

#[cfg(test)]
mod tests;
