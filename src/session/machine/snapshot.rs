//! `Machine` → `SessionCheckpoint` 的投影。
//!
//! 线路格式沿用现有的 `RunSnapshot` / `RunResumePoint`：内存里的真相是 `Phase`，盘上是它的
//! 投影，就像 serde 一样。这样 `persistence.rs` 与 SQLite 后端一行不改。
//!
//! 运行中的 checkpoint 用 `turn` 的可提交后缀作 conversation——排空进 turn 的信箱内容、
//! 回执、工具结果都在那里，`core.conversation` 只在收尾时更新。

use crate::persistence::{EffectId, EffectOrigin, RunResumePoint, RunSnapshot, SessionCheckpoint};
use crate::types::ToolBatchId;

use super::{AfterCompaction, AttemptMode, CoreOutcome, Machine, Phase, Run};

impl Machine {
    /// 当前状态的完整持久映像。驱动循环在 `Persist` 时调它。
    pub(crate) fn checkpoint(&self) -> SessionCheckpoint {
        let core = &self.core;
        let mut checkpoint = SessionCheckpoint::empty(core.session_id.clone());
        checkpoint.run_sequence = core.run_seq;
        checkpoint.mailbox = core.mailbox.snapshot();
        checkpoint.consumption = core.consumption.clone();
        checkpoint.effects = core.effects.clone();
        match &self.run {
            Some(run) => {
                checkpoint.conversation = std::sync::Arc::new(
                    run.turn.committable_transcript()[run.protected_prefix_len..].to_vec(),
                );
                checkpoint.active_run = Some(self.run_snapshot(run));
            }
            None => {
                checkpoint.conversation = core.conversation.clone();
                checkpoint.active_run = core.recovered.clone();
            }
        }
        checkpoint
    }

    fn run_snapshot(&self, run: &Run) -> RunSnapshot {
        let resume_from = match &run.phase {
            Phase::ReadyToCall => before_model_or_completion(run),
            // 模型尝试进行中；已交付的槽以 effect_ids 引用。重开时 InFlight → 对账。
            Phase::CallingModel(attempt) | Phase::Aborting { attempt, .. } => {
                RunResumePoint::ModelAttempt {
                    request_index: attempt.request_index,
                    batch_attempt_id: attempt.id.clone(),
                    effect_ids: self.attempt_effect_ids(run, &attempt.id),
                }
            }
            Phase::WaitingTools(attempt) => match (attempt.mode, &attempt.output) {
                (AttemptMode::Incremental, Some(output)) => RunResumePoint::ToolBatch {
                    batch_id: ToolBatchId::from(&attempt.id),
                    output: output.clone(),
                    effect_ids: self.attempt_effect_ids(run, &attempt.id),
                },
                // 运行时还没回答要不要接管，什么都没交出去：当作模型尝试，重开后重发请求。
                _ => RunResumePoint::ModelAttempt {
                    request_index: attempt.request_index,
                    batch_attempt_id: attempt.id.clone(),
                    effect_ids: Vec::new(),
                },
            },
            Phase::Dispatching { attempt } => {
                let batch_id = ToolBatchId::from(&attempt.id);
                let calls = attempt
                    .output
                    .as_ref()
                    .map(|o| o.tool_calls.len())
                    .unwrap_or(0);
                RunResumePoint::ToolBatch {
                    batch_id: batch_id.clone(),
                    output: attempt.output.clone().unwrap_or_default(),
                    effect_ids: (0..calls)
                        .map(|index| {
                            EffectId::for_tool_call(
                                &self.core.session_id,
                                &run.id,
                                &EffectOrigin::CompletedBatch {
                                    batch_id: batch_id.clone(),
                                },
                                crate::types::ToolCallSlot::new(index as u32),
                            )
                        })
                        .collect(),
                }
            }
            Phase::Compacting { then, .. } => match then {
                AfterCompaction::Drain(_) => before_model_or_completion(run),
                AfterCompaction::Finish(CoreOutcome::Completed { output, .. }) => {
                    RunResumePoint::BeforeCompletion {
                        output: output.clone(),
                    }
                }
                AfterCompaction::Finish(_) => before_model_or_completion(run),
            },
        };
        RunSnapshot {
            run_id: run.id.clone(),
            resume_from,
            model_request_sequence: run.stats.model_requests,
            tool_batch_sequence: run.turn.tool_batch_sequence(),
            interrupted_tool_recoveries: u32::try_from(run.stats.interrupted_tool_recoveries)
                .unwrap_or(u32::MAX),
            usage: run.turn.usage().cloned(),
        }
    }

    fn attempt_effect_ids(
        &self,
        run: &Run,
        attempt_id: &crate::types::ToolBatchAttemptId,
    ) -> Vec<EffectId> {
        let mut records: Vec<_> = self
            .core
            .effects
            .iter()
            .filter(|(_, record)| {
                record.intent.run_id == run.id
                    && matches!(&record.intent.origin,
                        EffectOrigin::StreamingAttempt { batch_attempt_id } if batch_attempt_id == attempt_id)
            })
            .map(|(id, record)| (record.intent.slot, id.clone()))
            .collect();
        records.sort_by_key(|(slot, _)| *slot);
        records.into_iter().map(|(_, id)| id).collect()
    }
}

/// 停在边界上：有待提交的最终输出就是完成边界，否则是模型请求边界。
fn before_model_or_completion(run: &Run) -> RunResumePoint {
    match run.turn.pending_completion_output() {
        Some(output) => RunResumePoint::BeforeCompletion {
            output: output.clone(),
        },
        None => RunResumePoint::BeforeModelRequest,
    }
}
