//! 中止：断流 / 取消 / 运行时拒绝之后的分类、回执与未知态；批次超时。

use crate::events::{AgentEvent, EventKind, ModelStreamObservation, RunStage};
use crate::persistence::{EffectOutcome, EffectState};
use crate::types::{
    AbortClassification, AbortedSlot, AgentError, AgentErrorKind, ContentPart, DurableFactKind,
    InputMessage, InterruptedToolBatchReceipt, InterruptedToolCallOutcome,
    InterruptedToolCallReceipt, RunId, ToolBatchAbortReason, ToolCallBatch, ToolCallSlot,
};

use super::super::{
    Attempt, AttemptMode, Core, CoreOutcome, Effect, Next, Phase, Run, RuntimeCall,
};
use super::ledger_failure;

impl Core {
    /// 把这个 attempt 账本里的事实（已结算的结果 + 结果未知的槽）写成一条中断回执进历史。
    /// 确定未开始的没有副作用，不进回执。返回 `(settled, unknown)` 计数。
    ///
    /// **无论运行接下来怎么收口（继续 / 失败 / 取消），账本里的事实都必须先有一个下一轮
    /// 看得见的出口**——账本是唯一真相，但模型看的是对话。
    #[allow(clippy::result_large_err)]
    fn write_interrupted_receipt(
        &mut self,
        run: &mut Run,
        attempt: &Attempt,
    ) -> Result<(usize, usize), CoreOutcome> {
        let (mut settled, mut unknown) = (0usize, 0usize);
        let mut receipts = Vec::new();
        for (slot, record) in self.attempt_records(&run.id, &attempt.id) {
            let outcome = match record.state {
                EffectState::Outcome {
                    outcome: EffectOutcome::Settled(result),
                } => {
                    settled += 1;
                    InterruptedToolCallOutcome::Settled(result)
                }
                EffectState::Outcome {
                    outcome: EffectOutcome::OutcomeUnknown { summary },
                } => {
                    unknown += 1;
                    InterruptedToolCallOutcome::OutcomeUnknown { summary }
                }
                _ => continue,
            };
            receipts.push(InterruptedToolCallReceipt {
                slot,
                call_id: record.intent.call.id,
                name: record.intent.call.name,
                arguments: record.intent.call.arguments,
                outcome,
            });
        }
        if receipts.is_empty() {
            return Ok((0, 0));
        }
        let receipt = InterruptedToolBatchReceipt {
            batch_attempt_id: attempt.id.clone(),
            calls: receipts,
        };
        let message = self
            .interrupted_receipt_message(&receipt)
            .map_err(|error| CoreOutcome::Failed {
                error,
                stage: RunStage::Boundary,
            })?;
        run.turn
            .recover_model_attempt(message)
            .map_err(|error| CoreOutcome::Failed {
                error,
                stage: RunStage::Boundary,
            })?;
        self.touch();
        Ok((settled, unknown))
    }

    /// 运行时不可用，没有人可以再被问「哪些开始了」：commit 已经把批次对象拿走（之后没有
    /// `abort` 可问）、或者所有者任务没了。仍在途的一律未知，事实写进历史，本轮到此为止：
    /// 取消 → Stopped；有未知 → 对账要求；全已结算 → 仍然失败——运行时自相矛盾，
    /// 不拿一批「它自己说不算」的结果去问模型。不伪装成可恢复的中断。
    pub(crate) fn conclude_without_runtime(
        &mut self,
        run: &mut Run,
        attempt: Attempt,
        cause: &str,
        error: &AgentError,
        out: &mut Vec<Effect>,
    ) -> Next {
        out.push(Effect::Release {
            attempt: attempt.id.clone(),
        });
        self.mark_attempt_unknown(&run.id, &attempt, &format!("{cause}:{}", error.summary));
        let (settled, unknown) = match self.write_interrupted_receipt(run, &attempt) {
            Ok(counts) => counts,
            Err(outcome) => return Next::Finish(outcome),
        };
        let run_id = run.id.clone();
        let batch_attempt_id = attempt.id.clone();
        self.emit(out, EventKind::ToolBatchAborted, |sequence| {
            AgentEvent::ToolBatchAborted {
                sequence,
                run_id,
                batch_attempt_id,
                settled_count: settled,
                cancelled_count: 0,
                unknown_count: unknown,
            }
        });
        if error.kind == AgentErrorKind::Cancelled {
            return Next::Finish(CoreOutcome::Stopped);
        }
        let error = if unknown > 0 {
            AgentError::new(
                AgentErrorKind::ReconciliationRequired,
                format!(
                    "tool_batch_outcome_requires_reconciliation:{}",
                    error.summary
                ),
            )
        } else {
            AgentError::new(
                AgentErrorKind::ToolDispatch,
                format!("tool_batch_{cause}:{}", error.summary),
            )
        };
        Next::Finish(CoreOutcome::Failed {
            error,
            stage: RunStage::Tools,
        })
    }

    /// `Aborting` 期间收到 commit 的回执：所有者刚刚消费了批次对象并退出，我们发出去的
    /// `Abort` 没人接。这条回执就是「运行时说了什么」的全部——仍在途的一律未知。
    pub(crate) fn commit_acked_while_aborting(
        &mut self,
        run: &mut Run,
        result: Result<(), AgentError>,
        out: &mut Vec<Effect>,
    ) -> Next {
        let Phase::Aborting { attempt, error, .. } =
            std::mem::replace(&mut run.phase, Phase::ReadyToCall)
        else {
            unreachable!("checked by caller")
        };
        match result {
            Ok(()) => {
                self.conclude_without_runtime(run, attempt, "aborted_after_commit", &error, out)
            }
            Err(commit_error) => {
                // commit 失败 + 正在中止：中止的原因（比如取消）决定终态，commit 的失败摘要
                // 进未知态的说明。
                let cause = format!("commit_failed:{}", commit_error.summary);
                self.conclude_without_runtime(run, attempt, &cause, &error, out)
            }
        }
    }

    /// 从 CallingModel / WaitingTools 进入中止路径。
    ///
    /// 增量模式：发 `Runtime::Abort`，转 `Aborting` 等报告。
    /// 其他模式：什么都没交出去，不必问运行时——取消 → Stopped，否则 → Failed。
    pub(crate) fn begin_abort(
        &mut self,
        run: &mut Run,
        mut attempt: Attempt,
        reason: ToolBatchAbortReason,
        error: AgentError,
        out: &mut Vec<Effect>,
    ) -> Next {
        let error_kind = error.kind;
        self.emit_stream(out, &run.id, attempt.request_index, || {
            ModelStreamObservation::AttemptAborted { error_kind }
        });
        attempt.failure.get_or_insert(error.clone());
        match attempt.mode {
            AttemptMode::Incremental if attempt.committed => {
                // commit 之后运行时拿着批次对象自己在跑，没有 `abort` 可问、也没有所有者
                // 任务在等指令：仍在途的一律未知。
                self.conclude_without_runtime(run, attempt, "aborted_after_commit", &error, out)
            }
            AttemptMode::Incremental => {
                out.push(Effect::Runtime {
                    attempt: attempt.id.clone(),
                    call: RuntimeCall::Abort(reason),
                });
                self.arm_batch_timer(&mut attempt, out);
                run.phase = Phase::Aborting {
                    attempt,
                    reason,
                    error,
                };
                // 从 WaitingTools 进来时快照投影会从 ToolBatch 变成 ModelAttempt：必须落盘，
                // 否则崩溃重开会把一次被中止的尝试当成可重建的正常工具轮。
                self.touch();
                Next::Park
            }
            AttemptMode::Undecided | AttemptMode::Batch => {
                if error.kind == AgentErrorKind::Cancelled {
                    Next::Finish(CoreOutcome::Stopped)
                } else {
                    let stage = if reason == ToolBatchAbortReason::ToolRuntimeRejected {
                        RunStage::Tools
                    } else {
                        RunStage::Model
                    };
                    Next::Finish(CoreOutcome::Failed { error, stage })
                }
            }
        }
    }

    pub(crate) fn batch_aborted(
        &mut self,
        run: &mut Run,
        result: Result<AbortClassification, AgentError>,
        out: &mut Vec<Effect>,
    ) -> Next {
        let Phase::Aborting {
            attempt,
            reason,
            error,
        } = std::mem::replace(&mut run.phase, Phase::ReadyToCall)
        else {
            unreachable!("checked by caller")
        };
        out.push(Effect::Release {
            attempt: attempt.id.clone(),
        });

        let classification = match result {
            Ok(classification) => classification,
            Err(abort_error) => {
                // 运行时连报告都给不出来：仍 InFlight 的一律未知，本轮以对账要求收尾。
                self.mark_attempt_unknown(
                    &run.id,
                    &attempt,
                    &format!("abort_failed:{}", abort_error.summary),
                );
                return Next::Finish(CoreOutcome::Failed {
                    error: AgentError::new(
                        AgentErrorKind::ReconciliationRequired,
                        format!(
                            "tool_batch_abort_requires_reconciliation:{}",
                            abort_error.summary
                        ),
                    ),
                    stage: RunStage::Tools,
                });
            }
        };

        let (mut cancelled, mut unknown) = (0usize, 0usize);
        for slot in self.in_flight_slots(&run.id, &attempt.id) {
            let outcome = match classification.slots.get(&slot) {
                Some(AbortedSlot::CancelledBeforeStart) => {
                    cancelled += 1;
                    EffectOutcome::CancelledBeforeStart
                }
                Some(AbortedSlot::OutcomeUnknown { summary }) => {
                    unknown += 1;
                    EffectOutcome::OutcomeUnknown {
                        summary: summary.clone(),
                    }
                }
                None => {
                    unknown += 1;
                    EffectOutcome::OutcomeUnknown {
                        summary: "runtime_omitted_slot_from_abort_report".to_owned(),
                    }
                }
            };
            let id = self.attempt_effect_id(&run.id, &attempt.id, slot);
            if let Err(ledger) = self.effects.resolve(&id, outcome) {
                return Next::Finish(ledger_failure(ledger));
            }
        }
        self.touch();

        // 回执：已结算的结果 + 结果未知的槽。确定未开始的没有副作用，不进回执。
        // 先写事实，再决定怎么收口——取消也一样：跑过的东西不能因为用户按了停就从历史里消失。
        let (settled, unknown_in_receipt) = match self.write_interrupted_receipt(run, &attempt) {
            Ok(counts) => counts,
            Err(outcome) => return Next::Finish(outcome),
        };
        debug_assert_eq!(unknown, unknown_in_receipt);
        let run_id = run.id.clone();
        let batch_attempt_id = attempt.id.clone();
        self.emit(out, EventKind::ToolBatchAborted, |sequence| {
            AgentEvent::ToolBatchAborted {
                sequence,
                run_id,
                batch_attempt_id,
                settled_count: settled,
                cancelled_count: cancelled,
                unknown_count: unknown,
            }
        });

        if error.kind == AgentErrorKind::Cancelled {
            return Next::Finish(CoreOutcome::Stopped);
        }
        if settled == 0 && unknown == 0 {
            // 没有事实可交代：这就是一次失败的模型请求。内核不重试。
            let stage = if reason == ToolBatchAbortReason::ToolRuntimeRejected {
                RunStage::Tools
            } else {
                RunStage::Model
            };
            return Next::Finish(CoreOutcome::Failed { error, stage });
        }
        // 事实已经写进历史。接下来还继不继续，看这次中断的性质：
        if unknown > 0 {
            // 有结果不明的副作用：先对账，再谈下一次模型请求。
            return Next::Finish(CoreOutcome::Failed {
                error: AgentError::new(
                    AgentErrorKind::ReconciliationRequired,
                    "interrupted_tool_outcome_requires_reconciliation",
                ),
                stage: RunStage::Tools,
            });
        }
        if reason == ToolBatchAbortReason::ToolRuntimeRejected {
            // 运行时拒绝不是模型流的瞬时故障，自动再来一次多半还是拒绝。
            return Next::Finish(CoreOutcome::Failed {
                error,
                stage: RunStage::Tools,
            });
        }
        run.stats.interrupted_tool_recoveries =
            run.stats.interrupted_tool_recoveries.saturating_add(1);
        if run.stats.interrupted_tool_recoveries
            > u64::from(self.config.max_interrupted_tool_recoveries)
        {
            return Next::Finish(CoreOutcome::Failed {
                error: AgentError::new(
                    AgentErrorKind::Model,
                    "interrupted_tool_recovery_limit_exceeded",
                ),
                stage: RunStage::Model,
            });
        }
        run.phase = Phase::ReadyToCall;
        Next::Pump
    }

    /// 本 attempt 下仍 InFlight 的槽全部标为未知。
    pub(crate) fn mark_attempt_unknown(
        &mut self,
        run_id: &RunId,
        attempt: &Attempt,
        summary: &str,
    ) {
        for slot in self.in_flight_slots(run_id, &attempt.id) {
            let id = self.attempt_effect_id(run_id, &attempt.id, slot);
            let _ = self.effects.resolve(
                &id,
                EffectOutcome::OutcomeUnknown {
                    summary: summary.to_owned(),
                },
            );
        }
        self.touch();
    }

    /// 整批下仍 InFlight 的槽全部标为未知。
    pub(crate) fn mark_batch_unknown(
        &mut self,
        run_id: &RunId,
        batch: &ToolCallBatch,
        summary: &str,
    ) {
        for index in 0..batch.calls.len() {
            let id = self.batch_effect_id(run_id, &batch.batch_id, ToolCallSlot::new(index as u32));
            let _ = self.effects.resolve(
                &id,
                EffectOutcome::OutcomeUnknown {
                    summary: summary.to_owned(),
                },
            );
        }
        self.touch();
    }

    /// 等待阶段超时：仍 InFlight 的一律未知；本轮以对账要求收尾。
    pub(crate) fn batch_timed_out(&mut self, run: &mut Run, out: &mut Vec<Effect>) -> Next {
        match std::mem::replace(&mut run.phase, Phase::ReadyToCall) {
            Phase::WaitingTools(attempt) | Phase::Aborting { attempt, .. } => {
                out.push(Effect::Release {
                    attempt: attempt.id.clone(),
                });
                self.mark_attempt_unknown(&run.id, &attempt, "tool_batch_timeout");
            }
            Phase::Dispatching { attempt } => {
                if let Some(batch) = attempt.batch(&run.id) {
                    out.push(Effect::AbortDispatch {
                        batch_id: batch.batch_id.clone(),
                    });
                    self.mark_batch_unknown(&run.id, &batch, "tool_batch_timeout");
                }
            }
            _ => unreachable!("checked by caller"),
        }
        Next::Finish(CoreOutcome::Failed {
            error: AgentError::new(
                AgentErrorKind::ReconciliationRequired,
                "tool_batch_outcome_requires_reconciliation:tool_batch_timeout",
            ),
            stage: RunStage::Tools,
        })
    }

    /// 没有成功的 ModelOutput 就没有可合法关联的普通 ToolResults，因此恢复事实用
    /// InputMessage 表达。`ToolResult::metadata` 是应用内部关联数据，这里不给模型看。
    pub(crate) fn interrupted_receipt_message(
        &self,
        receipt: &InterruptedToolBatchReceipt,
    ) -> Result<InputMessage, AgentError> {
        let mut visible_calls = receipt.calls.clone();
        for call in &mut visible_calls {
            if let InterruptedToolCallOutcome::Settled(result) = &mut call.outcome {
                result.metadata.clear();
            }
        }
        let calls = serde_json::to_value(&visible_calls).map_err(|error| {
            AgentError::new(
                AgentErrorKind::InvalidState,
                format!("serialize_interrupted_tool_receipt_failed:{error}"),
            )
        })?;
        Ok(InputMessage::new(
            self.config.interrupted_tool_receipt_role.clone(),
            vec![ContentPart::json(serde_json::json!({
                "kind": DurableFactKind::InterruptedToolBatch.as_wire(),
                "aborted": true,
                "batch_attempt_id": receipt.batch_attempt_id.as_str(),
                "calls": calls,
                "unconfirmed_remainder": "discarded"
            }))],
        ))
    }
}
