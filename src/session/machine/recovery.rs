//! 重开、显式续跑、对账。
//!
//! 重开不自动续跑半截的运行：`Boot` 只在没有半截运行时看信箱。半截运行留在盘上那个
//! 位置，应用对账完毕后显式 `Resume`。崩溃后账本里 InFlight 的槽一律要求对账——内核不猜。

use std::sync::Arc;

use tokio::sync::oneshot;

use crate::events::{AgentEvent, EventKind, RunStage};
use crate::persistence::{
    EffectId, EffectOutcome, EffectRecoveryAction, EffectState, RunResumePoint, RunSnapshot,
};
use crate::run::{Turn, TurnStep};
use crate::types::{
    AgentError, AgentErrorKind, ContentPart, DurableFactKind, InputMessage,
    InterruptedToolBatchReceipt, InterruptedToolCallOutcome, InterruptedToolCallReceipt,
    ToolResult, ToolResultBatch, TranscriptItem,
};

use super::super::TurnOutcome;
use super::{
    reply_to, Attempt, AttemptMode, Core, CoreOutcome, Effect, Machine, Next, Phase, Run, RunStats,
    StartOutcome,
};

impl Machine {
    pub(super) fn on_boot(&mut self, out: &mut Vec<Effect>) {
        if self.core.recovered.is_none() {
            let _ = self.try_start(out);
        }
    }

    pub(super) fn on_resume(
        &mut self,
        reply: oneshot::Sender<Result<Option<StartOutcome>, AgentError>>,
        out: &mut Vec<Effect>,
    ) {
        let Some(snapshot) = self.core.recovered.clone() else {
            // 没有半截运行可续。此刻是否有新运行在跑无关紧要：那不是「续」的对象。
            reply_to(out, reply, Ok(None));
            return;
        };
        if self.run.is_some() {
            reply_to(
                out,
                reply,
                Err(AgentError::new(
                    AgentErrorKind::InvalidState,
                    "session_is_busy",
                )),
            );
            return;
        }
        let actions = self.core.effects.recovery_actions_for_run(&snapshot.run_id);
        if actions
            .iter()
            .any(|action| matches!(action, EffectRecoveryAction::Reconcile { .. }))
        {
            reply_to(
                out,
                reply,
                Err(AgentError::new(
                    AgentErrorKind::ReconciliationRequired,
                    "recovered_run_has_ambiguous_effects",
                )),
            );
            return;
        }
        match self.core.resume_from_snapshot(snapshot, out) {
            Ok((run, next)) => {
                let run_id = run.id.clone();
                let (completion_tx, completion_rx) = oneshot::channel();
                let mut run = run;
                run.waiters.push(completion_tx);
                self.run = Some(run);
                reply_to(
                    out,
                    reply,
                    Ok(Some(StartOutcome::Started {
                        run_id,
                        completion: completion_rx,
                    })),
                );
                self.settle(next, out);
            }
            Err(error) => reply_to(out, reply, Err(error)),
        }
    }

    pub(super) fn on_reconcile(
        &mut self,
        effect_id: EffectId,
        outcome: EffectOutcome,
        reply: oneshot::Sender<Result<(), AgentError>>,
        out: &mut Vec<Effect>,
    ) {
        let owned_by_active = self.run.as_ref().is_some_and(|run| {
            self.core
                .effects
                .get(&effect_id)
                .is_some_and(|record| record.intent.run_id == run.id)
        });
        if owned_by_active {
            reply_to(
                out,
                reply,
                Err(AgentError::new(
                    AgentErrorKind::InvalidState,
                    "effect_is_owned_by_active_driver",
                )),
            );
            return;
        }
        let result = self.core.reconcile(effect_id, outcome);
        let cleared = result.is_ok();
        reply_to(out, reply, result);
        if cleared {
            // 对账可能刚清掉最后一个未知态：被 `ReconciliationRequired` 扣住的信箱内容
            // 现在有人取了。与 `SetAutoStart(true)` 一样，这里只是重新检查触发条件。
            let _ = self.try_start(out);
        }
    }

    /// 当前必须对账、不能自动重投的副作用。活跃运行自己的 InFlight 不算——那只是它已经
    /// 授权投递；只有运行释放所有权后它才成为需要操作员核对的状态。
    pub(crate) fn pending_reconciliation(&self) -> Vec<EffectRecoveryAction> {
        let active_run_id = self.run.as_ref().map(|run| &run.id);
        self.core
            .effects
            .iter()
            .filter(|(_, record)| active_run_id != Some(&record.intent.run_id))
            .filter_map(|(id, record)| match &record.state {
                EffectState::InFlight { delivery_attempt } => {
                    Some(EffectRecoveryAction::Reconcile {
                        id: id.clone(),
                        intent: record.intent.clone(),
                        summary: format!(
                            "effect_was_in_flight:delivery_attempt={delivery_attempt}"
                        ),
                    })
                }
                EffectState::Outcome {
                    outcome: EffectOutcome::OutcomeUnknown { summary },
                } => Some(EffectRecoveryAction::Reconcile {
                    id: id.clone(),
                    intent: record.intent.clone(),
                    summary: summary.clone(),
                }),
                _ => None,
            })
            .collect()
    }
}

impl Core {
    /// 用外部查询、幂等存储或人工确认得到的权威结果解决一个未知 effect。
    fn reconcile(&mut self, effect_id: EffectId, outcome: EffectOutcome) -> Result<(), AgentError> {
        let record = self
            .effects
            .get(&effect_id)
            .cloned()
            .ok_or_else(|| AgentError::new(AgentErrorKind::Persistence, "effect_not_found"))?;
        let visible_outcome = sanitized_effect_outcome(&outcome);
        self.effects
            .reconcile(&effect_id, outcome)
            .map_err(super::tools::ledger_error)?;
        Arc::make_mut(&mut self.conversation).push(TranscriptItem::Input(InputMessage::new(
            self.config.interrupted_tool_receipt_role.clone(),
            vec![ContentPart::json(serde_json::json!({
                "kind": DurableFactKind::EffectReconciliation.as_wire(),
                "effect_id": effect_id.as_str(),
                "call_id": record.intent.call.id.as_str(),
                "tool": record.intent.call.name.as_str(),
                "arguments": record.intent.call.arguments,
                "outcome": visible_outcome,
            }))],
        )));
        self.touch();
        Ok(())
    }

    /// 把盘上的半截运行变成一个内存 `Run`，并给出第一步。
    fn resume_from_snapshot(
        &mut self,
        mut snapshot: RunSnapshot,
        out: &mut Vec<Effect>,
    ) -> Result<(Run, Next), AgentError> {
        let run_id = snapshot.run_id.clone();

        // 模型尝试中断：已结算的进回执，Prepared 的取消（从未离开内核），然后从头再请求。
        if let RunResumePoint::ModelAttempt {
            batch_attempt_id,
            effect_ids,
            ..
        } = &snapshot.resume_from
        {
            let mut receipt_calls = Vec::new();
            for id in effect_ids {
                let record = self.effects.get(id).cloned().ok_or_else(|| {
                    AgentError::new(AgentErrorKind::Persistence, "resume_effect_not_in_ledger")
                })?;
                match record.state {
                    EffectState::Prepared => {
                        self.effects
                            .cancel_prepared(id)
                            .map_err(super::tools::ledger_error)?;
                    }
                    EffectState::Outcome {
                        outcome: EffectOutcome::Settled(result),
                    } => receipt_calls.push(InterruptedToolCallReceipt {
                        slot: record.intent.slot,
                        call_id: record.intent.call.id.clone(),
                        name: record.intent.call.name.clone(),
                        arguments: record.intent.call.arguments.clone(),
                        outcome: InterruptedToolCallOutcome::Settled(result),
                    }),
                    EffectState::Outcome {
                        outcome: EffectOutcome::CancelledBeforeStart,
                    } => {}
                    // `on_resume` 已经用 recovery_actions_for_run 挡在前面；这里再守一道，
                    // 未来校验演进也不许把它变成循环级 panic。
                    EffectState::InFlight { .. }
                    | EffectState::Outcome {
                        outcome: EffectOutcome::OutcomeUnknown { .. },
                    } => {
                        return Err(AgentError::new(
                            AgentErrorKind::ReconciliationRequired,
                            "resume_requires_reconciliation",
                        ))
                    }
                }
            }
            if !receipt_calls.is_empty() {
                let receipt = InterruptedToolBatchReceipt {
                    batch_attempt_id: batch_attempt_id.clone(),
                    calls: receipt_calls,
                };
                let message = self.interrupted_receipt_message(&receipt)?;
                Arc::make_mut(&mut self.conversation).push(TranscriptItem::Input(message));
            }
            snapshot.resume_from = RunResumePoint::BeforeModelRequest;
        }

        // 整批里有确定未开始的：不能当作一个完整工具轮，改成回执再从头请求。
        if let RunResumePoint::ToolBatch {
            batch_id,
            effect_ids,
            ..
        } = &snapshot.resume_from
        {
            let has_cancelled = effect_ids.iter().any(|id| {
                matches!(
                    self.effects.get(id).map(|record| &record.state),
                    Some(EffectState::Outcome {
                        outcome: EffectOutcome::CancelledBeforeStart
                    })
                )
            });
            if has_cancelled {
                let calls = effect_ids
                    .iter()
                    .map(|id| {
                        let record = self.effects.get(id).ok_or_else(|| {
                            AgentError::new(
                                AgentErrorKind::Persistence,
                                "resume_effect_not_in_ledger",
                            )
                        })?;
                        let visible_state = match &record.state {
                            EffectState::Outcome { outcome } => EffectState::Outcome {
                                outcome: sanitized_effect_outcome(outcome),
                            },
                            state => state.clone(),
                        };
                        Ok(serde_json::json!({
                            "effect_id": id.as_str(),
                            "call_id": record.intent.call.id.as_str(),
                            "tool": record.intent.call.name.as_str(),
                            "arguments": &record.intent.call.arguments,
                            "outcome": visible_state,
                        }))
                    })
                    .collect::<Result<Vec<_>, AgentError>>()?;
                Arc::make_mut(&mut self.conversation).push(TranscriptItem::Input(
                    InputMessage::new(
                        self.config.interrupted_tool_receipt_role.clone(),
                        vec![ContentPart::json(serde_json::json!({
                            "kind": DurableFactKind::RecoveredToolBatch.as_wire(),
                            "batch_id": batch_id.as_str(),
                            "calls": calls,
                        }))],
                    ),
                ));
                snapshot.resume_from = RunResumePoint::BeforeModelRequest;
            }
        }

        self.recovered = None;
        self.touch();
        let prior_items = self.conversation.len();
        self.emit(out, EventKind::RunStarted, |sequence| {
            AgentEvent::RunStarted {
                sequence,
                run_id: run_id.clone(),
                prior_transcript_items: prior_items,
            }
        });

        let mut prefix =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.prompt.base_context()))
                .unwrap_or_else(|_| {
                Err(AgentError::new(
                    AgentErrorKind::InvalidState,
                    "prompt_source_panicked",
                ))
            })?;
        let protected_prefix_len = prefix.len();
        prefix.extend(self.conversation.iter().cloned());
        let stats = RunStats {
            model_requests: snapshot.model_request_sequence,
            tool_batches: snapshot.tool_batch_sequence,
            interrupted_tool_recoveries: u64::from(snapshot.interrupted_tool_recoveries),
        };
        let usage = snapshot.usage.clone();
        let tool_sequence = snapshot.tool_batch_sequence;

        let (turn, next_seed) = match snapshot.resume_from {
            RunResumePoint::BeforeModelRequest | RunResumePoint::ModelAttempt { .. } => (
                Turn::restore_before_model(run_id.clone(), prefix, tool_sequence, usage)?,
                Seed::Pump,
            ),
            RunResumePoint::BeforeCompletion { output } => (
                Turn::restore_before_completion(
                    run_id.clone(),
                    prefix,
                    output,
                    tool_sequence,
                    usage,
                )?,
                Seed::Pump,
            ),
            RunResumePoint::ToolBatch {
                batch_id,
                output,
                effect_ids,
            } => (
                Turn::restore_tool_batch(
                    run_id.clone(),
                    prefix,
                    output.clone(),
                    batch_id,
                    tool_sequence,
                    usage,
                )?,
                Seed::ToolBatch { output, effect_ids },
            ),
        };
        let mut run = Run {
            id: run_id,
            turn,
            protected_prefix_len,
            stats: RunStats {
                // 待恢复的那一批不重复计数。
                tool_batches: stats
                    .tool_batches
                    .saturating_sub(u64::from(matches!(next_seed, Seed::ToolBatch { .. }))),
                ..stats
            },
            phase: Phase::ReadyToCall,
            waiters: Vec::new(),
            last_compaction_checked_len: None,
        };
        let next = match next_seed {
            Seed::Pump => Next::Pump,
            Seed::ToolBatch { output, effect_ids } => {
                self.resume_tool_batch(&mut run, output, effect_ids, out)
            }
        };
        Ok((run, next))
    }

    /// 权威模型输出已形成、工具结果尚未全部写回：全部已结算就直接复用，一个都没交付
    /// 就重新 dispatch，混合则不敢猜。
    fn resume_tool_batch(
        &mut self,
        run: &mut Run,
        output: crate::types::ModelOutput,
        effect_ids: Vec<EffectId>,
        out: &mut Vec<Effect>,
    ) -> Next {
        let batch = match run.turn.next_step() {
            Ok(TurnStep::DispatchTools { batch }) => batch,
            Ok(_) | Err(_) => {
                return Next::Finish(CoreOutcome::Failed {
                    error: AgentError::new(
                        AgentErrorKind::InvalidState,
                        "resumed_turn_did_not_request_tools",
                    ),
                    stage: RunStage::Boundary,
                })
            }
        };
        let mut results: Vec<ToolResult> = Vec::with_capacity(effect_ids.len());
        let mut prepared = 0usize;
        for id in &effect_ids {
            let Some(record) = self.effects.get(id) else {
                return Next::Finish(CoreOutcome::Failed {
                    error: AgentError::new(
                        AgentErrorKind::Persistence,
                        "effect_missing_during_resume",
                    ),
                    stage: RunStage::Boundary,
                });
            };
            match &record.state {
                EffectState::Prepared => prepared += 1,
                EffectState::Outcome {
                    outcome: EffectOutcome::Settled(result),
                } => results.push(result.clone()),
                EffectState::InFlight { .. }
                | EffectState::Outcome {
                    outcome: EffectOutcome::OutcomeUnknown { .. },
                } => {
                    return Next::Finish(CoreOutcome::Failed {
                        error: AgentError::new(
                            AgentErrorKind::ReconciliationRequired,
                            "tool_batch_has_ambiguous_effects",
                        ),
                        stage: RunStage::Boundary,
                    })
                }
                EffectState::Outcome {
                    outcome: EffectOutcome::CancelledBeforeStart,
                } => {
                    return Next::Finish(CoreOutcome::Failed {
                        error: AgentError::new(
                            AgentErrorKind::InvalidState,
                            "completed_tool_batch_contains_cancelled_effect",
                        ),
                        stage: RunStage::Boundary,
                    })
                }
            }
        }
        // 重建 attempt：批次 ID 就是尝试 ID（Turn 一直这么给）。
        let mut attempt = Attempt::new(&run.id, run.stats.model_requests);
        attempt.id = crate::types::ToolBatchAttemptId::new(batch.batch_id.as_str());
        attempt.output = Some(output);
        attempt.tools_started_at = Some(std::time::Instant::now());
        if prepared == 0 && results.len() == effect_ids.len() {
            // 全部已结算：直接复用，不再调用工具。
            attempt.mode = AttemptMode::Batch;
            run.stats.tool_batches = run.stats.tool_batches.saturating_add(1);
            return self.finish_tool_round(
                run,
                batch.batch_id,
                attempt.tools_started_at,
                ToolResultBatch { results },
                out,
            );
        }
        if prepared == effect_ids.len() {
            // 一个都没交付：授权后整批 dispatch。
            for id in &effect_ids {
                if let Err(error) = self.effects.begin_delivery(id) {
                    return Next::Finish(CoreOutcome::Failed {
                        error: super::tools::ledger_error(error),
                        stage: RunStage::Boundary,
                    });
                }
            }
            self.touch();
            attempt.mode = AttemptMode::Batch;
            run.stats.tool_batches = run.stats.tool_batches.saturating_add(1);
            if let Some(after) = self.config.tool_timeout {
                out.push(Effect::ArmTimer {
                    attempt: attempt.id.clone(),
                    after,
                });
            }
            out.push(Effect::Dispatch {
                batch_id: batch.batch_id.clone(),
                batch,
            });
            run.phase = Phase::Dispatching { attempt };
            return Next::Park;
        }
        Next::Finish(CoreOutcome::Failed {
            error: AgentError::new(
                AgentErrorKind::InvalidState,
                "tool_batch_partially_delivered_without_outcomes",
            ),
            stage: RunStage::Boundary,
        })
    }
}

enum Seed {
    Pump,
    ToolBatch {
        output: crate::types::ModelOutput,
        effect_ids: Vec<EffectId>,
    },
}

/// 对账结论进对话时不带工具正文：那是宿主保留的，模型只需要知道结论。
pub(crate) fn sanitized_effect_outcome(outcome: &EffectOutcome) -> EffectOutcome {
    match outcome {
        EffectOutcome::Settled(result) => {
            let mut result = result.clone();
            result.content = vec![ContentPart::text("[effect outcome retained by host]")];
            EffectOutcome::Settled(result)
        }
        outcome => outcome.clone(),
    }
}

#[allow(dead_code)]
fn _outcome_type_check(_: TurnOutcome) {}
