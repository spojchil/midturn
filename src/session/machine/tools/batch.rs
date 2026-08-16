//! 整批模式：模型说完后一次交给 `dispatch`，结果整批回来。

use crate::events::RunStage;
use crate::persistence::{EffectIntent, EffectOrigin, EffectOutcome};
use crate::types::{
    order_tool_results, AgentError, AgentErrorKind, ToolCallBatch, ToolCallSlot, ToolResultBatch,
};

use super::super::{Attempt, Core, CoreOutcome, Effect, Next, Phase, Run};
use super::ledger_failure;

impl Core {
    pub(crate) fn dispatch_whole_batch(
        &mut self,
        run: &mut Run,
        attempt: Attempt,
        batch: ToolCallBatch,
        out: &mut Vec<Effect>,
    ) -> Next {
        for (index, call) in batch.calls.iter().enumerate() {
            let slot = ToolCallSlot::new(index as u32);
            let intent = match EffectIntent::new(
                self.session_id.clone(),
                run.id.clone(),
                EffectOrigin::CompletedBatch {
                    batch_id: batch.batch_id.clone(),
                },
                slot,
                call.clone(),
            ) {
                Ok(intent) => intent,
                Err(error) => return Next::Finish(ledger_failure(error)),
            };
            let id = intent.id.clone();
            if let Err(error) = self
                .effects
                .record_intent(intent)
                .and_then(|()| self.effects.begin_delivery(&id))
            {
                return Next::Finish(ledger_failure(error));
            }
        }
        self.touch();
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
        Next::Park
    }

    pub(crate) fn batch_dispatched(
        &mut self,
        run: &mut Run,
        result: Result<ToolResultBatch, AgentError>,
        out: &mut Vec<Effect>,
    ) -> Next {
        let attempt = run.phase.take_attempt().expect("checked by caller");
        let Some(batch) = attempt.batch(&run.id) else {
            return Next::Finish(CoreOutcome::Failed {
                error: AgentError::new(AgentErrorKind::InvalidState, "dispatching_without_output"),
                stage: RunStage::Boundary,
            });
        };
        let expected_ids: Vec<_> = batch.calls.iter().map(|call| call.id.clone()).collect();
        let results = match result {
            Ok(results) => results,
            Err(error) => {
                self.mark_batch_unknown(
                    &run.id,
                    &batch,
                    &format!("tool_runtime_returned_error:{}", error.summary),
                );
                return Next::Finish(CoreOutcome::Failed {
                    error: AgentError::new(
                        AgentErrorKind::ReconciliationRequired,
                        format!(
                            "tool_batch_outcome_requires_reconciliation:{}",
                            error.summary
                        ),
                    ),
                    stage: RunStage::Tools,
                });
            }
        };
        // 在修改任何一条账本记录前，先校验数量、重复项、未知 ID 与顺序。
        let ordered = match order_tool_results(&expected_ids, results.results) {
            Ok(ordered) => ordered,
            Err(error) => {
                self.mark_batch_unknown(
                    &run.id,
                    &batch,
                    &format!(
                        "tool_runtime_returned_invalid_result_batch:{}",
                        error.summary
                    ),
                );
                return Next::Finish(CoreOutcome::Failed {
                    error: AgentError::new(
                        AgentErrorKind::ReconciliationRequired,
                        "invalid_tool_result_batch_requires_reconciliation",
                    ),
                    stage: RunStage::Tools,
                });
            }
        };
        for (index, result) in ordered.iter().enumerate() {
            let id =
                self.batch_effect_id(&run.id, &batch.batch_id, ToolCallSlot::new(index as u32));
            if let Err(ledger) = self
                .effects
                .resolve(&id, EffectOutcome::Settled(result.clone()))
            {
                return Next::Finish(ledger_failure(ledger));
            }
        }
        self.touch();
        self.finish_tool_round(
            run,
            batch.batch_id,
            attempt.tools_started_at,
            ToolResultBatch { results: ordered },
            out,
        )
    }
}
