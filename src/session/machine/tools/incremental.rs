//! 增量模式：逐槽交付、提前结算、运行时回执、收口。

use crate::events::{AgentEvent, EventKind, RunStage};
use crate::persistence::{EffectIntent, EffectOrigin, EffectOutcome, EffectState};
use crate::types::{
    AgentError, AgentErrorKind, IncrementalToolCall, RunId, ToolBatchAbortReason,
    ToolBatchAttemptId, ToolBatchId, ToolCall, ToolCallSlot, ToolResult, ToolResultBatch,
};

use super::super::{
    Attempt, AttemptMode, Core, CoreOutcome, Effect, Next, Phase, Run, RuntimeCall, RuntimeOp,
};
use super::ledger_error;

impl Core {
    /// 把一个已解析的槽交给运行时：账本 Prepared → InFlight，再发 `Runtime::Submit`。
    /// 落盘（`Persist` 在效果列表首位）必然先于 `Submit` 被执行。
    pub(crate) fn deliver_slot(
        &mut self,
        run_id: &RunId,
        attempt_id: &ToolBatchAttemptId,
        slot: ToolCallSlot,
        call: ToolCall,
        out: &mut Vec<Effect>,
    ) -> Result<(), AgentError> {
        let intent = EffectIntent::new(
            self.session_id.clone(),
            run_id.clone(),
            EffectOrigin::StreamingAttempt {
                batch_attempt_id: attempt_id.clone(),
            },
            slot,
            call.clone(),
        )
        .map_err(ledger_error)?;
        let id = intent.id.clone();
        self.effects.record_intent(intent).map_err(ledger_error)?;
        self.effects.begin_delivery(&id).map_err(ledger_error)?;
        self.touch();
        out.push(Effect::Runtime {
            attempt: attempt_id.clone(),
            call: RuntimeCall::Submit(IncrementalToolCall {
                run_id: run_id.clone(),
                batch_attempt_id: attempt_id.clone(),
                slot,
                call,
            }),
        });
        Ok(())
    }

    /// 增量批次全部结算：按槽序取结果，闭合工具轮。
    pub(crate) fn close_incremental_batch(
        &mut self,
        run: &mut Run,
        attempt: &Attempt,
        out: &mut Vec<Effect>,
    ) -> Next {
        let results: Vec<ToolResult> = self
            .attempt_records(&run.id, &attempt.id)
            .into_iter()
            .filter_map(|(_, record)| match record.state {
                EffectState::Outcome {
                    outcome: EffectOutcome::Settled(result),
                } => Some(result),
                _ => None,
            })
            .collect();
        self.finish_tool_round(
            run,
            ToolBatchId::from(&attempt.id),
            attempt.tools_started_at,
            ToolResultBatch { results },
            out,
        )
    }

    /// 只有账本一处判定：未交付 / 已终态 / call_id 不符全在这里被拒。
    pub(crate) fn settle_slot(
        &mut self,
        run: &mut Run,
        attempt_id: &ToolBatchAttemptId,
        slot: ToolCallSlot,
        result: ToolResult,
        out: &mut Vec<Effect>,
    ) -> Result<Next, AgentError> {
        let call_id = result.call_id.clone();
        let id = self.attempt_effect_id(&run.id, attempt_id, slot);
        self.effects
            .resolve(&id, EffectOutcome::Settled(result))
            .map_err(|error| {
                AgentError::new(
                    AgentErrorKind::InvalidToolBatch,
                    format!("settlement_rejected:{error}"),
                )
            })?;
        self.touch();
        let run_id = run.id.clone();
        let attempt_for_event = attempt_id.clone();
        self.emit(out, EventKind::ToolCallSettled, |sequence| {
            AgentEvent::ToolCallSettled {
                sequence,
                run_id,
                batch_attempt_id: attempt_for_event,
                slot,
                call_id,
            }
        });
        // 等最后一个的那个阶段：这可能就是最后一个。但要等运行时确认了 commit 才算——
        // 它可能正在 commit() 里结算最后一槽然后返回 Err（见 wait_or_close）。
        let closes = matches!(&run.phase, Phase::WaitingTools(attempt)
            if attempt.mode == AttemptMode::Incremental
                && attempt.committed
                && self.in_flight_slots(&run.id, &attempt.id).is_empty());
        if closes {
            let attempt = run.phase.take_attempt().expect("checked");
            return Ok(self.close_incremental_batch(run, &attempt, out));
        }
        Ok(Next::Park)
    }

    pub(crate) fn runtime_ack(
        &mut self,
        run: &mut Run,
        op: RuntimeOp,
        result: Result<(), AgentError>,
        out: &mut Vec<Effect>,
    ) -> Next {
        let was_waiting = matches!(run.phase, Phase::WaitingTools(_));
        // 合法组合只有：Undecided 收 Begin；Incremental 收 Submit/Seal/Commit。别的都是
        // 迟到或错投的回执（Batch 决定后所有者已退出；重复 Begin 会重复交付）：忽略。
        let legal = match run.phase.attempt().map(|attempt| attempt.mode) {
            Some(AttemptMode::Undecided) => matches!(op, RuntimeOp::Begin(_)),
            Some(AttemptMode::Incremental) => !matches!(op, RuntimeOp::Begin(_)),
            Some(AttemptMode::Batch) | None => false,
        };
        if !legal {
            debug_assert!(false, "runtime ack does not fit the attempt mode: {op:?}");
            return Next::Park;
        }
        let mut attempt = run.phase.take_attempt().expect("checked by caller");
        if let Err(error) = result {
            if matches!(op, RuntimeOp::Commit) {
                // commit 消费了运行时的批次对象，没有人可以再被问「哪些开始了」。
                return self.conclude_without_runtime(run, attempt, "commit_failed", &error, out);
            }
            // Begin / Submit / Seal 失败：Undecided 时什么都没交出去，begin_abort 直接收尾；
            // Incremental 时所有者还活着，问它要分类。
            return self.begin_abort(
                run,
                attempt,
                ToolBatchAbortReason::ToolRuntimeRejected,
                error,
                out,
            );
        }
        match op {
            RuntimeOp::Begin(true) => {
                attempt.mode = AttemptMode::Incremental;
                // 把 Undecided 期间攒下的按槽序全部补交。
                let pending: Vec<(ToolCallSlot, ToolCall)> = attempt
                    .ready
                    .iter()
                    .map(|(slot, call)| (*slot, call.clone()))
                    .collect();
                let run_id = run.id.clone();
                let attempt_id = attempt.id.clone();
                for (slot, call) in pending {
                    if let Err(error) = self.deliver_slot(&run_id, &attempt_id, slot, call, out) {
                        return self.begin_abort(
                            run,
                            attempt,
                            ToolBatchAbortReason::ToolRuntimeRejected,
                            error,
                            out,
                        );
                    }
                }
                if let Some(sealed) = attempt.sealed.filter(|n| *n > 0) {
                    out.push(Effect::Runtime {
                        attempt: attempt_id.clone(),
                        call: RuntimeCall::Seal(sealed),
                    });
                }
                if was_waiting {
                    // ModelDone 已经来过：现在 Commit，并看是不是已经全结算了。
                    out.push(Effect::Runtime {
                        attempt: attempt_id,
                        call: RuntimeCall::Commit,
                    });
                    self.wait_or_close(run, attempt, out)
                } else {
                    run.phase = Phase::CallingModel(attempt);
                    Next::Park
                }
            }
            RuntimeOp::Begin(false) => {
                attempt.mode = AttemptMode::Batch;
                // 运行时不接管：所有者任务已经退出，别让指令通道留在表里。
                out.push(Effect::Release {
                    attempt: attempt.id.clone(),
                });
                if was_waiting {
                    let Some(batch) = attempt.batch(&run.id) else {
                        return Next::Finish(CoreOutcome::Failed {
                            error: AgentError::new(
                                AgentErrorKind::InvalidState,
                                "waiting_tools_without_output",
                            ),
                            stage: RunStage::Boundary,
                        });
                    };
                    self.dispatch_whole_batch(run, attempt, batch, out)
                } else {
                    run.phase = Phase::CallingModel(attempt);
                    Next::Park
                }
            }
            RuntimeOp::Commit => {
                // commit 消费了批次对象，所有者任务随之退出：收回指令通道。结果只走 settled。
                out.push(Effect::Release {
                    attempt: attempt.id.clone(),
                });
                attempt.committed = true;
                if was_waiting {
                    // 结果可能早就齐了（运行时在 commit() 里就结算完），只是在等这句确认。
                    self.wait_or_close(run, attempt, out)
                } else {
                    run.phase = Phase::CallingModel(attempt);
                    Next::Park
                }
            }
            RuntimeOp::Submit | RuntimeOp::Seal => {
                run.phase = if was_waiting {
                    Phase::WaitingTools(attempt)
                } else {
                    Phase::CallingModel(attempt)
                };
                Next::Park
            }
        }
    }
}
