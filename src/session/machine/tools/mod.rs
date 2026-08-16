//! 轮内触发源：工具运行时。
//!
//! 「未返回的工具列表」= 账本里当前 attempt 下处于 `InFlight` 的记录。这里从不维护
//! 第二份 `submitted` / `settled`——每次都从账本现算。`ToolSettled` 在任何阶段规则相同，
//! 只有账本一处判定：被拒就回 `Err`。

use tokio::sync::oneshot;

use crate::events::{
    AgentEvent, ContentEvent, ContentEventKind, EventKind, RunStage, ToolCallSummary,
};
use crate::persistence::{EffectId, EffectOrigin, EffectRecord, EffectState};
use crate::types::{
    AbortClassification, AgentError, AgentErrorKind, RunId, ToolBatchAttemptId, ToolBatchId,
    ToolBatchStart, ToolCallBatch, ToolCallSlot, ToolResult, ToolResultBatch, ToolResultStatus,
};

use super::model::elapsed_ms;
use super::{
    reply_to, Attempt, AttemptMode, Core, CoreOutcome, Effect, Machine, Next, Phase, Run,
    RuntimeCall, RuntimeOp,
};

mod abort;
mod batch;
mod incremental;

// ═══════════════════════════════════════════════════════════════
// Machine 层：找到运行、委托给 Core、执行结论
// ═══════════════════════════════════════════════════════════════

impl Machine {
    pub(super) fn on_tool_settled(
        &mut self,
        attempt_id: ToolBatchAttemptId,
        slot: ToolCallSlot,
        result: ToolResult,
        reply: oneshot::Sender<Result<(), AgentError>>,
        out: &mut Vec<Effect>,
    ) {
        // 增量结算只对 CallingModel / WaitingTools / Aborting 有意义；整批 dispatch 不走这里。
        let live = self.run.as_ref().is_some_and(|run| {
            run.phase
                .attempt()
                .is_some_and(|attempt| attempt.id == attempt_id)
                && !matches!(run.phase, Phase::Dispatching { .. })
        });
        let Some(run) = self.run.as_mut().filter(|_| live) else {
            reply_to(out, reply, Err(unknown_attempt()));
            return;
        };
        match self.core.settle_slot(run, &attempt_id, slot, result, out) {
            Err(error) => reply_to(out, reply, Err(error)),
            Ok(next) => {
                // Reply(Ok) 排在 Persist 之后（Persist 是效果列表首位）：运行时收到 Ok 时已落盘。
                reply_to(out, reply, Ok(()));
                self.settle(next, out);
            }
        }
    }

    pub(super) fn on_runtime_ack(
        &mut self,
        attempt_id: ToolBatchAttemptId,
        op: RuntimeOp,
        result: Result<(), AgentError>,
        out: &mut Vec<Effect>,
    ) {
        let Some(run) = self.run.as_mut() else {
            return;
        };
        let next = match &run.phase {
            Phase::CallingModel(a) | Phase::WaitingTools(a) if a.id == attempt_id => {
                self.core.runtime_ack(run, op, result, out)
            }
            // 已经在中止：Submit/Seal 的回执不再重要（所有者活着，会去处理 Abort）；
            // 但 Commit 的回执意味着所有者刚消费了批次对象并退出——Abort 没人接了。
            Phase::Aborting { attempt, .. }
                if attempt.id == attempt_id && matches!(op, RuntimeOp::Commit) =>
            {
                self.core.commit_acked_while_aborting(run, result, out)
            }
            _ => return,
        };
        self.settle(next, out);
    }

    pub(super) fn on_batch_aborted(
        &mut self,
        attempt_id: ToolBatchAttemptId,
        result: Result<AbortClassification, AgentError>,
        out: &mut Vec<Effect>,
    ) {
        let Some(run) = self.run.as_mut() else {
            return;
        };
        let next = match (&run.phase, &result) {
            (Phase::Aborting { attempt, .. }, _) if attempt.id == attempt_id => {
                self.core.batch_aborted(run, result, out)
            }
            // 没在中止却收到一份失败的「报告」：只可能是所有者任务自己没了（胶水 panic）。
            // 运行时不可用，仍在途的一律未知，本轮到此为止。
            (Phase::CallingModel(attempt) | Phase::WaitingTools(attempt), Err(error))
                if attempt.id == attempt_id =>
            {
                let error = error.clone();
                let attempt = run.phase.take_attempt().expect("checked");
                self.core
                    .conclude_without_runtime(run, attempt, "batch_owner_gone", &error, out)
            }
            _ => return,
        };
        self.settle(next, out);
    }

    pub(super) fn on_batch_dispatched(
        &mut self,
        batch_id: ToolBatchId,
        result: Result<ToolResultBatch, AgentError>,
        out: &mut Vec<Effect>,
    ) {
        let live = self.run.as_ref().is_some_and(|run| {
            matches!(&run.phase, Phase::Dispatching { attempt }
                if ToolBatchId::from(&attempt.id) == batch_id)
        });
        let Some(run) = self.run.as_mut().filter(|_| live) else {
            return;
        };
        let next = self.core.batch_dispatched(run, result, out);
        self.settle(next, out);
    }

    pub(super) fn on_batch_timed_out(
        &mut self,
        attempt_id: ToolBatchAttemptId,
        out: &mut Vec<Effect>,
    ) {
        // 计时器只对等待阶段有意义；attempt 早已关闭或还在流就忽略。
        let live = self.run.as_ref().is_some_and(|run| {
            run.phase
                .attempt()
                .is_some_and(|attempt| attempt.id == attempt_id)
                && !matches!(run.phase, Phase::CallingModel(_))
        });
        let Some(run) = self.run.as_mut().filter(|_| live) else {
            return;
        };
        let next = self.core.batch_timed_out(run, out);
        self.settle(next, out);
    }
}

// ═══════════════════════════════════════════════════════════════
// Core 层：运行内逻辑，只回答 Next
// ═══════════════════════════════════════════════════════════════

impl Core {
    // ── 账本查询 ─────────────────────────────────────────────────

    pub(super) fn attempt_effect_id(
        &self,
        run_id: &RunId,
        attempt_id: &ToolBatchAttemptId,
        slot: ToolCallSlot,
    ) -> EffectId {
        EffectId::for_tool_call(
            &self.session_id,
            run_id,
            &EffectOrigin::StreamingAttempt {
                batch_attempt_id: attempt_id.clone(),
            },
            slot,
        )
    }

    pub(super) fn batch_effect_id(
        &self,
        run_id: &RunId,
        batch_id: &ToolBatchId,
        slot: ToolCallSlot,
    ) -> EffectId {
        EffectId::for_tool_call(
            &self.session_id,
            run_id,
            &EffectOrigin::CompletedBatch {
                batch_id: batch_id.clone(),
            },
            slot,
        )
    }

    /// 本 attempt 下的全部账本记录，按槽序。
    pub(super) fn attempt_records(
        &self,
        run_id: &RunId,
        attempt_id: &ToolBatchAttemptId,
    ) -> Vec<(ToolCallSlot, EffectRecord)> {
        let mut records: Vec<_> = self
            .effects
            .iter()
            .filter(|(_, record)| {
                &record.intent.run_id == run_id
                    && matches!(&record.intent.origin,
                        EffectOrigin::StreamingAttempt { batch_attempt_id } if batch_attempt_id == attempt_id)
            })
            .map(|(_, record)| (record.intent.slot, record.clone()))
            .collect();
        records.sort_by_key(|(slot, _)| *slot);
        records
    }

    /// 未返回的工具列表：一个差集，每次现算。
    pub(super) fn in_flight_slots(
        &self,
        run_id: &RunId,
        attempt_id: &ToolBatchAttemptId,
    ) -> Vec<ToolCallSlot> {
        self.attempt_records(run_id, attempt_id)
            .into_iter()
            .filter(|(_, record)| matches!(record.state, EffectState::InFlight { .. }))
            .map(|(slot, _)| slot)
            .collect()
    }

    // ── 进入工具阶段 ─────────────────────────────────────────────

    /// `ModelDone` 之后：决定是增量收口 / 等结算 / 整批 dispatch / 等运行时回答。
    /// 前置：`run.turn` 已停在 WaitingTools。
    pub(super) fn enter_tool_phase(
        &mut self,
        run: &mut Run,
        mut attempt: Attempt,
        batch: ToolCallBatch,
        out: &mut Vec<Effect>,
    ) -> Next {
        run.stats.tool_batches = run.stats.tool_batches.saturating_add(1);
        if let Err(error) = self.charge_tool_calls(batch.calls.len() as u64) {
            return Next::Finish(CoreOutcome::Failed {
                error,
                stage: RunStage::Tools,
            });
        }
        let run_id = run.id.clone();
        let batch_id = batch.batch_id.clone();
        self.emit(out, EventKind::ToolBatchStarted, |sequence| {
            AgentEvent::ToolBatchStarted {
                sequence,
                run_id: run_id.clone(),
                batch_id: batch_id.clone(),
                calls: batch
                    .calls
                    .iter()
                    .map(|call| ToolCallSummary {
                        id: call.id.clone(),
                        name: call.name.clone(),
                    })
                    .collect(),
            }
        });
        attempt.tools_started_at = Some(std::time::Instant::now());
        match attempt.mode {
            AttemptMode::Incremental => {
                out.push(Effect::Runtime {
                    attempt: attempt.id.clone(),
                    call: RuntimeCall::Commit,
                });
                self.wait_or_close(run, attempt, out)
            }
            AttemptMode::Batch => self.dispatch_whole_batch(run, attempt, batch, out),
            AttemptMode::Undecided => {
                if attempt.ready.is_empty() {
                    // 流里没逐项报过调用（非流式模型）：现在才第一次见到它们，现在才问运行时。
                    for (index, call) in batch.calls.iter().enumerate() {
                        attempt
                            .ready
                            .insert(ToolCallSlot::new(index as u32), call.clone());
                    }
                    attempt.sealed = Some(batch.calls.len() as u32);
                    out.push(Effect::BeginIncremental {
                        attempt: attempt.id.clone(),
                        start: ToolBatchStart {
                            run_id: run.id.clone(),
                            batch_attempt_id: attempt.id.clone(),
                        },
                    });
                }
                // 运行时还没回答要不要接管；等 Begin 回执再决定整批还是逐项。
                run.phase = Phase::WaitingTools(attempt);
                Next::Park
            }
        }
    }

    /// 增量模式：InFlight 已空则立即收口，否则等。
    /// 一个 attempt 只武装一次计时器：计时起点是第一次进入等待阶段。
    pub(super) fn arm_batch_timer(&self, attempt: &mut Attempt, out: &mut Vec<Effect>) {
        if attempt.timer_armed {
            return;
        }
        if let Some(after) = self.config.tool_timeout {
            attempt.timer_armed = true;
            out.push(Effect::ArmTimer {
                attempt: attempt.id.clone(),
                after,
            });
        }
    }

    pub(super) fn wait_or_close(
        &mut self,
        run: &mut Run,
        mut attempt: Attempt,
        out: &mut Vec<Effect>,
    ) -> Next {
        // 两个条件缺一不可：InFlight 已空 **且** 运行时已确认 commit。第二条防的是「运行时在
        // commit() 里先把最后一槽 settled 完、然后返回 Err」——那时结果齐了、工具轮却不该闭合，
        // 因为 Commit 的失败回执还在路上；先闭合了它就撞不上任何 attempt，严格终止成了空话。
        if attempt.committed && self.in_flight_slots(&run.id, &attempt.id).is_empty() {
            return self.close_incremental_batch(run, &attempt, out);
        }
        self.arm_batch_timer(&mut attempt, out);
        run.phase = Phase::WaitingTools(attempt);
        Next::Park
    }

    /// 工具轮的结果进 turn，发事件，回到枢纽。
    pub(super) fn finish_tool_round(
        &mut self,
        run: &mut Run,
        batch_id: ToolBatchId,
        started_at: Option<std::time::Instant>,
        results: ToolResultBatch,
        out: &mut Vec<Effect>,
    ) -> Next {
        let error_count = results
            .results
            .iter()
            .filter(|result| result.status == ToolResultStatus::Error)
            .count();
        let run_id = run.id.clone();
        self.emit(out, EventKind::ToolBatchFinished, |sequence| {
            AgentEvent::ToolBatchFinished {
                sequence,
                run_id: run_id.clone(),
                batch_id: batch_id.clone(),
                result_count: results.results.len(),
                error_count,
                duration_ms: started_at.map(elapsed_ms).unwrap_or(0),
            }
        });
        self.emit_content(out, ContentEventKind::ToolBatchResults, |sequence| {
            ContentEvent::ToolBatchResults {
                sequence,
                run_id: run_id.clone(),
                batch_id: batch_id.clone(),
                results: results.clone(),
            }
        });
        if let Err(error) = run.turn.tool_results(results) {
            return Next::Finish(CoreOutcome::Failed {
                error,
                stage: RunStage::Tools,
            });
        }
        self.touch();
        run.phase = Phase::ReadyToCall;
        Next::Pump
    }
}

pub(crate) fn ledger_error(error: crate::persistence::EffectLedgerError) -> AgentError {
    AgentError::new(
        AgentErrorKind::Persistence,
        format!("effect_ledger_error:{error}"),
    )
}

fn ledger_failure(error: crate::persistence::EffectLedgerError) -> CoreOutcome {
    CoreOutcome::Failed {
        error: ledger_error(error),
        stage: RunStage::Boundary,
    }
}

fn unknown_attempt() -> AgentError {
    AgentError::new(
        AgentErrorKind::InvalidToolBatch,
        "settlement_for_unknown_attempt",
    )
}
