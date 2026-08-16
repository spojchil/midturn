//! 轮内触发源：模型流。
//!
//! 文本与推理增量对状态透明，只 `Emit`。改状态的只有 `ToolCallReady` / `ToolCallsSealed`
//! 和流的收口（`ModelDone` / `ModelFailed`）。

use tokio::sync::oneshot;

use crate::events::{
    AgentEvent, ContentEvent, ContentEventKind, EventKind, ModelStreamObservation,
};
use crate::ports::{ModelResponse, ModelStreamEvent};
use crate::run::TurnStep;
use crate::types::{
    AgentError, AgentErrorKind, ToolBatchAbortReason, ToolBatchAttemptId, ToolBatchStart, ToolCall,
    ToolCallSlot,
};

use super::{
    reply_to, Attempt, AttemptMode, Core, CoreOutcome, Effect, Machine, Next, Phase, Run,
    RuntimeCall,
};

impl Machine {
    pub(super) fn on_model_event(
        &mut self,
        attempt_id: ToolBatchAttemptId,
        event: ModelStreamEvent,
        reply: oneshot::Sender<Result<(), AgentError>>,
        out: &mut Vec<Effect>,
    ) {
        let live = self.run.as_ref().is_some_and(|run| {
            matches!(&run.phase, Phase::CallingModel(attempt)
                if attempt.id == attempt_id && attempt.failure.is_none())
        });
        let Some(run) = self.run.as_mut().filter(|_| live) else {
            reply_to(out, reply, Err(stale()));
            return;
        };
        let result = self.core.model_event(run, event, out);
        reply_to(out, reply, result);
    }

    pub(super) fn on_model_done(
        &mut self,
        attempt_id: ToolBatchAttemptId,
        response: ModelResponse,
        out: &mut Vec<Effect>,
    ) {
        let live = self.run.as_ref().is_some_and(
            |run| matches!(&run.phase, Phase::CallingModel(attempt) if attempt.id == attempt_id),
        );
        let Some(run) = self.run.as_mut().filter(|_| live) else {
            return;
        };
        let next = self.core.model_done(run, response, out);
        self.settle(next, out);
    }

    pub(super) fn on_model_failed(
        &mut self,
        attempt_id: ToolBatchAttemptId,
        error: AgentError,
        out: &mut Vec<Effect>,
    ) {
        let live = self.run.as_ref().is_some_and(
            |run| matches!(&run.phase, Phase::CallingModel(attempt) if attempt.id == attempt_id),
        );
        let Some(run) = self.run.as_mut().filter(|_| live) else {
            return;
        };
        let attempt = run.phase.take_attempt().expect("checked live");
        // 流里更早的失败优先：那才是根因。
        let error = attempt.failure.clone().unwrap_or(error);
        let next = self.core.begin_abort(
            run,
            attempt,
            ToolBatchAbortReason::ModelStreamInterrupted,
            error,
            out,
        );
        self.settle(next, out);
    }
}

impl Core {
    fn model_event(
        &mut self,
        run: &mut Run,
        event: ModelStreamEvent,
        out: &mut Vec<Effect>,
    ) -> Result<(), AgentError> {
        let request_index = run.phase.attempt().map(|a| a.request_index).unwrap_or(0);
        self.emit_stream(out, &run.id, request_index, || {
            ModelStreamObservation::Delta(event.clone())
        });
        let result = match event {
            // 正文与推理只是观测内容，不参与工具接管的控制流。
            ModelStreamEvent::TextStart { .. }
            | ModelStreamEvent::TextDelta { .. }
            | ModelStreamEvent::TextEnd { .. }
            | ModelStreamEvent::ReasoningStart { .. }
            | ModelStreamEvent::ReasoningDelta { .. }
            | ModelStreamEvent::ReasoningEnd { .. } => Ok(()),
            ModelStreamEvent::ToolCallReady { slot, call } => {
                self.accept_ready(run, slot, call, out)
            }
            ModelStreamEvent::ToolCallsSealed { call_count } => {
                self.accept_seal(run, call_count, out)
            }
        };
        if let Err(error) = &result {
            // 粘滞记录首次失败：适配器应停下并从 complete_stream 返回 Err；
            // 即使它吞错后发 ModelDone，也会被当作失败处理。
            if let Some(attempt) = run.phase.attempt_mut() {
                attempt.failure.get_or_insert(error.clone());
            }
        }
        result
    }

    fn accept_ready(
        &mut self,
        run: &mut Run,
        slot: ToolCallSlot,
        call: ToolCall,
        out: &mut Vec<Effect>,
    ) -> Result<(), AgentError> {
        let Phase::CallingModel(attempt) = &mut run.phase else {
            unreachable!("checked by caller")
        };
        if attempt.sealed.is_some() {
            return Err(model_error("tool_call_ready_after_calls_sealed"));
        }
        if call.id.as_str().is_empty() {
            return Err(AgentError::new(
                AgentErrorKind::InvalidToolBatch,
                "empty_tool_call_id",
            ));
        }
        if let Some(previous) = attempt.ready.get(&slot) {
            return if previous == &call {
                Ok(())
            } else {
                Err(model_error("conflicting_tool_call_slot"))
            };
        }
        if attempt
            .ready
            .values()
            .any(|previous| previous.id == call.id)
        {
            return Err(AgentError::new(
                AgentErrorKind::InvalidToolBatch,
                "duplicate_tool_call_id",
            ));
        }
        attempt.ready.insert(slot, call.clone());
        match attempt.mode {
            AttemptMode::Undecided => {
                if attempt.ready.len() == 1 {
                    // 第一个调用到了，现在才去问运行时要不要增量接管。
                    out.push(Effect::BeginIncremental {
                        attempt: attempt.id.clone(),
                        start: ToolBatchStart {
                            run_id: run.id.clone(),
                            batch_attempt_id: attempt.id.clone(),
                        },
                    });
                }
                Ok(())
            }
            AttemptMode::Incremental => {
                let attempt_id = attempt.id.clone();
                let run_id = run.id.clone();
                self.deliver_slot(&run_id, &attempt_id, slot, call, out)
            }
            AttemptMode::Batch => Ok(()),
        }
    }

    fn accept_seal(
        &mut self,
        run: &mut Run,
        call_count: u32,
        out: &mut Vec<Effect>,
    ) -> Result<(), AgentError> {
        let Phase::CallingModel(attempt) = &mut run.phase else {
            unreachable!("checked by caller")
        };
        if attempt.sealed.is_some() {
            return Err(model_error("duplicate_tool_calls_sealed"));
        }
        let contiguous = attempt.ready.len() == call_count as usize
            && attempt
                .ready
                .keys()
                .enumerate()
                .all(|(index, slot)| slot.get() as usize == index);
        if !contiguous {
            return Err(model_error("tool_calls_sealed_before_all_calls_ready"));
        }
        attempt.sealed = Some(call_count);
        let attempt_id = attempt.id.clone();
        if attempt.mode == AttemptMode::Incremental && call_count > 0 {
            out.push(Effect::Runtime {
                attempt: attempt_id.clone(),
                call: RuntimeCall::Seal(call_count),
            });
        }
        let run_id = run.id.clone();
        self.emit(out, EventKind::ToolCallsSealed, |sequence| {
            AgentEvent::ToolCallsSealed {
                sequence,
                run_id,
                batch_attempt_id: attempt_id,
                call_count,
            }
        });
        Ok(())
    }

    fn model_done(
        &mut self,
        run: &mut Run,
        response: ModelResponse,
        out: &mut Vec<Effect>,
    ) -> Next {
        let mut attempt = run.phase.take_attempt().expect("checked by caller");
        // 流里出过错就不算成功结束，哪怕适配器吞了错误照发 ModelDone。
        if let Some(error) = attempt.failure.clone() {
            return self.begin_abort(
                run,
                attempt,
                ToolBatchAbortReason::ModelStreamInterrupted,
                error,
                out,
            );
        }
        if let Err(error) = validate_final_response(&attempt, &response) {
            return self.begin_abort(
                run,
                attempt,
                ToolBatchAbortReason::ModelOutputMismatch,
                error,
                out,
            );
        }

        let request_index = attempt.request_index;
        let started_at = attempt.started_at;
        let run_id = run.id.clone();
        self.emit_stream(out, &run_id, request_index, || {
            ModelStreamObservation::AttemptCommitted
        });
        self.emit(out, EventKind::ModelRequestFinished, |sequence| {
            AgentEvent::ModelRequestFinished {
                sequence,
                run_id: run_id.clone(),
                request_index,
                local_tool_calls: response.output.tool_calls.len(),
                duration_ms: elapsed_ms(started_at),
                usage: response.usage.clone(),
            }
        });
        // 服务商刚刚告诉了我们这次请求实际有多大，比任何本地估算都准。
        if let Some(tokens) = response.usage.as_ref().and_then(|usage| usage.input_tokens) {
            self.consumption.last_input_tokens = Some(tokens);
        }
        self.emit_content(out, ContentEventKind::ModelResponseOutput, |sequence| {
            ContentEvent::ModelResponseOutput {
                sequence,
                run_id: run_id.clone(),
                request_index,
                output: response.output.clone(),
            }
        });

        attempt.output = Some(response.output.clone());
        let has_tools = !response.output.tool_calls.is_empty();
        if let Err(error) = run
            .turn
            .model_response_for_attempt(response, attempt.id.clone())
        {
            // 无 I/O 状态机与已校验的规范响应不一致，是严格的内核错误。
            return self.begin_abort(
                run,
                attempt,
                ToolBatchAbortReason::ModelOutputMismatch,
                error,
                out,
            );
        }
        self.touch();
        if !has_tools {
            // 完成边界：从枢纽出发。
            return Next::Pump;
        }
        // 有工具：把 Turn 推到 WaitingTools 并进入工具阶段。
        match run.turn.next_step() {
            Ok(TurnStep::DispatchTools { batch }) => {
                self.enter_tool_phase(run, attempt, batch, out)
            }
            Ok(_) => Next::Finish(CoreOutcome::Failed {
                error: AgentError::new(AgentErrorKind::InvalidState, "turn_did_not_request_tools"),
                stage: RunStage::Boundary,
            }),
            Err(error) => Next::Finish(CoreOutcome::Failed {
                error,
                stage: RunStage::Boundary,
            }),
        }
    }
}

use crate::events::RunStage;

fn validate_final_response(attempt: &Attempt, response: &ModelResponse) -> Result<(), AgentError> {
    let calls = &response.output.tool_calls;
    if let Some(sealed) = attempt.sealed {
        if sealed as usize != calls.len() {
            return Err(model_error("final_tool_calls_disagree_with_sealed_count"));
        }
    }
    if attempt.ready.is_empty() && attempt.sealed.is_none() {
        // 流里什么都没说（非流式模型，或不逐项上报的适配器）：最终响应就是唯一事实。
        return Ok(());
    }
    if attempt.ready.len() != calls.len() {
        return Err(model_error("final_tool_calls_disagree_with_ready_calls"));
    }
    for (index, (slot, ready)) in attempt.ready.iter().enumerate() {
        if slot.get() as usize != index {
            return Err(model_error("ready_tool_call_slots_not_contiguous"));
        }
        if &calls[index] != ready {
            return Err(model_error("final_tool_call_disagrees_with_ready_call"));
        }
    }
    Ok(())
}

fn model_error(summary: &'static str) -> AgentError {
    AgentError::new(AgentErrorKind::Model, summary)
}

fn stale() -> AgentError {
    AgentError::new(
        AgentErrorKind::InvalidState,
        "model_stream_event_after_attempt_closed",
    )
}

pub(super) fn elapsed_ms(started_at: std::time::Instant) -> u64 {
    u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX)
}
