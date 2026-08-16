//! 一次运行的生命周期：建立、枢纽推进、压缩、收尾、取消。
//!
//! `ReadyToCall` 是枢纽（瞬时态）：`pump` 从这里出发，靠 `turn.next_step()` 决定排空信箱、
//! 叫模型、还是收尾，直到停在一个真的要等世界的阶段。

use std::sync::Arc;
use std::time::Instant;

use tokio::sync::oneshot;

use crate::events::{AgentEvent, ContentEvent, ContentEventKind, EventKind, RunStage};
use crate::ports::ModelRequest;
use crate::run::{RequestBoundaryKind, Turn, TurnStep};
use crate::types::{
    durable_fact_kind, validate_closed_transcript, AgentError, AgentErrorKind, RunId,
    ToolBatchAbortReason, TranscriptItem,
};

use super::super::TurnOutcome;
use super::{
    reply_to, AfterCompaction, Attempt, Core, CoreOutcome, Effect, Machine, Next, Phase, Run,
    RunStats,
};

impl Machine {
    /// 建运行并推进到第一个等待点。调用方已经占好 `run_seq`。
    pub(crate) fn start_run(
        &mut self,
        run_id: RunId,
        out: &mut Vec<Effect>,
    ) -> oneshot::Receiver<TurnOutcome> {
        let (completion_tx, completion_rx) = oneshot::channel();
        let core = &mut self.core;
        let prior_items = core.conversation.len();
        core.emit(out, EventKind::RunStarted, |sequence| {
            AgentEvent::RunStarted {
                sequence,
                run_id: run_id.clone(),
                prior_transcript_items: prior_items,
            }
        });

        // 受保护前缀是同步端口；它 panic 不能连累循环。
        let prefix =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| core.prompt.base_context()))
                .unwrap_or_else(|_| {
                    Err(AgentError::new(
                        AgentErrorKind::InvalidState,
                        "prompt_source_panicked",
                    ))
                });
        let turn = prefix.and_then(|mut prefix| {
            let protected_prefix_len = prefix.len();
            prefix.extend(core.conversation.iter().cloned());
            Turn::try_new(run_id.clone(), prefix).map(|turn| (turn, protected_prefix_len))
        });

        match turn {
            Ok((turn, protected_prefix_len)) => {
                self.run = Some(Run {
                    id: run_id,
                    turn,
                    protected_prefix_len,
                    stats: RunStats::default(),
                    phase: Phase::ReadyToCall,
                    waiters: vec![completion_tx],
                    last_compaction_checked_len: None,
                });
                self.pump(out);
            }
            Err(error) => {
                // 运行占了位但一步没走：直接以失败收尾，让等待方拿到终态。
                core.emit(out, EventKind::RunFailed, |sequence| {
                    AgentEvent::RunFailed {
                        sequence,
                        run_id: run_id.clone(),
                        stage: RunStage::Boundary,
                        error_kind: error.kind,
                    }
                });
                let _ = completion_tx.send(TurnOutcome::Failed {
                    run_id,
                    stage: RunStage::Boundary,
                    usage: None,
                    error,
                });
                core.wake_idle_waiters(out);
            }
        }
        completion_rx
    }

    /// 从枢纽出发推进，直到停在一个要等世界的阶段，或收尾。
    pub(crate) fn pump(&mut self, out: &mut Vec<Effect>) {
        loop {
            let Some(run) = self.run.as_mut() else {
                return;
            };
            debug_assert!(matches!(run.phase, Phase::ReadyToCall));
            match self.core.advance_once(run, out) {
                Advance::Continue => continue,
                Advance::Then(Next::Park) => return,
                Advance::Then(Next::Pump) => continue,
                Advance::Then(Next::Finish(outcome)) => {
                    self.finish_run(outcome, out);
                    return;
                }
            }
        }
    }

    /// 收尾。若收尾前还该压一次，先转 `Compacting`，压完再回来。
    pub(crate) fn finish_run(&mut self, outcome: CoreOutcome, out: &mut Vec<Effect>) {
        let Some(run) = self.run.as_mut() else {
            return;
        };
        // 只有正常完成才值得在收尾前压一次；失败和停止的运行没有「下一次请求」要省。
        if matches!(outcome, CoreOutcome::Completed { .. }) {
            if let Some(started_at) = self.core.request_compaction(run, out) {
                run.phase = Phase::Compacting {
                    then: AfterCompaction::Finish(outcome),
                    started_at,
                };
                return;
            }
        }
        self.finish_now(outcome, None, out);
    }

    /// `compacted` 给出时用它替代 turn 的可提交后缀作为最终 conversation。
    fn finish_now(
        &mut self,
        outcome: CoreOutcome,
        compacted: Option<Vec<TranscriptItem>>,
        out: &mut Vec<Effect>,
    ) {
        let Some(Run {
            id: run_id,
            turn,
            protected_prefix_len,
            stats,
            waiters,
            ..
        }) = self.run.take()
        else {
            return;
        };
        let core = &mut self.core;
        core.conversation = Arc::new(
            compacted
                .unwrap_or_else(|| turn.committable_transcript()[protected_prefix_len..].to_vec()),
        );
        core.touch();

        let usage = turn.usage().cloned();
        let outcome = match outcome {
            CoreOutcome::Completed { output, usage } => {
                core.emit(out, EventKind::RunCompleted, |sequence| {
                    AgentEvent::RunCompleted {
                        sequence,
                        run_id: run_id.clone(),
                        model_requests: stats.model_requests,
                        tool_batches: stats.tool_batches,
                        interrupted_tool_recoveries: stats.interrupted_tool_recoveries,
                        usage: usage.clone(),
                    }
                });
                TurnOutcome::Completed {
                    run_id,
                    output,
                    usage,
                }
            }
            CoreOutcome::Stopped => {
                core.emit(out, EventKind::RunStopped, |sequence| {
                    AgentEvent::RunStopped {
                        sequence,
                        run_id: run_id.clone(),
                    }
                });
                TurnOutcome::Stopped { run_id, usage }
            }
            CoreOutcome::Failed { error, stage } => {
                core.emit(out, EventKind::RunFailed, |sequence| {
                    AgentEvent::RunFailed {
                        sequence,
                        run_id: run_id.clone(),
                        stage,
                        error_kind: error.kind,
                    }
                });
                TurnOutcome::Failed {
                    run_id,
                    stage,
                    usage,
                    error,
                }
            }
        };
        for waiter in waiters {
            reply_to(out, waiter, outcome.clone());
        }
        core.wake_idle_waiters(out);
        // 内容可能在最后一次边界检查之后才到；收尾之后必须再看一次信箱——
        // 但那是下一步的事：先把「已结束」落盘。
        out.push(Effect::Wake);
    }

    // ── 压缩 ─────────────────────────────────────────────────────

    pub(super) fn on_compacted(
        &mut self,
        run_id: RunId,
        result: Result<Vec<TranscriptItem>, AgentError>,
        out: &mut Vec<Effect>,
    ) {
        let Some(run) = self.run.as_mut() else {
            return;
        };
        if run.id != run_id || !matches!(run.phase, Phase::Compacting { .. }) {
            return;
        }
        let Phase::Compacting { then, started_at } =
            std::mem::replace(&mut run.phase, Phase::ReadyToCall)
        else {
            unreachable!("checked above")
        };
        let accepted = match self.core.validate_compaction(run, result, started_at, out) {
            Ok(accepted) => accepted,
            Err(outcome) => {
                self.finish_now(outcome, None, out);
                return;
            }
        };
        match then {
            AfterCompaction::Drain(kind) => {
                // Turn 停在 WaitingBoundary，这正是替换可提交后缀的合法时机。
                if let Some(compacted) = accepted {
                    if let Err(error) = run
                        .turn
                        .replace_committable_suffix(run.protected_prefix_len, compacted)
                    {
                        self.finish_now(
                            CoreOutcome::Failed {
                                error,
                                stage: RunStage::Compaction,
                            },
                            None,
                            out,
                        );
                        return;
                    }
                    self.core.touch();
                }
                run.last_compaction_checked_len = Some(run.turn.committable_transcript().len());
                match self.core.drain_boundary(run, kind, out) {
                    Advance::Continue => self.pump(out),
                    Advance::Then(next) => self.settle(next, out),
                }
            }
            AfterCompaction::Finish(outcome) => self.finish_now(outcome, accepted, out),
        }
    }

    // ── 取消 ─────────────────────────────────────────────────────

    pub(super) fn on_cancel(
        &mut self,
        reply: oneshot::Sender<Option<RunId>>,
        out: &mut Vec<Effect>,
    ) {
        let Some(run) = self.run.as_mut() else {
            reply_to(out, reply, None);
            return;
        };
        let run_id = run.id.clone();
        let next = self.core.cancel(run, out);
        reply_to(out, reply, Some(run_id));
        self.settle(next, out);
    }
}

/// `advance_once` 的结论：多一个「继续下一步」，因为 pump 是个循环。
pub(super) enum Advance {
    Continue,
    Then(Next),
}

impl Core {
    /// 从枢纽走一步：排空边界 / 叫模型 / 进工具阶段 / 完成。
    fn advance_once(&mut self, run: &mut Run, out: &mut Vec<Effect>) -> Advance {
        let step = match run.turn.next_step() {
            Ok(step) => step,
            Err(error) => {
                return Advance::Then(Next::Finish(CoreOutcome::Failed {
                    error,
                    stage: RunStage::Boundary,
                }))
            }
        };
        match step {
            TurnStep::RequestBoundary { kind } => {
                // 只有确实要再请求模型时才值得先压缩：完成边界上信箱空着就直接收尾。
                let will_call_model = kind == RequestBoundaryKind::BeforeModelRequest
                    || self.mailbox.has_deliverable_items(kind);
                if will_call_model {
                    if let Some(started_at) = self.request_compaction(run, out) {
                        // Turn 已经停在 WaitingBoundary；压完直接排空这个边界，不再 next_step。
                        run.phase = Phase::Compacting {
                            then: AfterCompaction::Drain(kind),
                            started_at,
                        };
                        return Advance::Then(Next::Park);
                    }
                }
                self.drain_boundary(run, kind, out)
            }
            TurnStep::CallModel { transcript } => self.call_model(run, transcript, out),
            TurnStep::DispatchTools { .. } => {
                // 工具阶段由 model_done 直接进入（它手里有 attempt）；枢纽不该撞见它。
                Advance::Then(Next::Finish(CoreOutcome::Failed {
                    error: AgentError::new(
                        AgentErrorKind::InvalidState,
                        "tool_batch_requested_outside_model_done",
                    ),
                    stage: RunStage::Boundary,
                }))
            }
            TurnStep::Done { output, usage } => {
                Advance::Then(Next::Finish(CoreOutcome::Completed { output, usage }))
            }
        }
    }

    /// 排空信箱进 turn。前置：turn 停在 WaitingBoundary(kind)。
    pub(super) fn drain_boundary(
        &mut self,
        run: &mut Run,
        kind: RequestBoundaryKind,
        out: &mut Vec<Effect>,
    ) -> Advance {
        // 运行的第一个请求边界就是「会话空闲」那一刻：WhenIdle 的内容此时投递。
        let idle_start =
            kind == RequestBoundaryKind::BeforeModelRequest && run.stats.model_requests == 0;
        let items = if idle_start {
            self.mailbox.drain_idle()
        } else {
            self.mailbox.drain(kind)
        };
        let sealed = kind == RequestBoundaryKind::BeforeCompletion && items.is_empty();
        let item_count = items.len();
        self.touch();
        let run_id = run.id.clone();
        self.emit(out, EventKind::BoundaryDrained, |sequence| {
            AgentEvent::BoundaryDrained {
                sequence,
                run_id: run_id.clone(),
                kind,
                item_count,
            }
        });
        if sealed {
            self.emit(out, EventKind::CompletionSealed, |sequence| {
                AgentEvent::CompletionSealed {
                    sequence,
                    run_id: run_id.clone(),
                }
            });
        }
        match run.turn.resume_boundary(items) {
            Ok(()) => Advance::Continue,
            Err(error) => Advance::Then(Next::Finish(CoreOutcome::Failed {
                error,
                stage: RunStage::Boundary,
            })),
        }
    }

    fn call_model(
        &mut self,
        run: &mut Run,
        transcript: Vec<TranscriptItem>,
        out: &mut Vec<Effect>,
    ) -> Advance {
        if let Err(error) = self.charge_model_request() {
            return Advance::Then(Next::Finish(CoreOutcome::Failed {
                error,
                stage: RunStage::Model,
            }));
        }
        run.stats.model_requests = run.stats.model_requests.saturating_add(1);
        let request_index = run.stats.model_requests;
        let attempt = Attempt::new(&run.id, request_index);
        let Ok(definitions) =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.tools.definitions()))
        else {
            // 工具目录都拿不到，就别装作没有工具去问模型：那会得到一个静默错误的回答。
            return Advance::Then(Next::Finish(CoreOutcome::Failed {
                error: AgentError::new(AgentErrorKind::InvalidState, "tool_definitions_panicked"),
                stage: RunStage::Model,
            }));
        };
        let run_id = run.id.clone();
        self.emit(out, EventKind::ModelRequestStarted, |sequence| {
            AgentEvent::ModelRequestStarted {
                sequence,
                run_id: run_id.clone(),
                request_index,
                transcript_items: transcript.len(),
                function_tools: definitions.len(),
            }
        });
        self.emit_content(out, ContentEventKind::ModelRequestTranscript, |sequence| {
            ContentEvent::ModelRequestTranscript {
                sequence,
                run_id: run_id.clone(),
                request_index,
                transcript: transcript.clone(),
                function_tools: definitions.clone(),
            }
        });
        out.push(Effect::CallModel {
            attempt: attempt.id.clone(),
            request: ModelRequest {
                session_id: self.session_id.clone(),
                run_id: run.id.clone(),
                request_index,
                transcript,
                function_tools: definitions,
            },
        });
        run.phase = Phase::CallingModel(attempt);
        self.touch();
        Advance::Then(Next::Park)
    }

    // ── 压缩 ─────────────────────────────────────────────────────

    /// 当前上下文是否大到该压缩；是则发 `Compact` 并返回开始时刻。
    /// 同一个可提交长度只检查一次。
    fn request_compaction(&mut self, run: &mut Run, out: &mut Vec<Effect>) -> Option<Instant> {
        let committable_len = run.turn.committable_transcript().len();
        if run.last_compaction_checked_len == Some(committable_len) {
            return None;
        }
        run.last_compaction_checked_len = Some(committable_len);
        let conversation = run.turn.committable_transcript()[run.protected_prefix_len..].to_vec();
        if conversation.is_empty() {
            return None;
        }
        let estimated_bytes = serde_json::to_vec(&conversation)
            .map(|encoded| encoded.len())
            .unwrap_or(usize::MAX);
        let budget = &self.config.budget.context;
        // token 有值就用 token，否则回退到字节估算——同一类度量，精度不同。
        let needs = match (
            self.consumption.last_input_tokens,
            budget.compact_above_tokens,
        ) {
            (Some(tokens), Some(limit)) => tokens > limit,
            _ => estimated_bytes > budget.compact_above_bytes,
        };
        if !needs {
            return None;
        }
        let run_id = run.id.clone();
        let items = conversation.len();
        self.emit(out, EventKind::CompactionStarted, |sequence| {
            AgentEvent::CompactionStarted {
                sequence,
                run_id: run_id.clone(),
                transcript_items: items,
                estimated_bytes,
            }
        });
        out.push(Effect::Compact {
            run_id: run.id.clone(),
            conversation,
        });
        Some(Instant::now())
    }

    /// 校验压缩结果。`Ok(Some)` = 接受；`Ok(None)` = 拒绝、沿用原记录；`Err` = 本轮以此收尾。
    #[allow(clippy::result_large_err)]
    fn validate_compaction(
        &mut self,
        run: &mut Run,
        result: Result<Vec<TranscriptItem>, AgentError>,
        started_at: Instant,
        out: &mut Vec<Effect>,
    ) -> Result<Option<Vec<TranscriptItem>>, CoreOutcome> {
        let compacted = match result {
            Ok(compacted) => compacted,
            Err(error) if error.kind == AgentErrorKind::Cancelled => {
                return Err(CoreOutcome::Stopped)
            }
            Err(error) => {
                return Err(CoreOutcome::Failed {
                    error,
                    stage: RunStage::Compaction,
                })
            }
        };
        // 压缩是受信端口，但其输出仍不能破坏闭合不变量，也不能动耐久事实的序列。
        // 拒绝时保留原始对话，避免一次可选优化污染下一次模型请求。
        let original = &run.turn.committable_transcript()[run.protected_prefix_len..];
        let preserves_facts = original
            .iter()
            .filter(|item| durable_fact_kind(item).is_some())
            .eq(compacted
                .iter()
                .filter(|item| durable_fact_kind(item).is_some()));
        let accepted = validate_closed_transcript(&compacted).is_ok() && preserves_facts;
        let items = if accepted {
            compacted.len()
        } else {
            original.len()
        };
        let run_id = run.id.clone();
        self.emit(out, EventKind::CompactionFinished, |sequence| {
            AgentEvent::CompactionFinished {
                sequence,
                run_id,
                transcript_items: items,
                duration_ms: super::model::elapsed_ms(started_at),
            }
        });
        Ok(accepted.then_some(compacted))
    }

    // ── 取消 ─────────────────────────────────────────────────────

    fn cancel(&mut self, run: &mut Run, out: &mut Vec<Effect>) -> Next {
        let cancelled = AgentError::new(AgentErrorKind::Cancelled, "run_cancelled");
        match std::mem::replace(&mut run.phase, Phase::ReadyToCall) {
            Phase::CallingModel(attempt) => {
                out.push(Effect::AbortModel {
                    attempt: attempt.id.clone(),
                });
                self.begin_abort(
                    run,
                    attempt,
                    ToolBatchAbortReason::ModelStreamInterrupted,
                    cancelled,
                    out,
                )
            }
            Phase::WaitingTools(attempt) => self.begin_abort(
                run,
                attempt,
                ToolBatchAbortReason::ModelStreamInterrupted,
                cancelled,
                out,
            ),
            Phase::Aborting {
                attempt, reason, ..
            } => {
                // 已经在中止；只把最终结论改成 Stopped。
                run.phase = Phase::Aborting {
                    attempt,
                    reason,
                    error: cancelled,
                };
                Next::Park
            }
            Phase::Dispatching { attempt } => {
                if let Some(batch) = attempt.batch(&run.id) {
                    out.push(Effect::AbortDispatch {
                        batch_id: batch.batch_id.clone(),
                    });
                    self.mark_batch_unknown(&run.id, &batch, "cancelled_during_dispatch");
                }
                Next::Finish(CoreOutcome::Stopped)
            }
            Phase::Compacting { .. } => {
                out.push(Effect::AbortCompact {
                    run_id: run.id.clone(),
                });
                Next::Finish(CoreOutcome::Stopped)
            }
            Phase::ReadyToCall => unreachable!("transient phase never rests"),
        }
    }

    // ── 预算 ─────────────────────────────────────────────────────

    fn charge_model_request(&mut self) -> Result<(), AgentError> {
        if let Some(limit) = self.config.budget.cumulative.max_model_requests {
            if self.consumption.model_requests >= limit {
                self.auto_start = false;
                return Err(AgentError::new(
                    AgentErrorKind::BudgetExceeded,
                    "session_model_request_budget_exhausted",
                ));
            }
        }
        self.consumption.model_requests = self.consumption.model_requests.saturating_add(1);
        self.touch();
        Ok(())
    }

    /// 按调用数而不是批次数计费：一批里有几个调用就是几份成本。
    pub(super) fn charge_tool_calls(&mut self, calls: u64) -> Result<(), AgentError> {
        if let Some(limit) = self.config.budget.cumulative.max_tool_calls {
            if self.consumption.tool_calls.saturating_add(calls) > limit {
                self.auto_start = false;
                return Err(AgentError::new(
                    AgentErrorKind::BudgetExceeded,
                    "session_tool_call_budget_exhausted",
                ));
            }
        }
        self.consumption.tool_calls = self.consumption.tool_calls.saturating_add(calls);
        self.touch();
        Ok(())
    }
}
