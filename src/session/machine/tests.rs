//! `step` 的纯函数测试。不需要 tokio runtime：oneshot 通道离开 runtime 也能收发。
//!
//! 每条测试对应转移表的一行或几行。断言的是 `Machine` 之后处在哪个阶段、产出了哪些
//! effect、顺序对不对——尤其是 `Persist` 是否排在依赖它的 effect 之前。

use std::sync::Arc;

use serde_json::json;
use tokio::sync::oneshot;

use crate::mailbox::{Delivery, MailboxInput, MailboxRejectedReason};
use crate::persistence::{EffectLedger, SessionId};
use crate::ports::{
    IncrementalToolBatch, ModelResponse, ModelStreamEvent, PortFuture, PromptSource,
    ToolCallReporter, ToolRuntime,
};
use crate::types::{
    durable_fact_kind, AbortClassification, AgentError, AgentErrorKind, ContentPart, InputMessage,
    ModelOutput, ToolBatchAttemptId, ToolBatchStart, ToolCall, ToolCallBatch, ToolCallId,
    ToolCallSlot, ToolDefinition, ToolResult, ToolResultBatch, TranscriptItem,
};

use super::super::{HoldReason, SessionConfig, TurnOutcome};
use super::{
    AttemptMode, Command, Effect, Machine, Observation, Phase, RuntimeCall, RuntimeOp, StartOutcome,
};

// ── 测试替身 ─────────────────────────────────────────────────────

struct StaticPrompt;

impl PromptSource for StaticPrompt {
    fn base_context(&self) -> Result<Vec<TranscriptItem>, AgentError> {
        Ok(vec![InputMessage::text("system", "base").into()])
    }
}

/// 只声明一个工具，别的什么都不做——`step` 不调用运行时的执行方法。
struct DeclaredTools;

impl ToolRuntime for DeclaredTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition::new("read", json!({"type": "object"}))]
    }

    fn dispatch<'a>(
        &'a self,
        _batch: ToolCallBatch,
    ) -> PortFuture<'a, Result<ToolResultBatch, AgentError>> {
        Box::pin(async { unreachable!("step 不执行工具") })
    }

    fn begin_incremental<'a>(
        &'a self,
        _batch: ToolBatchStart,
        _reporter: Arc<dyn ToolCallReporter>,
    ) -> PortFuture<'a, Result<Option<Box<dyn IncrementalToolBatch + 'a>>, AgentError>> {
        Box::pin(async { unreachable!("step 不执行工具") })
    }
}

fn machine() -> Machine {
    Machine::new(
        SessionId::new("test").unwrap(),
        SessionConfig::default(),
        Arc::new(StaticPrompt),
        Arc::new(DeclaredTools),
        Arc::new(Vec::new()),
        Default::default(),
        EffectLedger::default(),
        Default::default(),
        0,
        None,
    )
}

fn user(text: &str) -> TranscriptItem {
    InputMessage::text("user", text).into()
}

fn call(id: &str) -> ToolCall {
    ToolCall::new(id, "read", json!({"path": id}))
}

fn result(id: &str) -> ToolResult {
    ToolResult::success_json(ToolCallId::new(id), json!({"ok": true}))
}

/// 执行回复闭包，把其余 effect 原样返回，便于断言。
fn run_replies(effects: Vec<Effect>) -> Vec<Effect> {
    effects
        .into_iter()
        .filter_map(|effect| match effect {
            Effect::Reply(reply) => {
                reply();
                None
            }
            other => Some(other),
        })
        .collect()
}

fn names(effects: &[Effect]) -> Vec<&'static str> {
    effects
        .iter()
        .map(|effect| match effect {
            Effect::Persist => "persist",
            Effect::Reply(_) => "reply",
            Effect::Wake => "wake",
            Effect::CallModel { .. } => "call_model",
            Effect::AbortModel { .. } => "abort_model",
            Effect::BeginIncremental { .. } => "begin_incremental",
            Effect::Runtime {
                call: RuntimeCall::Submit(_),
                ..
            } => "submit",
            Effect::Runtime {
                call: RuntimeCall::Seal(_),
                ..
            } => "seal",
            Effect::Runtime {
                call: RuntimeCall::Commit,
                ..
            } => "commit",
            Effect::Runtime {
                call: RuntimeCall::Abort(_),
                ..
            } => "abort",
            Effect::Release { .. } => "release",
            Effect::Dispatch { .. } => "dispatch",
            Effect::AbortDispatch { .. } => "abort_dispatch",
            Effect::Compact { .. } => "compact",
            Effect::AbortCompact { .. } => "abort_compact",
            Effect::ArmTimer { .. } => "arm_timer",
            Effect::Emit(Observation::Agent(_)) => "emit",
            Effect::Emit(Observation::Content(_)) => "emit_content",
            Effect::Emit(Observation::Stream(_)) => "emit_stream",
        })
        .collect()
}

/// 去掉观测事件，只留控制流。
fn control_flow(effects: &[Effect]) -> Vec<&'static str> {
    names(effects)
        .into_iter()
        .filter(|name| !name.starts_with("emit"))
        .collect()
}

fn phase_name(machine: &Machine) -> &'static str {
    match machine.run.as_ref().map(|run| &run.phase) {
        None => "idle",
        Some(Phase::ReadyToCall) => "ready_to_call",
        Some(Phase::CallingModel(_)) => "calling_model",
        Some(Phase::WaitingTools(_)) => "waiting_tools",
        Some(Phase::Aborting { .. }) => "aborting",
        Some(Phase::Dispatching { .. }) => "dispatching",
        Some(Phase::Compacting { .. }) => "compacting",
    }
}

fn attempt_id(machine: &Machine) -> ToolBatchAttemptId {
    machine
        .run
        .as_ref()
        .and_then(|run| run.phase.attempt())
        .map(|attempt| attempt.id.clone())
        .expect("有 attempt")
}

fn transcript(machine: &Machine) -> &[TranscriptItem] {
    machine.run.as_ref().unwrap().turn.transcript()
}

/// 投递一条触发内容并要求它把空闲会话叫醒；返回本轮的终态接收端。
fn enqueue_and_start(machine: &mut Machine, text: &str) -> oneshot::Receiver<TurnOutcome> {
    let (tx, mut rx) = oneshot::channel();
    let effects = machine.step(Command::Enqueue {
        input: MailboxInput::next_model_request(vec![user(text)]),
        may_start: true,
        expected_run: None,
        reply: tx,
    });
    run_replies(effects);
    let Ok(Ok(StartOutcome::Started { completion, .. })) = rx.try_recv() else {
        panic!("空闲会话收到触发内容应当开跑")
    };
    completion
}

fn accept_incremental(machine: &mut Machine) -> Vec<Effect> {
    let effects = machine.step(Command::RuntimeAck {
        attempt: attempt_id(machine),
        op: RuntimeOp::Begin(true),
        result: Ok(()),
    });
    run_replies(effects)
}

/// 送一条 ToolCallReady，执行回复，返回其余 effect。
fn ready(machine: &mut Machine, slot: u32, id: &str) -> Vec<Effect> {
    let (tx, mut rx) = oneshot::channel();
    let effects = machine.step(Command::ModelEvent {
        attempt: attempt_id(machine),
        event: ModelStreamEvent::ToolCallReady {
            slot: ToolCallSlot::new(slot),
            call: call(id),
        },
        reply: tx,
    });
    let effects = run_replies(effects);
    assert!(
        matches!(rx.try_recv(), Ok(Ok(()))),
        "ToolCallReady 应被接受"
    );
    effects
}

fn settle(machine: &mut Machine, slot: u32, id: &str) -> (Result<(), AgentError>, Vec<Effect>) {
    let (tx, mut rx) = oneshot::channel();
    let effects = machine.step(Command::ToolSettled {
        attempt: attempt_id(machine),
        slot: ToolCallSlot::new(slot),
        result: result(id),
        reply: tx,
    });
    let effects = run_replies(effects);
    (rx.try_recv().expect("有回复"), effects)
}

fn done(machine: &mut Machine, output: ModelOutput) -> Vec<Effect> {
    let effects = machine.step(Command::ModelDone {
        attempt: attempt_id(machine),
        response: ModelResponse {
            output,
            finish_reason: None,
            usage: None,
        },
    });
    run_replies(effects)
}

fn fail_stream(machine: &mut Machine, summary: &str) -> Vec<Effect> {
    let effects = machine.step(Command::ModelFailed {
        attempt: attempt_id(machine),
        error: AgentError::new(AgentErrorKind::Model, summary),
    });
    run_replies(effects)
}

// ── 轮外：信箱 ───────────────────────────────────────────────────

#[test]
fn enqueue_on_idle_starts_a_run_and_persists_before_replying() {
    let mut m = machine();
    let (tx, mut rx) = oneshot::channel();
    let effects = m.step(Command::Enqueue {
        input: MailboxInput::next_model_request(vec![user("hi")]),
        may_start: true,
        expected_run: None,
        reply: tx,
    });
    // Persist 必须是第一项：调用方拿到 Started 时，信箱已排、运行已建、turn 已含内容。
    assert!(matches!(effects.first(), Some(Effect::Persist)));
    let effects = run_replies(effects);
    assert!(matches!(
        rx.try_recv(),
        Ok(Ok(StartOutcome::Started { .. }))
    ));
    assert_eq!(phase_name(&m), "calling_model");
    assert!(control_flow(&effects).contains(&"call_model"));
    // 信箱已经排空进 turn。
    assert!(!m.core.mailbox.has_trigger_items());
    assert!(transcript(&m)
        .iter()
        .any(|item| matches!(item, TranscriptItem::Input(msg) if msg.role == "user")));
}

#[test]
fn passive_content_does_not_wake_an_idle_session() {
    let mut m = machine();
    let (tx, mut rx) = oneshot::channel();
    run_replies(m.step(Command::Enqueue {
        input: MailboxInput::passive(vec![user("fyi")]),
        may_start: true,
        expected_run: None,
        reply: tx,
    }));
    assert!(matches!(
        rx.try_recv(),
        Ok(Ok(StartOutcome::Held(HoldReason::Passive)))
    ));
    assert_eq!(phase_name(&m), "idle");
}

#[test]
fn enqueue_while_running_is_pending_and_drains_at_the_next_boundary() {
    let mut m = machine();
    let _completion = enqueue_and_start(&mut m, "first");
    let (tx, mut rx) = oneshot::channel();
    run_replies(m.step(Command::Enqueue {
        input: MailboxInput::next_model_request(vec![user("second")]),
        may_start: true,
        expected_run: None,
        reply: tx,
    }));
    assert!(matches!(rx.try_recv(), Ok(Ok(StartOutcome::Pending))));
    assert_eq!(phase_name(&m), "calling_model");
    // 模型说完（无工具）→ 完成边界发现有内容 → 再叫一次模型。
    let effects = done(&mut m, ModelOutput::text("ok"));
    assert_eq!(phase_name(&m), "calling_model");
    assert!(control_flow(&effects).contains(&"call_model"));
    assert!(transcript(&m).iter().any(|item| matches!(
        item,
        TranscriptItem::Input(msg) if msg.content == vec![ContentPart::text("second")]
    )));
}

#[test]
fn reschedule_promotes_passive_and_wakes_the_session() {
    let mut m = machine();
    let (tx, _rx) = oneshot::channel();
    run_replies(m.step(Command::Enqueue {
        input: MailboxInput::passive(vec![user("later")]),
        may_start: true,
        expected_run: None,
        reply: tx,
    }));
    let (tx, mut rx) = oneshot::channel();
    let effects = m.step(Command::Reschedule {
        from: Delivery::Passive,
        to: Delivery::NextModelRequest,
        reply: tx,
    });
    assert!(matches!(effects.first(), Some(Effect::Persist)));
    run_replies(effects);
    let reply = rx.try_recv().unwrap().unwrap();
    assert_eq!(reply.envelopes, 1);
    assert!(matches!(reply.outcome, StartOutcome::Started { .. }));
    assert_eq!(phase_name(&m), "calling_model");
}

// ── 轮内：模型流与工具 ───────────────────────────────────────────

#[test]
fn first_tool_call_asks_the_runtime_and_buffers_until_it_answers() {
    let mut m = machine();
    enqueue_and_start(&mut m, "go");
    let effects = ready(&mut m, 0, "a");
    // Undecided：问运行时，不交付，也不落盘。
    assert_eq!(control_flow(&effects), vec!["begin_incremental"]);
    let effects = ready(&mut m, 1, "b");
    assert!(
        control_flow(&effects).is_empty(),
        "第二个只进 ready，不再问：{:?}",
        names(&effects)
    );
    let attempt = m.run.as_ref().unwrap().phase.attempt().unwrap();
    assert_eq!(attempt.mode, AttemptMode::Undecided);
    assert_eq!(attempt.ready.len(), 2);
}

#[test]
fn runtime_taking_over_delivers_buffered_calls_after_persisting() {
    let mut m = machine();
    enqueue_and_start(&mut m, "go");
    ready(&mut m, 0, "a");
    ready(&mut m, 1, "b");
    let effects = accept_incremental(&mut m);
    // 落盘（InFlight）必须先于 Submit。
    assert_eq!(control_flow(&effects), vec!["persist", "submit", "submit"]);
    let attempt = m.run.as_ref().unwrap().phase.attempt().unwrap();
    assert_eq!(attempt.mode, AttemptMode::Incremental);
    // 之后每个 ToolCallReady 直接交付。
    let effects = ready(&mut m, 2, "c");
    assert_eq!(control_flow(&effects), vec!["persist", "submit"]);
}

#[test]
fn early_settlement_is_persisted_before_the_runtime_hears_ok() {
    let mut m = machine();
    enqueue_and_start(&mut m, "go");
    ready(&mut m, 0, "a");
    accept_incremental(&mut m);
    let (tx, mut rx) = oneshot::channel();
    let effects = m.step(Command::ToolSettled {
        attempt: attempt_id(&m),
        slot: ToolCallSlot::new(0),
        result: result("a"),
        reply: tx,
    });
    let order = names(&effects);
    let persist_at = order.iter().position(|n| *n == "persist").unwrap();
    let reply_at = order.iter().position(|n| *n == "reply").unwrap();
    assert!(
        persist_at < reply_at,
        "Reply(Ok) 必须排在 Persist 之后：{order:?}"
    );
    run_replies(effects);
    assert!(matches!(rx.try_recv(), Ok(Ok(()))));
    // 模型还没说完，阶段不变。
    assert_eq!(phase_name(&m), "calling_model");
}

#[test]
fn settling_the_same_slot_twice_is_rejected_by_the_ledger() {
    let mut m = machine();
    enqueue_and_start(&mut m, "go");
    ready(&mut m, 0, "a");
    accept_incremental(&mut m);
    let (first, _) = settle(&mut m, 0, "a");
    assert!(first.is_ok());
    let (second, effects) = settle(&mut m, 0, "a");
    assert_eq!(second.unwrap_err().kind, AgentErrorKind::InvalidToolBatch);
    assert!(
        !control_flow(&effects).contains(&"persist"),
        "被拒的上报不落盘"
    );
}

/// 运行时确认 commit 之前，工具轮不闭合——哪怕结果早就齐了。运行时可能正在 commit() 里
/// 结算最后一槽然后返回 Err；先闭合，那条 Err 回执就撞不上任何 attempt。
#[test]
fn model_done_with_everything_already_settled_closes_on_the_commit_ack() {
    let mut m = machine();
    enqueue_and_start(&mut m, "go");
    ready(&mut m, 0, "a");
    accept_incremental(&mut m);
    settle(&mut m, 0, "a").0.unwrap();
    let effects = done(&mut m, ModelOutput::calls(vec![call("a")]));
    let order = control_flow(&effects);
    assert!(order.contains(&"commit"), "{order:?}");
    assert!(
        !order.contains(&"call_model"),
        "确认之前不叫下一次模型：{order:?}"
    );
    assert_eq!(phase_name(&m), "waiting_tools");

    let effects = run_replies(m.step(Command::RuntimeAck {
        attempt: attempt_id(&m),
        op: RuntimeOp::Commit,
        result: Ok(()),
    }));
    let order = control_flow(&effects);
    assert!(order.contains(&"release"), "{order:?}");
    assert!(
        order.contains(&"call_model"),
        "确认到了、结果齐了：闭合 → 边界 → 下一次模型请求 {order:?}"
    );
    assert_eq!(phase_name(&m), "calling_model");
    let stats = m.run.as_ref().unwrap().stats;
    assert_eq!(stats.model_requests, 2);
    assert_eq!(stats.tool_batches, 1);
}

/// 运行时在 commit() 里把最后一槽结算完、然后返回 Err：结果齐了也不算数，本轮严格终止。
#[test]
fn commit_error_after_the_last_settlement_still_terminates_the_run() {
    let mut m = machine();
    let mut completion = enqueue_and_start(&mut m, "go");
    ready(&mut m, 0, "a");
    accept_incremental(&mut m);
    done(&mut m, ModelOutput::calls(vec![call("a")]));
    // 运行时在 commit() 里先结算……
    let (result, effects) = settle(&mut m, 0, "a");
    result.unwrap();
    assert!(!control_flow(&effects).contains(&"call_model"));
    assert_eq!(phase_name(&m), "waiting_tools");
    // ……然后 commit 返回 Err。
    run_replies(m.step(Command::RuntimeAck {
        attempt: attempt_id(&m),
        op: RuntimeOp::Commit,
        result: Err(AgentError::new(
            AgentErrorKind::ToolDispatch,
            "commit exploded",
        )),
    }));
    assert_eq!(phase_name(&m), "idle");
    let Ok(TurnOutcome::Failed { error, .. }) = completion.try_recv() else {
        panic!("commit 失败必须终止本轮")
    };
    assert_eq!(error.summary, "tool_batch_commit_failed:commit exploded");
    assert!(
        m.pending_reconciliation().is_empty(),
        "全都结算了，没有要对账的"
    );
    // 但结算过的事实不能只留在账本里：下一轮看的是对话。
    assert!(
        m.core.conversation.iter().any(|item| {
            matches!(item, TranscriptItem::Input(msg)
                if msg.content.iter().any(|part| matches!(part, ContentPart::Json { value }
                    if value["calls"][0]["outcome"]["status"] == "settled")))
        }),
        "已结算的结果必须以回执形式进历史"
    );
}

#[test]
fn model_done_with_pending_slots_waits_and_the_last_settlement_closes_the_round() {
    let mut m = machine();
    enqueue_and_start(&mut m, "go");
    ready(&mut m, 0, "a");
    ready(&mut m, 1, "b");
    accept_incremental(&mut m);
    let effects = done(&mut m, ModelOutput::calls(vec![call("a"), call("b")]));
    assert_eq!(phase_name(&m), "waiting_tools");
    assert!(control_flow(&effects).contains(&"arm_timer"));
    let (_, effects) = settle(&mut m, 1, "b");
    assert_eq!(phase_name(&m), "waiting_tools", "还差 slot 0");
    assert!(!control_flow(&effects).contains(&"call_model"));
    // 运行时确认了 commit；还差 slot 0，继续等。
    run_replies(m.step(Command::RuntimeAck {
        attempt: attempt_id(&m),
        op: RuntimeOp::Commit,
        result: Ok(()),
    }));
    assert_eq!(phase_name(&m), "waiting_tools", "确认了但还差 slot 0");
    let (_, effects) = settle(&mut m, 0, "a");
    assert_eq!(
        phase_name(&m),
        "calling_model",
        "最后一个到了就闭合并叫下一次模型"
    );
    assert!(control_flow(&effects).contains(&"call_model"));
    // 结果按槽序进了 turn。
    let results = transcript(&m)
        .iter()
        .find_map(|item| match item {
            TranscriptItem::ToolResults(batch) => Some(batch),
            _ => None,
        })
        .expect("工具结果已进对话");
    assert_eq!(results.results[0].call_id.as_str(), "a");
    assert_eq!(results.results[1].call_id.as_str(), "b");
}

#[test]
fn model_done_that_disagrees_with_ready_calls_aborts_the_attempt() {
    let mut m = machine();
    enqueue_and_start(&mut m, "go");
    ready(&mut m, 0, "a");
    accept_incremental(&mut m);
    let effects = done(&mut m, ModelOutput::calls(vec![call("zzz")]));
    assert_eq!(phase_name(&m), "aborting");
    assert!(control_flow(&effects).contains(&"abort"));
}

// ── 中止 ─────────────────────────────────────────────────────────

#[test]
fn stream_failure_with_nothing_delivered_fails_the_run_without_asking_the_runtime() {
    let mut m = machine();
    let mut completion = enqueue_and_start(&mut m, "go");
    let effects = fail_stream(&mut m, "boom");
    assert!(
        !control_flow(&effects).contains(&"abort"),
        "什么都没交出去，不必问运行时"
    );
    assert_eq!(phase_name(&m), "idle");
    let Ok(TurnOutcome::Failed { error, .. }) = completion.try_recv() else {
        panic!("应以失败收尾")
    };
    assert_eq!(error.summary, "boom");
}

#[test]
fn stream_failure_after_delivery_asks_the_runtime_and_keeps_settled_facts() {
    let mut m = machine();
    let _completion = enqueue_and_start(&mut m, "go");
    ready(&mut m, 0, "a");
    ready(&mut m, 1, "b");
    accept_incremental(&mut m);
    // slot 0 提前跑完并上报；slot 1 还在跑时断流。
    settle(&mut m, 0, "a").0.unwrap();
    let attempt = attempt_id(&m);
    let effects = fail_stream(&mut m, "disconnected");
    assert_eq!(phase_name(&m), "aborting");
    assert!(control_flow(&effects).contains(&"abort"));
    // 运行时说 slot 1 确定未开始。
    let effects = run_replies(m.step(Command::BatchAborted {
        attempt,
        result: Ok(AbortClassification::new().cancelled_before_start(ToolCallSlot::new(1))),
    }));
    let order = control_flow(&effects);
    assert!(order.contains(&"release"), "{order:?}");
    // 有事实（slot 0 已结算）→ 回执进 turn → 重新叫模型。
    assert_eq!(phase_name(&m), "calling_model");
    assert!(order.contains(&"call_model"), "{order:?}");
    assert!(
        transcript(&m)
            .iter()
            .any(|item| durable_fact_kind(item).is_some()),
        "回执是耐久事实"
    );
}

#[test]
fn abort_report_cannot_erase_a_settlement_that_already_hit_the_ledger() {
    let mut m = machine();
    enqueue_and_start(&mut m, "go");
    ready(&mut m, 0, "a");
    accept_incremental(&mut m);
    settle(&mut m, 0, "a").0.unwrap();
    let attempt = attempt_id(&m);
    fail_stream(&mut m, "disconnected");
    // 报告声称 slot 0 确定未开始——但它已经结算并落盘了。报告里这一项被忽略。
    run_replies(m.step(Command::BatchAborted {
        attempt,
        result: Ok(AbortClassification::new().cancelled_before_start(ToolCallSlot::new(0))),
    }));
    let receipt = transcript(&m)
        .iter()
        .find(|item| durable_fact_kind(item).is_some())
        .expect("已结算的事实必须进回执");
    let TranscriptItem::Input(message) = receipt else {
        unreachable!()
    };
    let text = serde_json::to_string(&message.content).unwrap();
    assert!(text.contains("settled"), "{text}");
}

#[test]
fn stream_failure_with_only_unstarted_deliveries_fails_the_run() {
    let mut m = machine();
    let mut completion = enqueue_and_start(&mut m, "go");
    ready(&mut m, 0, "a");
    accept_incremental(&mut m);
    let attempt = attempt_id(&m);
    fail_stream(&mut m, "disconnected");
    run_replies(m.step(Command::BatchAborted {
        attempt,
        result: Ok(AbortClassification::new().cancelled_before_start(ToolCallSlot::new(0))),
    }));
    // 没有任何事实可交代：这就是一次失败的模型请求。内核不重试。
    assert_eq!(phase_name(&m), "idle");
    assert!(matches!(
        completion.try_recv(),
        Ok(TurnOutcome::Failed { .. })
    ));
}

#[test]
fn cancel_while_streaming_with_nothing_delivered_stops_cleanly() {
    let mut m = machine();
    let mut completion = enqueue_and_start(&mut m, "go");
    let (tx, mut rx) = oneshot::channel();
    let effects = run_replies(m.step(Command::Cancel { reply: tx }));
    assert!(rx.try_recv().unwrap().is_some());
    assert!(control_flow(&effects).contains(&"abort_model"));
    assert_eq!(phase_name(&m), "idle");
    assert!(matches!(
        completion.try_recv(),
        Ok(TurnOutcome::Stopped { .. })
    ));
}

/// commit 之后所有者任务已经退出：没有 abort 可问。取消不能挂在 Aborting 里等一个永远
/// 不会来的报告——仍在途的一律未知，运行 Stopped，下一轮开跑前先对账。
#[test]
fn cancel_after_commit_marks_in_flight_unknown_and_stops() {
    let mut m = machine();
    let mut completion = enqueue_and_start(&mut m, "go");
    ready(&mut m, 0, "a");
    accept_incremental(&mut m);
    let effects = done(&mut m, ModelOutput::calls(vec![call("a")]));
    assert!(control_flow(&effects).contains(&"commit"));
    // 运行时确认 commit：所有者退出，指令通道该收回。
    let effects = run_replies(m.step(Command::RuntimeAck {
        attempt: attempt_id(&m),
        op: RuntimeOp::Commit,
        result: Ok(()),
    }));
    assert!(control_flow(&effects).contains(&"release"));
    assert_eq!(phase_name(&m), "waiting_tools");

    let (tx, mut rx) = oneshot::channel();
    let effects = run_replies(m.step(Command::Cancel { reply: tx }));
    assert!(rx.try_recv().unwrap().is_some());
    assert!(
        !control_flow(&effects).contains(&"abort"),
        "commit 之后没有 abort 可发：{:?}",
        control_flow(&effects)
    );
    assert_eq!(phase_name(&m), "idle");
    assert!(matches!(
        completion.try_recv(),
        Ok(TurnOutcome::Stopped { .. })
    ));
    assert_eq!(
        m.pending_reconciliation().len(),
        1,
        "在途的那一槽必须变成未知，等对账"
    );
}

/// 取消插在「Commit 已发出」与「Commit 回执」之间：所有者可能已经消费了批次对象并退出，
/// 我们发出去的 Abort 没人接。Commit 的回执随后到达 Aborting——它就是全部答案。
#[test]
fn commit_ack_arriving_while_aborting_resolves_without_a_report() {
    let mut m = machine();
    let mut completion = enqueue_and_start(&mut m, "go");
    ready(&mut m, 0, "a");
    accept_incremental(&mut m);
    let effects = done(&mut m, ModelOutput::calls(vec![call("a")]));
    assert!(control_flow(&effects).contains(&"commit"));
    let attempt = attempt_id(&m);

    let (tx, _rx) = oneshot::channel();
    let effects = run_replies(m.step(Command::Cancel { reply: tx }));
    // 还不知道 commit 有没有被消费：照常发 Abort（所有者若因更早失败还活着，会答）。
    assert!(control_flow(&effects).contains(&"abort"));
    assert_eq!(phase_name(&m), "aborting");

    // 所有者其实已经把 commit 交给运行时并退出了：Commit 回执到达。
    let effects = run_replies(m.step(Command::RuntimeAck {
        attempt,
        op: RuntimeOp::Commit,
        result: Ok(()),
    }));
    assert!(control_flow(&effects).contains(&"release"));
    assert_eq!(phase_name(&m), "idle");
    assert!(matches!(
        completion.try_recv(),
        Ok(TurnOutcome::Stopped { .. })
    ));
    assert_eq!(m.pending_reconciliation().len(), 1);
}

/// 运行时不接管（`Begin(false)`）时所有者任务同样已经退出：指令通道也要收回。
#[test]
fn declined_incremental_takeover_releases_the_owner() {
    let mut m = machine();
    enqueue_and_start(&mut m, "go");
    ready(&mut m, 0, "a");
    let effects = run_replies(m.step(Command::RuntimeAck {
        attempt: attempt_id(&m),
        op: RuntimeOp::Begin(false),
        result: Ok(()),
    }));
    assert!(control_flow(&effects).contains(&"release"));
}

#[test]
fn late_settlement_after_the_run_ended_is_rejected() {
    let mut m = machine();
    enqueue_and_start(&mut m, "go");
    ready(&mut m, 0, "a");
    accept_incremental(&mut m);
    let attempt = attempt_id(&m);
    let (tx, _rx) = oneshot::channel();
    run_replies(m.step(Command::Cancel { reply: tx }));
    // 取消进入 Aborting；运行时报了；运行结束。
    run_replies(m.step(Command::BatchAborted {
        attempt: attempt.clone(),
        result: Ok(AbortClassification::new().cancelled_before_start(ToolCallSlot::new(0))),
    }));
    assert_eq!(phase_name(&m), "idle");
    let (tx, mut rx) = oneshot::channel();
    run_replies(m.step(Command::ToolSettled {
        attempt,
        slot: ToolCallSlot::new(0),
        result: result("a"),
        reply: tx,
    }));
    assert!(
        rx.try_recv().unwrap().is_err(),
        "运行结束后的上报得到稳定错误"
    );
}

// ── 隔离 ─────────────────────────────────────────────────────────

#[test]
fn persist_failure_fences_the_session() {
    let mut m = machine();
    m.step(Command::PersistFailed {
        error: AgentError::new(AgentErrorKind::PersistenceCommitUnknown, "lost ack"),
    });
    let (tx, mut rx) = oneshot::channel();
    run_replies(m.step(Command::Enqueue {
        input: MailboxInput::next_model_request(vec![user("hi")]),
        may_start: true,
        expected_run: None,
        reply: tx,
    }));
    let rejected = rx.try_recv().unwrap().unwrap_err();
    assert_eq!(
        rejected.reason,
        MailboxRejectedReason::PersistenceCommitUnknown
    );
    assert_eq!(phase_name(&m), "idle");
}

/// 取消 × Commit 回执 Err：取消决定终态（Stopped），commit 的失败摘要进未知态的说明。
#[test]
fn commit_error_while_cancelling_still_stops_but_records_the_failure() {
    let mut m = machine();
    let mut completion = enqueue_and_start(&mut m, "go");
    ready(&mut m, 0, "a");
    accept_incremental(&mut m);
    done(&mut m, ModelOutput::calls(vec![call("a")]));
    let attempt = attempt_id(&m);
    let (tx, _rx) = oneshot::channel();
    run_replies(m.step(Command::Cancel { reply: tx }));
    assert_eq!(phase_name(&m), "aborting");
    run_replies(m.step(Command::RuntimeAck {
        attempt,
        op: RuntimeOp::Commit,
        result: Err(AgentError::new(
            AgentErrorKind::ToolDispatch,
            "commit exploded",
        )),
    }));
    assert_eq!(phase_name(&m), "idle");
    assert!(matches!(
        completion.try_recv(),
        Ok(TurnOutcome::Stopped { .. })
    ));
    let pending = m.pending_reconciliation();
    assert_eq!(pending.len(), 1);
    let crate::persistence::EffectRecoveryAction::Reconcile { summary, .. } = &pending[0] else {
        panic!("在途的槽应当要求对账：{pending:?}")
    };
    assert!(summary.contains("commit_failed"), "{summary}");
}

/// 取消时已经有一槽结算了：Stopped 照旧，但那条事实必须先进历史。
#[test]
fn cancel_after_a_settlement_keeps_the_fact_in_history() {
    let mut m = machine();
    let mut completion = enqueue_and_start(&mut m, "go");
    ready(&mut m, 0, "a");
    accept_incremental(&mut m);
    settle(&mut m, 0, "a").0.unwrap();
    let attempt = attempt_id(&m);
    let (tx, _rx) = oneshot::channel();
    run_replies(m.step(Command::Cancel { reply: tx }));
    run_replies(m.step(Command::BatchAborted {
        attempt,
        result: Ok(AbortClassification::new()),
    }));
    assert!(matches!(
        completion.try_recv(),
        Ok(TurnOutcome::Stopped { .. })
    ));
    assert!(m.core.conversation.iter().any(|item| {
        matches!(item, TranscriptItem::Input(msg)
            if msg.content.iter().any(|part| matches!(part, ContentPart::Json { value }
                if value["calls"][0]["call_id"] == "a")))
    }));
}
