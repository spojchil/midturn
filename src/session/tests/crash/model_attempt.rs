//! 崩溃点：模型尝试进行中（开跑后、交付后、提前结算后、中止中）。

use super::*;

/// 表第 1/2 行：投递并开跑之后、模型还没说任何话。
///
/// 盘上是 `ModelAttempt{effect_ids: []}`，已排空的输入在 turn 里。重开：`Held(ResumeRequired)`，
/// `try_resume` 从头再请求，输入不丢也不重复。
#[tokio::test]
async fn crash_after_start_resumes_by_requesting_the_model_again() {
    let mut before = before("after-start", None).await;
    let _handle = start(&before.session, vec![input("operator", "before crash")])
        .await
        .unwrap();
    before.requests.recv().await.unwrap();
    let checkpoint = before.store.latest();
    assert!(matches!(
        checkpoint.active_run.as_ref().map(|run| &run.resume_from),
        Some(RunResumePoint::ModelAttempt { effect_ids, .. }) if effect_ids.is_empty()
    ));
    drop(before);

    let session_id = checkpoint.session_id.clone();
    let mut after = reopen(&session_id, checkpoint).await;
    assert!(matches!(
        after
            .session
            .enqueue(MailboxInput::next_model_request(vec![input(
                "operator",
                "while resumable"
            )]))
            .await
            .unwrap(),
        Enqueued::Held(HoldReason::ResumeRequired)
    ));
    assert!(after.session.pending_reconciliation().await.is_empty());
    let request = resume_and_finish(&mut after, "resumed").await;
    assert_eq!(
        visible_texts(&request.transcript),
        ["base", "base-constraint", "before crash", "while resumable"]
    );
    assert!(!request.transcript.iter().any(is_recovery_receipt));
}

/// 表第 3 行：一个调用已交给运行时（账本 InFlight），模型还在流。
///
/// 重开：这个槽必须先对账；对账成 Settled 后 `try_resume` 把它写成回执再从头请求。
#[tokio::test]
async fn crash_after_delivery_requires_reconciliation_then_resumes_with_a_receipt() {
    let mut before = before("after-delivery", None).await;
    let _handle = start(&before.session, vec![input("operator", "deliver")])
        .await
        .unwrap();
    before.requests.recv().await.unwrap();
    before
        .commands
        .send(StreamCommand::Event(ModelStreamEvent::ToolCallReady {
            slot: ToolCallSlot::new(0),
            call: read_call("in-flight"),
        }))
        .unwrap();
    assert!(matches!(
        before.updates.recv().await.unwrap(),
        IncrementalUpdate::Began(_)
    ));
    assert!(matches!(
        before.updates.recv().await.unwrap(),
        IncrementalUpdate::Submitted(slot, _) if slot.get() == 0
    ));
    // Persist 排在 Submit 之前：运行时拿到调用时，InFlight 一定已经在盘上。
    let checkpoint = before.store.latest();
    let effect_ids = match checkpoint.active_run.as_ref().map(|run| &run.resume_from) {
        Some(RunResumePoint::ModelAttempt { effect_ids, .. }) => effect_ids.clone(),
        other => panic!("交付后应停在 ModelAttempt：{other:?}"),
    };
    assert_eq!(effect_ids.len(), 1);
    assert!(matches!(
        checkpoint
            .effects
            .get(&effect_ids[0])
            .map(|record| &record.state),
        Some(EffectState::InFlight { .. })
    ));
    drop(before);

    let session_id = checkpoint.session_id.clone();
    let mut after = reopen(&session_id, checkpoint).await;
    // 未知副作用挡在前面：连续跑都不行。
    assert!(matches!(
        after
            .session
            .enqueue(MailboxInput::next_model_request(vec![input(
                "operator", "blocked"
            )]))
            .await
            .unwrap(),
        Enqueued::Held(HoldReason::ReconciliationRequired)
    ));
    let pending = after.session.pending_reconciliation().await;
    assert_eq!(pending.len(), 1);
    let Err(error) = after.session.try_resume().await else {
        panic!("未知副作用没对账之前不能续跑")
    };
    assert_eq!(error.kind, AgentErrorKind::ReconciliationRequired);

    after
        .session
        .reconcile_effect(
            &effect_ids[0],
            DurableEffectOutcome::Settled(ToolResult::success_json(
                ToolCallId::new("in-flight"),
                json!({"recovered": true}),
            )),
        )
        .await
        .unwrap();
    // 半截运行还在：对账不会替它开跑，也不会让 blocked 那条越过它。
    assert!(after.session.mailbox_snapshot().await.len() == 1);
    let request = resume_and_finish(&mut after, "resumed").await;
    assert_eq!(
        visible_texts(&request.transcript),
        ["base", "base-constraint", "deliver", "blocked"]
    );
    assert!(request
        .transcript
        .iter()
        .any(|item| has_json_fact_kind(item, "effect_reconciliation_receipt")));
    assert!(request
        .transcript
        .iter()
        .any(|item| has_json_fact_kind(item, "interrupted_tool_batch_receipt")));
    assert!(tool_call_ids(&request.transcript).is_empty());
}

/// 表第 4 行：交付的槽已经提前结算（`settled` 返回 Ok = 已落盘），模型还在流。
///
/// 重开不需要对账：结算是终态事实，`try_resume` 直接把它写成回执。
#[tokio::test]
async fn crash_after_early_settlement_keeps_the_fact_without_reconciliation() {
    let mut before = before("after-settlement", Some(0)).await;
    let _handle = start(&before.session, vec![input("operator", "settle early")])
        .await
        .unwrap();
    before.requests.recv().await.unwrap();
    before
        .commands
        .send(StreamCommand::Event(ModelStreamEvent::ToolCallReady {
            slot: ToolCallSlot::new(0),
            call: read_call("early"),
        }))
        .unwrap();
    assert!(matches!(
        before.updates.recv().await.unwrap(),
        IncrementalUpdate::Began(_)
    ));
    // `Submitted` 在运行时的 submit 里、上报被确认之后才发（见测试替身）：确认 = 已落盘。
    assert!(matches!(
        before.updates.recv().await.unwrap(),
        IncrementalUpdate::Submitted(_, _)
    ));
    let checkpoint = before.store.latest();
    let settled_ids: Vec<_> = checkpoint
        .effects
        .iter()
        .filter(|(_, record)| {
            matches!(
                record.state,
                EffectState::Outcome {
                    outcome: DurableEffectOutcome::Settled(_)
                }
            )
        })
        .map(|(id, _)| id.clone())
        .collect();
    assert_eq!(settled_ids.len(), 1, "提前结算必须已经在盘上");
    drop(before);

    let session_id = checkpoint.session_id.clone();
    let mut after = reopen(&session_id, checkpoint).await;
    assert!(after.session.pending_reconciliation().await.is_empty());
    let request = resume_and_finish(&mut after, "resumed").await;
    let receipt = request
        .transcript
        .iter()
        .find(|item| has_json_fact_kind(item, "interrupted_tool_batch_receipt"))
        .expect("已结算的事实必须进回执");
    let TranscriptItem::Input(message) = receipt else {
        unreachable!()
    };
    let ContentPart::Json { value } = &message.content[0] else {
        panic!("回执应为结构化 JSON")
    };
    assert_eq!(value["calls"][0]["call_id"], "early");
    assert_eq!(value["calls"][0]["outcome"]["status"], "settled");
    assert!(!request
        .transcript
        .iter()
        .any(|item| has_json_fact_kind(item, "effect_reconciliation_receipt")));
}

/// `Aborting`：断流后 `Runtime::Abort` 已发，运行时的报告还没回来。
///
/// 快照把它投影成 `ModelAttempt`（InFlight 引用在），重开先对账再续。
#[tokio::test]
async fn crash_while_aborting_treats_the_slot_as_in_flight_and_reconciles() {
    let gate = Arc::new(Semaphore::new(0));
    let mut before = before_with("aborting", |updates| EagerIncrementalRuntime {
        abort_gate: Some(gate.clone()),
        ..EagerIncrementalRuntime::new(updates, 0)
    })
    .await;
    let _handle = start(&before.session, vec![input("operator", "abort in flight")])
        .await
        .unwrap();
    before.requests.recv().await.unwrap();
    before
        .commands
        .send(StreamCommand::Event(ModelStreamEvent::ToolCallReady {
            slot: ToolCallSlot::new(0),
            call: read_call("frozen"),
        }))
        .unwrap();
    assert!(matches!(
        before.updates.recv().await.unwrap(),
        IncrementalUpdate::Began(_)
    ));
    assert!(matches!(
        before.updates.recv().await.unwrap(),
        IncrementalUpdate::Submitted(_, _)
    ));
    before
        .commands
        .send(StreamCommand::Fail(AgentError::new(
            AgentErrorKind::Model,
            "stream dropped",
        )))
        .unwrap();
    // 运行时进了 abort()，卡在闸上。从 CallingModel 进 Aborting 投影不变（都是 ModelAttempt，
    // InFlight 引用在交付时就落了盘），所以这里取到的 checkpoint 就是重开会看到的。
    assert!(matches!(
        before.updates.recv().await.unwrap(),
        IncrementalUpdate::Aborted(ToolBatchAbortReason::ModelStreamInterrupted)
    ));
    let checkpoint = before.store.latest();
    let effect_ids = match checkpoint.active_run.as_ref().map(|run| &run.resume_from) {
        Some(RunResumePoint::ModelAttempt { effect_ids, .. }) => effect_ids.clone(),
        other => panic!("Aborting 应投影成 ModelAttempt：{other:?}"),
    };
    assert_eq!(effect_ids.len(), 1);
    drop(before);

    let session_id = checkpoint.session_id.clone();
    let mut after = reopen(&session_id, checkpoint).await;
    assert_eq!(after.session.pending_reconciliation().await.len(), 1);
    after
        .session
        .reconcile_effect(&effect_ids[0], DurableEffectOutcome::CancelledBeforeStart)
        .await
        .unwrap();
    let request = resume_and_finish(&mut after, "resumed").await;
    // 确定没开始：除了对账结论本身，没有别的回执要写。
    assert!(request
        .transcript
        .iter()
        .any(|item| has_json_fact_kind(item, "effect_reconciliation_receipt")));
    assert!(!request
        .transcript
        .iter()
        .any(|item| has_json_fact_kind(item, "interrupted_tool_batch_receipt")));
    assert!(tool_call_ids(&request.transcript).is_empty());
}

/// Commit 在飞时取消：`WaitingTools → Aborting` 改变了快照投影（ToolBatch → ModelAttempt），
/// 必须落盘。否则崩溃重开会把一次被中止的尝试当成可重建的正常工具轮。
#[tokio::test]
async fn crash_after_cancelling_a_commit_in_flight_persists_the_aborting_projection() {
    let gate = Arc::new(Semaphore::new(0));
    let mut before = before_with("cancel-commit-in-flight", |updates| {
        EagerIncrementalRuntime {
            commit_gate: Some(gate.clone()),
            ..EagerIncrementalRuntime::new(updates, 0)
        }
    })
    .await;
    let handle = start(&before.session, vec![input("operator", "cancel me")])
        .await
        .unwrap();
    before.requests.recv().await.unwrap();
    let call = read_call("in-commit");
    before
        .commands
        .send(StreamCommand::Event(ModelStreamEvent::ToolCallReady {
            slot: ToolCallSlot::new(0),
            call: call.clone(),
        }))
        .unwrap();
    assert!(matches!(
        before.updates.recv().await.unwrap(),
        IncrementalUpdate::Began(_)
    ));
    assert!(matches!(
        before.updates.recv().await.unwrap(),
        IncrementalUpdate::Submitted(_, _)
    ));
    before
        .commands
        .send(StreamCommand::Event(ModelStreamEvent::ToolCallsSealed {
            call_count: 1,
        }))
        .unwrap();
    assert!(matches!(
        before.updates.recv().await.unwrap(),
        IncrementalUpdate::CallsSealed(1)
    ));
    before
        .commands
        .send(StreamCommand::Complete(model_response(ModelOutput::calls(
            vec![call],
        ))))
        .unwrap();
    assert!(matches!(
        before.updates.recv().await.unwrap(),
        IncrementalUpdate::Committed
    ));
    assert!(matches!(
        before
            .store
            .latest()
            .active_run
            .as_ref()
            .map(|run| &run.resume_from),
        Some(RunResumePoint::ToolBatch { .. })
    ));
    // 取消：Abort 发出去了（所有者卡在 commit 里，Abort 排在后面永远轮不到）；
    // 但 Aborting 的投影必须已经落盘。
    handle.cancel();
    let mut aborting = None;
    for _ in 0..200 {
        if let Some(RunResumePoint::ModelAttempt { effect_ids, .. }) = before
            .store
            .latest()
            .active_run
            .as_ref()
            .map(|run| run.resume_from.clone())
        {
            aborting = Some(effect_ids);
            break;
        }
        tokio::task::yield_now().await;
    }
    let effect_ids = aborting.expect("取消后最新的 checkpoint 应是 Aborting 的投影：ModelAttempt");
    assert_eq!(effect_ids.len(), 1);
    let checkpoint = before.store.latest();
    drop(before);

    let session_id = checkpoint.session_id.clone();
    let mut after = reopen(&session_id, checkpoint).await;
    // 重开：这一槽是「在途」，先对账；对账成 Settled 后进回执、从头请求——不是完整工具轮。
    assert_eq!(after.session.pending_reconciliation().await.len(), 1);
    after
        .session
        .reconcile_effect(
            &effect_ids[0],
            DurableEffectOutcome::Settled(ToolResult::success_json(
                ToolCallId::new("in-commit"),
                json!({"ran": true}),
            )),
        )
        .await
        .unwrap();
    let request = resume_and_finish(&mut after, "resumed").await;
    assert!(
        tool_call_ids(&request.transcript).is_empty(),
        "被中止的尝试不是工具轮"
    );
    assert!(request
        .transcript
        .iter()
        .any(|item| has_json_fact_kind(item, "interrupted_tool_batch_receipt")));
}
