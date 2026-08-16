//! 崩溃点：模型说完、工具批次在飞（整批 dispatch / 增量 commit）。

use super::*;

/// 表第 5 行：模型说完了、整批已经交出去（`Dispatching`），结果还没回来。
///
/// 重开：整批 InFlight 全部对账；都 Settled 之后 `try_resume` 直接把它们拼成完整工具轮，
/// 模型看到的是正常的「调用 + 结果」，不是回执。
#[tokio::test]
async fn crash_while_dispatching_rebuilds_the_tool_round_from_reconciled_effects() {
    let session_id = SessionId::new("crash/dispatching").unwrap();
    let store = Arc::new(RecordingStore::default());
    let (request_tx, mut requests) = mpsc::unbounded_channel();
    let (responses, response_rx) = mpsc::unbounded_channel();
    let (batch_tx, mut batches) = mpsc::unbounded_channel();
    let session = Arc::new(
        AgentSession::open(
            session_id.clone(),
            store.clone(),
            Arc::new(StaticPrompt),
            Arc::new(GatedBatchRuntime {
                batches: batch_tx,
                gate: Arc::new(Semaphore::new(0)),
                dispatches: AtomicUsize::new(0),
            }),
            Arc::new(NoCompaction),
            Arc::new(ChannelModel {
                requests: request_tx,
                responses: Mutex::new(response_rx),
            }),
            SessionConfig::default(),
        )
        .await
        .unwrap(),
    );
    let _handle = start(&session, vec![input("operator", "dispatch")])
        .await
        .unwrap();
    requests.recv().await.unwrap();
    responses
        .send(Ok(model_response(ModelOutput::calls(vec![
            read_call("first"),
            read_call("second"),
        ]))))
        .unwrap();
    // 运行时拿到整批时，两个 InFlight 已经在盘上（Persist 先于 Dispatch）。
    batches.recv().await.unwrap();
    let checkpoint = store.latest();
    let effect_ids = match checkpoint.active_run.as_ref().map(|run| &run.resume_from) {
        Some(RunResumePoint::ToolBatch { effect_ids, .. }) => effect_ids.clone(),
        other => panic!("整批交出去后应停在 ToolBatch：{other:?}"),
    };
    assert_eq!(effect_ids.len(), 2);
    drop(session);

    let mut after = reopen(&session_id, checkpoint).await;
    assert_eq!(after.session.pending_reconciliation().await.len(), 2);
    for (id, call_id) in effect_ids.iter().zip(["first", "second"]) {
        after
            .session
            .reconcile_effect(
                id,
                DurableEffectOutcome::Settled(ToolResult::success_json(
                    ToolCallId::new(call_id),
                    json!({"reconciled": call_id}),
                )),
            )
            .await
            .unwrap();
    }
    let request = resume_and_finish(&mut after, "resumed").await;
    // 完整工具轮：调用与结果配对，按槽序。
    assert_eq!(tool_call_ids(&request.transcript), ["first", "second"]);
    let results = request
        .transcript
        .iter()
        .find_map(|item| match item {
            TranscriptItem::ToolResults(batch) => Some(batch),
            _ => None,
        })
        .expect("对账齐了就该有一轮完整的 ToolResults");
    assert_eq!(results.results.len(), 2);
    assert_eq!(results.results[0].call_id.as_str(), "first");
    assert_eq!(results.results[1].call_id.as_str(), "second");
    assert!(!request
        .transcript
        .iter()
        .any(|item| has_json_fact_kind(item, "interrupted_tool_batch_receipt")));
}

/// 表第 5 行的另一半：整批里有一个对账成「确定没开始」。
///
/// 那就不是一个完整工具轮了：改成回执，从头请求。
#[tokio::test]
async fn crash_while_dispatching_with_a_cancelled_slot_falls_back_to_a_receipt() {
    let session_id = SessionId::new("crash/dispatching-cancelled").unwrap();
    let store = Arc::new(RecordingStore::default());
    let (request_tx, mut requests) = mpsc::unbounded_channel();
    let (responses, response_rx) = mpsc::unbounded_channel();
    let (batch_tx, mut batches) = mpsc::unbounded_channel();
    let session = Arc::new(
        AgentSession::open(
            session_id.clone(),
            store.clone(),
            Arc::new(StaticPrompt),
            Arc::new(GatedBatchRuntime {
                batches: batch_tx,
                gate: Arc::new(Semaphore::new(0)),
                dispatches: AtomicUsize::new(0),
            }),
            Arc::new(NoCompaction),
            Arc::new(ChannelModel {
                requests: request_tx,
                responses: Mutex::new(response_rx),
            }),
            SessionConfig::default(),
        )
        .await
        .unwrap(),
    );
    let _handle = start(&session, vec![input("operator", "dispatch")])
        .await
        .unwrap();
    requests.recv().await.unwrap();
    responses
        .send(Ok(model_response(ModelOutput::calls(vec![
            read_call("ran"),
            read_call("never-ran"),
        ]))))
        .unwrap();
    batches.recv().await.unwrap();
    let checkpoint = store.latest();
    let effect_ids = match checkpoint.active_run.as_ref().map(|run| &run.resume_from) {
        Some(RunResumePoint::ToolBatch { effect_ids, .. }) => effect_ids.clone(),
        other => panic!("整批交出去后应停在 ToolBatch：{other:?}"),
    };
    drop(session);

    let mut after = reopen(&session_id, checkpoint).await;
    after
        .session
        .reconcile_effect(
            &effect_ids[0],
            DurableEffectOutcome::Settled(ToolResult::success_json(
                ToolCallId::new("ran"),
                json!({"ok": true}),
            )),
        )
        .await
        .unwrap();
    after
        .session
        .reconcile_effect(&effect_ids[1], DurableEffectOutcome::CancelledBeforeStart)
        .await
        .unwrap();
    let request = resume_and_finish(&mut after, "resumed").await;
    assert!(
        tool_call_ids(&request.transcript).is_empty(),
        "有一个没跑，就不能伪装成完整工具轮"
    );
    assert!(request
        .transcript
        .iter()
        .any(|item| has_json_fact_kind(item, "recovered_tool_batch_receipt")));
}

/// 表第 5 行的增量版：模型说完、Commit 已交给运行时、结果还没回来（`WaitingTools`）。
///
/// 盘上是 `ToolBatch{effect_ids: InFlight}`。重开：先对账；都 Settled 之后 `try_resume`
/// 直接拼成完整工具轮。
#[tokio::test]
async fn crash_while_incremental_commit_is_in_flight_rebuilds_the_round_after_reconciliation() {
    let gate = Arc::new(Semaphore::new(0));
    let mut before = before_with("incremental-commit", |updates| EagerIncrementalRuntime {
        commit_gate: Some(gate.clone()),
        ..EagerIncrementalRuntime::new(updates, 0)
    })
    .await;
    let _handle = start(&before.session, vec![input("operator", "commit in flight")])
        .await
        .unwrap();
    before.requests.recv().await.unwrap();
    let call = read_call("slow");
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
    // 运行时进了 commit()，卡在闸上：这一刻盘上是 WaitingTools 的投影。
    assert!(matches!(
        before.updates.recv().await.unwrap(),
        IncrementalUpdate::Committed
    ));
    let checkpoint = before.store.latest();
    let effect_ids = match checkpoint.active_run.as_ref().map(|run| &run.resume_from) {
        Some(RunResumePoint::ToolBatch { effect_ids, .. }) => effect_ids.clone(),
        other => panic!("模型说完、结果未回：应停在 ToolBatch：{other:?}"),
    };
    assert_eq!(effect_ids.len(), 1);
    drop(before);

    let session_id = checkpoint.session_id.clone();
    let mut after = reopen(&session_id, checkpoint).await;
    assert_eq!(after.session.pending_reconciliation().await.len(), 1);
    after
        .session
        .reconcile_effect(
            &effect_ids[0],
            DurableEffectOutcome::Settled(ToolResult::success_json(
                ToolCallId::new("slow"),
                json!({"finished": "elsewhere"}),
            )),
        )
        .await
        .unwrap();
    let request = resume_and_finish(&mut after, "resumed").await;
    assert_eq!(tool_call_ids(&request.transcript), ["slow"]);
    assert!(request.transcript.iter().any(
        |item| matches!(item, TranscriptItem::ToolResults(batch) if batch.results.len() == 1)
    ));
    assert!(!request
        .transcript
        .iter()
        .any(|item| has_json_fact_kind(item, "interrupted_tool_batch_receipt")));
}
