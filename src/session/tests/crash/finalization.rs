//! 崩溃点：回执已进历史、Unknown 收尾、收尾压缩在飞。

use super::*;

/// 表第 6 行：中断回执已经写进历史（`BeforeModelRequest`），下一次模型请求还没发。
///
/// 重开：什么都不用对账，`try_resume` 直接从边界继续；回执只出现一次。
#[tokio::test]
async fn crash_after_a_receipt_resumes_from_the_boundary_without_duplicating_it() {
    // 运行时在 submit 时就把这一槽结算掉；随后断流，abort 报告里没有它——回执里有。
    let mut before = before("after-receipt-settled", Some(0)).await;
    let _handle = start(&before.session, vec![input("operator", "interrupt")])
        .await
        .unwrap();
    before.requests.recv().await.unwrap();
    before
        .commands
        .send(StreamCommand::Event(ModelStreamEvent::ToolCallReady {
            slot: ToolCallSlot::new(0),
            call: read_call("settled-then-interrupted"),
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
    assert!(matches!(
        before.updates.recv().await.unwrap(),
        IncrementalUpdate::Aborted(ToolBatchAbortReason::ModelStreamInterrupted)
    ));
    // 回执写进历史、再请求模型：第二次请求到达时，盘上是 BeforeModelRequest 之后的
    // ModelAttempt——两者都含回执。取「第二次请求刚发出」那一刻的 checkpoint。
    let second = before.requests.recv().await.unwrap();
    assert!(second.transcript.iter().any(is_recovery_receipt));
    let checkpoint = before.store.latest();
    assert_eq!(
        checkpoint
            .conversation
            .iter()
            .filter(|item| is_recovery_receipt(item))
            .count(),
        1
    );
    drop(before);

    let session_id = checkpoint.session_id.clone();
    let mut after = reopen(&session_id, checkpoint).await;
    assert!(after.session.pending_reconciliation().await.is_empty());
    let request = resume_and_finish(&mut after, "resumed").await;
    assert_eq!(
        request
            .transcript
            .iter()
            .filter(|item| is_recovery_receipt(item))
            .count(),
        1,
        "回执是历史的一部分，重开不再造一份"
    );
    assert_eq!(
        visible_texts(&request.transcript),
        ["base", "base-constraint", "interrupt"]
    );
}

/// 回执带着 Unknown、本轮已 `Failed(ReconciliationRequired)`、然后崩溃。
///
/// 重开：没有半截运行；未知态挡着，`Held(ReconciliationRequired)`；对账清掉就自己开跑，
/// 第一次请求同时带着中断回执和对账结论。
#[tokio::test]
async fn crash_after_an_unknown_receipt_reopens_into_reconciliation_then_runs_by_itself() {
    let mut before = before_with("unknown-receipt", |updates| EagerIncrementalRuntime {
        unknown_on_abort: true,
        ..EagerIncrementalRuntime::new(updates, 0)
    })
    .await;
    let handle = start(&before.session, vec![input("operator", "unknown")])
        .await
        .unwrap();
    before.requests.recv().await.unwrap();
    before
        .commands
        .send(StreamCommand::Event(ModelStreamEvent::ToolCallReady {
            slot: ToolCallSlot::new(0),
            call: read_call("maybe-ran"),
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
    let outcome = handle.join().await;
    assert!(matches!(
        outcome,
        TurnOutcome::Failed { ref error, .. }
            if error.kind == AgentErrorKind::ReconciliationRequired
    ));
    let checkpoint = before.store.latest();
    assert!(checkpoint.active_run.is_none());
    assert!(checkpoint.conversation.iter().any(is_recovery_receipt));
    drop(before);

    let session_id = checkpoint.session_id.clone();
    let mut after = reopen(&session_id, checkpoint).await;
    assert!(after.session.try_resume().await.unwrap().is_none());
    assert!(matches!(
        after
            .session
            .enqueue(MailboxInput::next_model_request(vec![input(
                "operator", "next"
            )]))
            .await
            .unwrap(),
        Enqueued::Held(HoldReason::ReconciliationRequired)
    ));
    let pending = after.session.pending_reconciliation().await;
    assert_eq!(pending.len(), 1);
    assert_eq!(reconcile_all_as_cancelled(&after.session).await, 1);
    // 对账清掉阻塞 = 重新武装：不用任何额外入口。
    let request = after.requests.recv().await.unwrap();
    assert_eq!(
        visible_texts(&request.transcript),
        ["base", "base-constraint", "unknown", "next"]
    );
    assert!(request
        .transcript
        .iter()
        .any(|item| has_json_fact_kind(item, "interrupted_tool_batch_receipt")));
    assert!(request
        .transcript
        .iter()
        .any(|item| has_json_fact_kind(item, "effect_reconciliation_receipt")));
    after
        .responses
        .send(Ok(model_response(ModelOutput::text("recovered"))))
        .unwrap();
    after.session.wait_until_idle().await;
}

/// 收尾压缩在飞时崩溃：盘上是 `BeforeCompletion{output}`。
///
/// 重开不采用任何未确认的压缩结果：`try_resume` 从完成边界继续，信箱空着就直接完成，
/// 对话是原记录。
#[tokio::test]
async fn crash_during_final_compaction_resumes_at_the_completion_boundary_unchanged() {
    let session_id = SessionId::new("crash/final-compaction").unwrap();
    let store = Arc::new(RecordingStore::default());
    let (request_tx, mut requests) = mpsc::unbounded_channel();
    let (responses, response_rx) = mpsc::unbounded_channel();
    let (compaction_tx, mut compactions) = mpsc::unbounded_channel();
    let session = Arc::new(
        AgentSession::open(
            session_id.clone(),
            store.clone(),
            Arc::new(StaticPrompt),
            Arc::new(FailingBatchRuntime::default()),
            Arc::new(GatedSummaryCompaction {
                started: compaction_tx,
                gate: Arc::new(Semaphore::new(0)),
            }),
            Arc::new(ChannelModel {
                requests: request_tx,
                responses: Mutex::new(response_rx),
            }),
            SessionConfig {
                budget: SessionBudget {
                    context: ContextBudget {
                        compact_above_bytes: 0,
                        ..ContextBudget::default()
                    },
                    ..SessionBudget::default()
                },
                ..SessionConfig::default()
            },
        )
        .await
        .unwrap(),
    );
    let _handle = start(&session, vec![input("operator", "compact me")])
        .await
        .unwrap();
    requests.recv().await.unwrap();
    responses
        .send(Ok(model_response(ModelOutput::text("final words"))))
        .unwrap();
    // 压缩已开始、卡在闸上：这一刻盘上是 BeforeCompletion。
    let pre = compactions.recv().await.unwrap();
    assert_eq!(visible_texts(&pre), ["compact me", "final words"]);
    let checkpoint = store.latest();
    assert!(matches!(
        checkpoint.active_run.as_ref().map(|run| &run.resume_from),
        Some(RunResumePoint::BeforeCompletion { output }) if output.text_content() == "final words"
    ));
    drop(session);

    // 重开时不压缩：直接完成，对话是原记录——没确认的压缩结果不算数。
    let after = reopen(&session_id, checkpoint.clone()).await;
    let handle = after
        .session
        .try_resume()
        .await
        .unwrap()
        .expect("完成边界上的半截运行可续");
    assert!(matches!(
        handle.join().await,
        TurnOutcome::Completed { output, .. } if output.text_content() == "final words"
    ));
    assert_eq!(
        visible_texts(&after.session.conversation().await),
        ["compact me", "final words"],
        "没确认的压缩结果不算数"
    );

    // 重开时仍按同一策略压缩：条件仍成立就再压一次，采用的是**这一次**确认的结果。
    let (compaction_tx, mut compactions) = mpsc::unbounded_channel();
    let gate = Arc::new(Semaphore::new(1));
    let after = reopen_with(
        &session_id,
        checkpoint,
        Arc::new(GatedSummaryCompaction {
            started: compaction_tx,
            gate,
        }),
        SessionConfig {
            budget: SessionBudget {
                context: ContextBudget {
                    compact_above_bytes: 0,
                    ..ContextBudget::default()
                },
                ..SessionBudget::default()
            },
            ..SessionConfig::default()
        },
    )
    .await;
    let handle = after.session.try_resume().await.unwrap().unwrap();
    let again = compactions.recv().await.unwrap();
    assert_eq!(visible_texts(&again), ["compact me", "final words"]);
    assert!(matches!(
        handle.join().await,
        TurnOutcome::Completed { output, .. } if output.text_content() == "final words"
    ));
    assert_eq!(
        visible_texts(&after.session.conversation().await),
        ["summary"],
        "重开后重新压缩、采用这次确认的结果"
    );
    assert!(compactions.try_recv().is_err(), "只压一次");
}
