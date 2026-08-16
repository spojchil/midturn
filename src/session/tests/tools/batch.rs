//! 整批模式：一次转发、dispatch 失败、运行时 panic。

use super::super::*;

#[tokio::test]
async fn complete_tool_array_is_forwarded_once_and_steering_waits_for_it() {
    let mut fixture = fixture();
    let session = fixture.session.clone();
    let task = tokio::spawn(async move { run_once(&session, vec![input("operator", "go")]).await });

    let first = fixture.requests.recv().await.unwrap();
    assert_eq!(
        visible_texts(&first.transcript),
        vec!["base", "base-constraint", "go"]
    );
    fixture
        .responses
        .send(Ok(model_response(ModelOutput::calls(vec![
            ToolCall::new("call-a", "read", json!({"path": "a"})),
            ToolCall::new("call-b", "read", json!({"path": "b"})),
        ]))))
        .unwrap();

    let batch = fixture.batches.recv().await.unwrap();
    assert_eq!(batch.calls.len(), 2);
    post(
        &fixture.session,
        MailboxInput::next_model_request(vec![input("developer", "steer")]),
    )
    .await
    .unwrap();
    fixture.gate.add_permits(1);

    let second = fixture.requests.recv().await.unwrap();
    assert_eq!(visible_texts(&second.transcript).last().unwrap(), "steer");
    let TranscriptItem::ToolResults(results) = &second.transcript[4] else {
        panic!("tool batch should precede mailbox input");
    };
    assert_eq!(results.results[0].call_id.as_str(), "call-a");
    assert_eq!(results.results[1].call_id.as_str(), "call-b");
    assert_eq!(fixture.runtime.dispatches.load(Ordering::SeqCst), 1);

    fixture
        .responses
        .send(Ok(model_response(ModelOutput::text("done"))))
        .unwrap();
    let outcome = task.await.unwrap().unwrap();
    assert!(
        matches!(outcome, TurnOutcome::Completed { output, .. } if output.text_content() == "done")
    );
}

#[tokio::test]
async fn dispatch_failure_does_not_leak_an_orphaned_tool_call_into_the_next_run() {
    let (request_tx, mut request_rx) = mpsc::unbounded_channel();
    let (response_tx, response_rx) = mpsc::unbounded_channel();
    let runtime = Arc::new(FailingBatchRuntime::default());
    let session = Arc::new(AgentSession::new(
        Arc::new(StaticPrompt),
        runtime.clone(),
        Arc::new(NoCompaction),
        Arc::new(ChannelModel {
            requests: request_tx,
            responses: Mutex::new(response_rx),
        }),
        SessionConfig::default(),
    ));

    let running = session.clone();
    let first_run =
        tokio::spawn(async move { run_once(&running, vec![input("operator", "first run")]).await });
    request_rx.recv().await.unwrap();
    response_tx
        .send(Ok(model_response(ModelOutput::calls(vec![ToolCall::new(
            "orphan-call",
            "read",
            json!({"path": "missing"}),
        )]))))
        .unwrap();

    let outcome = first_run.await.unwrap().unwrap();
    let TurnOutcome::Failed { error, .. } = outcome else {
        panic!("dispatch 顶层错误应终止当前运行");
    };
    assert_eq!(error.kind, AgentErrorKind::ReconciliationRequired);
    assert_eq!(
        error.summary,
        "tool_batch_outcome_requires_reconciliation:dispatch infrastructure failed"
    );
    assert_eq!(runtime.dispatches.load(Ordering::SeqCst), 1);

    let conversation = session.conversation().await;
    assert!(
        tool_call_ids(&conversation).is_empty(),
        "失败运行不应持久化尚未配对的模型工具调用"
    );

    // 输入照收，但要说清楚为什么没人来取。
    let second = MailboxInput::next_model_request(vec![input("operator", "second run")]);
    assert!(matches!(
        session.enqueue(second).await.unwrap(),
        Enqueued::Held(HoldReason::ReconciliationRequired)
    ));
    assert_eq!(reconcile_all_as_cancelled(&session).await, 1);

    // 对账之后，刚才那条输入还在信箱里；最后一次对账清掉阻塞就等于重新武装，
    // 不需要额外的入口。
    let second_request = request_rx.recv().await.unwrap();
    assert!(
        tool_call_ids(&second_request.transcript).is_empty(),
        "新运行发送给模型的 transcript 不应包含上轮孤儿调用"
    );
    response_tx
        .send(Ok(model_response(ModelOutput::text("recovered"))))
        .unwrap();

    session.wait_until_idle().await;
    let conversation = session.conversation().await;
    assert_eq!(
        visible_texts(&conversation).last().map(String::as_str),
        Some("recovered")
    );
}

#[tokio::test]
async fn later_dispatch_failure_preserves_only_the_preceding_complete_tool_round() {
    let (request_tx, mut request_rx) = mpsc::unbounded_channel();
    let (response_tx, response_rx) = mpsc::unbounded_channel();
    let runtime = Arc::new(FailingSecondBatchRuntime::default());
    let session = Arc::new(AgentSession::new(
        Arc::new(StaticPrompt),
        runtime.clone(),
        Arc::new(NoCompaction),
        Arc::new(ChannelModel {
            requests: request_tx,
            responses: Mutex::new(response_rx),
        }),
        SessionConfig::default(),
    ));

    let running = session.clone();
    let run =
        tokio::spawn(
            async move { run_once(&running, vec![input("operator", "two batches")]).await },
        );
    request_rx.recv().await.unwrap();
    response_tx
        .send(Ok(model_response(ModelOutput::calls(vec![ToolCall::new(
            "complete-call",
            "read",
            json!({}),
        )]))))
        .unwrap();

    let continuation = request_rx.recv().await.unwrap();
    assert_eq!(tool_call_ids(&continuation.transcript), ["complete-call"]);
    response_tx
        .send(Ok(model_response(ModelOutput::calls(vec![ToolCall::new(
            "failed-call",
            "read",
            json!({}),
        )]))))
        .unwrap();

    let outcome = run.await.unwrap().unwrap();
    assert!(matches!(
        outcome,
        TurnOutcome::Failed { error, .. }
            if error.kind == AgentErrorKind::ReconciliationRequired
                && error.summary == "tool_batch_outcome_requires_reconciliation:second dispatch failed"
    ));
    assert_eq!(runtime.dispatches.load(Ordering::SeqCst), 2);

    let conversation = session.conversation().await;
    assert_eq!(tool_call_ids(&conversation), ["complete-call"]);
    assert!(matches!(
        conversation.get(2),
        Some(TranscriptItem::ToolResults(_))
    ));
    assert_eq!(reconcile_all_as_cancelled(&session).await, 1);
}

#[tokio::test]
async fn tool_runtime_panic_fails_the_run_and_leaves_the_batch_for_reconciliation() {
    let (request_tx, mut request_rx) = mpsc::unbounded_channel();
    let (response_tx, response_rx) = mpsc::unbounded_channel();
    let session = Arc::new(AgentSession::new(
        Arc::new(StaticPrompt),
        Arc::new(PanickingBatchRuntime),
        Arc::new(NoCompaction),
        Arc::new(ChannelModel {
            requests: request_tx,
            responses: Mutex::new(response_rx),
        }),
        SessionConfig::default(),
    ));

    let running = session.clone();
    let panicking_run =
        tokio::spawn(
            async move { run_once(&running, vec![input("operator", "trigger panic")]).await },
        );
    request_rx.recv().await.unwrap();
    response_tx
        .send(Ok(model_response(ModelOutput::calls(vec![ToolCall::new(
            "panic-call",
            "read",
            json!({}),
        )]))))
        .unwrap();

    // 运行时的 panic 由 supervisor 收口成一次失败的 dispatch：这批工具的结果不可知，
    // 运行以「需要对账」失败，而不是留下一个被遗弃的半截运行。
    let TurnOutcome::Failed { error, stage, .. } = panicking_run.await.unwrap().unwrap() else {
        panic!("工具实现 panic 后运行应当失败")
    };
    assert_eq!(stage, RunStage::Tools);
    assert_eq!(error.kind, AgentErrorKind::ReconciliationRequired);
    let conversation = session.conversation().await;
    assert_eq!(visible_texts(&conversation), ["trigger panic"]);
    assert!(tool_call_ids(&conversation).is_empty());
    assert!(!session.has_active_run().await);

    assert!(matches!(
        session
            .enqueue(MailboxInput::next_model_request(vec![input(
                "operator",
                "must not skip reconciliation"
            )]))
            .await
            .unwrap(),
        Enqueued::Held(HoldReason::ReconciliationRequired)
    ));
    // 最后一次对账清掉阻塞就等于重新武装：被扣住的输入直接进入下一次运行。
    assert_eq!(reconcile_all_as_cancelled(&session).await, 1);
    assert!(session.try_resume().await.unwrap().is_none());

    let next_request = request_rx.recv().await.unwrap();
    assert!(tool_call_ids(&next_request.transcript).is_empty());
    assert_eq!(
        visible_texts(&next_request.transcript),
        [
            "base",
            "base-constraint",
            "trigger panic",
            "must not skip reconciliation",
        ]
    );
    assert!(next_request
        .transcript
        .iter()
        .any(|item| has_json_fact_kind(item, "effect_reconciliation_receipt")));
    response_tx
        .send(Ok(model_response(ModelOutput::text("recovered"))))
        .unwrap();
    session.wait_until_idle().await;
    assert_eq!(
        visible_texts(&session.conversation().await)
            .last()
            .map(String::as_str),
        Some("recovered")
    );
}
