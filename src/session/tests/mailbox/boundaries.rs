//! 边界语义：NextModelRequest / WhenIdle / Passive 各自在哪个边界进入。

use super::*;

#[tokio::test]
async fn steering_arriving_during_final_model_call_reopens_the_run() {
    let mut fixture = fixture();
    let session = fixture.session.clone();
    let task =
        tokio::spawn(async move { run_once(&session, vec![input("operator", "draft")]).await });
    fixture.requests.recv().await.unwrap();

    post(
        &fixture.session,
        MailboxInput::next_model_request(vec![input("reviewer", "revise")]),
    )
    .await
    .unwrap();
    fixture
        .responses
        .send(Ok(model_response(ModelOutput::text("candidate"))))
        .unwrap();

    let second = fixture.requests.recv().await.unwrap();
    assert_eq!(
        visible_texts(&second.transcript),
        vec!["base", "base-constraint", "draft", "candidate", "revise"]
    );
    fixture
        .responses
        .send(Ok(model_response(ModelOutput::text("revised"))))
        .unwrap();
    let outcome = task.await.unwrap().unwrap();
    assert!(
        matches!(outcome, TurnOutcome::Completed { output, .. } if output.text_content() == "revised")
    );
}

#[tokio::test]
async fn when_idle_waits_through_tool_continuation() {
    let mut fixture = fixture();
    let session = fixture.session.clone();
    let task = tokio::spawn(async move { run_once(&session, vec![input("operator", "go")]).await });
    fixture.requests.recv().await.unwrap();
    fixture
        .responses
        .send(Ok(model_response(ModelOutput::calls(vec![ToolCall::new(
            "call-a",
            "read",
            json!({}),
        )]))))
        .unwrap();
    fixture.batches.recv().await.unwrap();
    post(
        &fixture.session,
        MailboxInput::when_idle(vec![input("operator", "follow up")]),
    )
    .await
    .unwrap();
    fixture.gate.add_permits(1);

    let after_tools = fixture.requests.recv().await.unwrap();
    assert!(!visible_texts(&after_tools.transcript).contains(&"follow up".to_owned()));
    fixture
        .responses
        .send(Ok(model_response(ModelOutput::text("first answer"))))
        .unwrap();

    let follow_up = fixture.requests.recv().await.unwrap();
    assert_eq!(
        visible_texts(&follow_up.transcript).last().unwrap(),
        "follow up"
    );
    fixture
        .responses
        .send(Ok(model_response(ModelOutput::text("second answer"))))
        .unwrap();
    let outcome = task.await.unwrap().unwrap();
    assert!(
        matches!(outcome, TurnOutcome::Completed { output, .. } if output.text_content() == "second answer")
    );
}

#[tokio::test]
async fn passive_arriving_mid_run_does_not_cause_another_model_request() {
    // Passive 必须在开跑**之后**投：开跑前放好的会在第一个 BeforeModelRequest 边界合法
    // 搭车，永远走不到「完成边界只剩 Passive」这条路径。
    let mut fixture = fixture();
    let handle = start(&fixture.session, vec![input("operator", "go")])
        .await
        .unwrap();
    fixture.requests.recv().await.unwrap();

    // 模型正在生成时到达：此刻已经过了本轮唯一的模型请求边界。
    assert!(matches!(
        fixture
            .session
            .enqueue(MailboxInput::passive(vec![input("tool", "late result")]))
            .await
            .unwrap(),
        Enqueued::Held(HoldReason::Passive)
    ));
    fixture
        .responses
        .send(Ok(model_response(ModelOutput::text("done"))))
        .unwrap();

    // 完成边界上只剩 Passive：它不构成再请求一次模型的理由，这一轮就该结束。
    assert!(matches!(handle.join().await, TurnOutcome::Completed { .. }));
    assert!(
        fixture.requests.try_recv().is_err(),
        "被动内容不应把本该结束的运行拉回一次模型请求"
    );
    // 也不该被那次排空顺手带走：它还在信箱里等下一次真正的请求。
    assert_eq!(
        fixture.session.mailbox_snapshot().await,
        vec![MailboxInput::passive(vec![input("tool", "late result")])]
    );
}

#[tokio::test]
async fn passive_delivery_is_kept_but_never_wakes_an_idle_session() {
    let mut fixture = fixture();

    // 被动投递照收、照样进 checkpoint，但不开跑。
    let held = fixture
        .session
        .enqueue(MailboxInput::passive(vec![input("tool", "late result")]))
        .await
        .unwrap();
    assert!(matches!(held, Enqueued::Held(HoldReason::Passive)));
    assert!(
        fixture.requests.try_recv().is_err(),
        "被动投递不应触发模型请求"
    );
    assert_eq!(fixture.session.mailbox_snapshot().await.len(), 1);

    // 下一次有别的原因开跑时，它搭便车进入同一次请求。
    let handle = start(&fixture.session, vec![input("operator", "go")])
        .await
        .unwrap();
    let request = fixture.requests.recv().await.unwrap();
    assert_eq!(
        visible_texts(&request.transcript),
        ["base", "base-constraint", "go", "late result"]
    );
    fixture
        .responses
        .send(Ok(model_response(ModelOutput::text("done"))))
        .unwrap();
    handle.join().await;
    // 搭完便车就没了，不会自己再要一次请求。
    assert!(fixture.requests.try_recv().is_err());
}

#[tokio::test]
async fn independently_closed_mailbox_segments_compose_even_when_ids_repeat() {
    let mut fixture = fixture();
    let running = fixture.session.clone();
    let task = tokio::spawn(async move {
        run_once(&running, vec![input("operator", "compose segments")]).await
    });
    fixture.requests.recv().await.unwrap();

    post(
        &fixture.session,
        MailboxInput::next_model_request(complete_tool_round("reused-history-id")),
    )
    .await
    .unwrap();
    post(
        &fixture.session,
        MailboxInput::next_model_request(complete_tool_round("reused-history-id")),
    )
    .await
    .unwrap();
    fixture
        .responses
        .send(Ok(model_response(ModelOutput::text("candidate"))))
        .unwrap();

    let continuation = fixture.requests.recv().await.unwrap();
    assert_eq!(
        tool_call_ids(&continuation.transcript)
            .iter()
            .filter(|id| id.as_str() == "reused-history-id")
            .count(),
        2
    );
    fixture
        .responses
        .send(Ok(model_response(ModelOutput::text("done"))))
        .unwrap();
    assert!(matches!(
        task.await.unwrap().unwrap(),
        TurnOutcome::Completed { output, .. } if output.text_content() == "done"
    ));
}

/// 续跑的是同一个未完成的运行：`WhenIdle` 内容要等**它**跑完，不像空闲开跑那样第一个边界就投递。
#[tokio::test]
async fn resumed_run_keeps_when_idle_for_its_own_completion_boundary() {
    let store = Arc::new(InMemoryCheckpointStore::new());
    let session_id = SessionId::new("mailbox/resume-when-idle").unwrap();
    let (request_tx, mut requests) = mpsc::unbounded_channel();
    let (_response_tx, response_rx) = mpsc::unbounded_channel();
    let first = Arc::new(
        AgentSession::open(
            session_id.clone(),
            store.clone(),
            Arc::new(StaticPrompt),
            Arc::new(PanickingBatchRuntime),
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
    // 开跑、模型在飞；这时投一条 WhenIdle。然后「崩溃」（丢掉实例）。
    let _handle = start(&first, vec![input("operator", "first")])
        .await
        .unwrap();
    requests.recv().await.unwrap();
    post(
        &first,
        MailboxInput::when_idle(vec![input("user", "after it")]),
    )
    .await
    .unwrap();
    drop(first);

    let (request_tx, mut requests) = mpsc::unbounded_channel();
    let (responses, response_rx) = mpsc::unbounded_channel();
    let reopened = Arc::new(
        AgentSession::open(
            session_id,
            store,
            Arc::new(StaticPrompt),
            Arc::new(PanickingBatchRuntime),
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
    let handle = reopened.try_resume().await.unwrap().expect("半截运行");
    let request = requests.recv().await.unwrap();
    assert_eq!(
        visible_texts(&request.transcript),
        ["base", "base-constraint", "first"],
        "续跑的第一次请求不带 WhenIdle：那是「等它跑完」"
    );
    responses
        .send(Ok(model_response(ModelOutput::text("answer"))))
        .unwrap();
    // 完成边界把它带上，重开一次请求。
    let request = requests.recv().await.unwrap();
    assert!(visible_texts(&request.transcript).contains(&"after it".to_owned()));
    responses
        .send(Ok(model_response(ModelOutput::text("done"))))
        .unwrap();
    assert!(matches!(
        handle.join().await,
        TurnOutcome::Completed { output, .. } if output.text_content() == "done"
    ));
}
