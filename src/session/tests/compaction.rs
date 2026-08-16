//! 压缩：边界压缩、收尾压缩、非法输出回退、压缩期间的信箱。

use super::*;

#[tokio::test]
async fn compaction_between_tool_rounds_replaces_history_before_draining_mailbox() {
    let (request_tx, mut request_rx) = mpsc::unbounded_channel();
    let (response_tx, response_rx) = mpsc::unbounded_channel();
    let (batch_tx, mut batch_rx) = mpsc::unbounded_channel();
    let tool_gate = Arc::new(Semaphore::new(0));
    let (compaction_tx, mut compaction_rx) = mpsc::unbounded_channel();
    let compaction_gate = Arc::new(Semaphore::new(0));
    let session = Arc::new(AgentSession::new(
        Arc::new(StaticPrompt),
        Arc::new(GatedBatchRuntime {
            batches: batch_tx,
            gate: tool_gate.clone(),
            dispatches: AtomicUsize::new(0),
        }),
        Arc::new(GatedSummaryCompaction {
            started: compaction_tx,
            gate: compaction_gate.clone(),
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
    ));

    let running = session.clone();
    let task = tokio::spawn(async move { run_once(&running, vec![input("operator", "go")]).await });
    let first = request_rx.recv().await.unwrap();
    assert_eq!(
        visible_texts(&first.transcript),
        ["base", "base-constraint", "go"]
    );
    response_tx
        .send(Ok(model_response(ModelOutput::calls(vec![ToolCall::new(
            "compact-call",
            "read",
            json!({"path": "before-summary"}),
        )]))))
        .unwrap();
    batch_rx.recv().await.unwrap();
    tool_gate.add_permits(1);

    let pre_compaction = compaction_rx.recv().await.unwrap();
    assert_eq!(visible_texts(&pre_compaction), ["go"]);
    assert_eq!(tool_call_ids(&pre_compaction), ["compact-call"]);
    assert!(pre_compaction
        .iter()
        .any(|item| matches!(item, TranscriptItem::ToolResults(_))));
    assert!(!visible_texts(&pre_compaction).contains(&"base".to_owned()));
    assert!(!visible_texts(&pre_compaction).contains(&"base-constraint".to_owned()));

    post(
        &session,
        MailboxInput::next_model_request(vec![input("developer", "during-compaction")]),
    )
    .await
    .unwrap();
    compaction_gate.add_permits(1);

    let second = request_rx.recv().await.unwrap();
    assert_eq!(
        visible_texts(&second.transcript),
        ["base", "base-constraint", "summary", "during-compaction"]
    );
    assert!(tool_call_ids(&second.transcript).is_empty());
    response_tx
        .send(Ok(model_response(ModelOutput::text("done"))))
        .unwrap();

    let final_compaction = compaction_rx.recv().await.unwrap();
    assert_eq!(
        visible_texts(&final_compaction),
        ["summary", "during-compaction", "done"]
    );
    assert!(!visible_texts(&final_compaction).contains(&"base".to_owned()));
    assert!(!visible_texts(&final_compaction).contains(&"base-constraint".to_owned()));
    compaction_gate.add_permits(1);

    assert!(matches!(
        task.await.unwrap().unwrap(),
        TurnOutcome::Completed { output, .. } if output.text_content() == "done"
    ));
    assert_eq!(visible_texts(&session.conversation().await), ["summary"]);
}

#[tokio::test]
async fn completion_follow_up_compacts_before_the_reopened_model_request() {
    let (request_tx, mut request_rx) = mpsc::unbounded_channel();
    let (response_tx, response_rx) = mpsc::unbounded_channel();
    let (batch_tx, _batch_rx) = mpsc::unbounded_channel();
    let (compaction_tx, mut compaction_rx) = mpsc::unbounded_channel();
    let compaction_gate = Arc::new(Semaphore::new(0));
    let session = Arc::new(AgentSession::new(
        Arc::new(StaticPrompt),
        Arc::new(GatedBatchRuntime {
            batches: batch_tx,
            gate: Arc::new(Semaphore::new(0)),
            dispatches: AtomicUsize::new(0),
        }),
        Arc::new(GatedSummaryCompaction {
            started: compaction_tx,
            gate: compaction_gate.clone(),
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
    ));

    let running = session.clone();
    let task = tokio::spawn(async move { run_once(&running, vec![input("operator", "go")]).await });
    request_rx.recv().await.unwrap();
    post(
        &session,
        MailboxInput::when_idle(vec![input("operator", "follow-up")]),
    )
    .await
    .unwrap();
    response_tx
        .send(Ok(model_response(ModelOutput::text("candidate"))))
        .unwrap();

    let pre_compaction = compaction_rx.recv().await.unwrap();
    assert_eq!(visible_texts(&pre_compaction), ["go", "candidate"]);
    assert!(!visible_texts(&pre_compaction).contains(&"base".to_owned()));
    assert!(!visible_texts(&pre_compaction).contains(&"base-constraint".to_owned()));
    post(
        &session,
        MailboxInput::next_model_request(vec![input("developer", "during-compaction")]),
    )
    .await
    .unwrap();
    post(
        &session,
        MailboxInput::when_idle(vec![input("operator", "late-idle")]),
    )
    .await
    .unwrap();
    compaction_gate.add_permits(1);

    let second = request_rx.recv().await.unwrap();
    assert_eq!(
        visible_texts(&second.transcript),
        [
            "base",
            "base-constraint",
            "summary",
            "during-compaction",
            "follow-up",
            "late-idle"
        ]
    );
    response_tx
        .send(Ok(model_response(ModelOutput::text("done"))))
        .unwrap();

    let final_compaction = compaction_rx.recv().await.unwrap();
    assert_eq!(
        visible_texts(&final_compaction),
        [
            "summary",
            "during-compaction",
            "follow-up",
            "late-idle",
            "done"
        ]
    );
    compaction_gate.add_permits(1);

    assert!(matches!(
        task.await.unwrap().unwrap(),
        TurnOutcome::Completed { output, .. } if output.text_content() == "done"
    ));
}

#[tokio::test]
async fn stop_during_boundary_compaction_keeps_the_undrained_mailbox() {
    let (request_tx, mut request_rx) = mpsc::unbounded_channel();
    let (response_tx, response_rx) = mpsc::unbounded_channel();
    let (batch_tx, mut batch_rx) = mpsc::unbounded_channel();
    let tool_gate = Arc::new(Semaphore::new(0));
    let (compaction_tx, mut compaction_rx) = mpsc::unbounded_channel();
    let compaction_gate = Arc::new(Semaphore::new(0));
    let session = Arc::new(AgentSession::new(
        Arc::new(StaticPrompt),
        Arc::new(GatedBatchRuntime {
            batches: batch_tx,
            gate: tool_gate.clone(),
            dispatches: AtomicUsize::new(0),
        }),
        Arc::new(GatedSummaryCompaction {
            started: compaction_tx,
            gate: compaction_gate.clone(),
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
    ));

    let running = session.clone();
    let task = tokio::spawn(async move { run_once(&running, vec![input("operator", "go")]).await });
    request_rx.recv().await.unwrap();
    response_tx
        .send(Ok(model_response(ModelOutput::calls(vec![ToolCall::new(
            "compact-before-stop",
            "read",
            json!({}),
        )]))))
        .unwrap();
    batch_rx.recv().await.unwrap();
    tool_gate.add_permits(1);
    compaction_rx.recv().await.unwrap();

    let pending = MailboxInput::next_model_request(vec![input("developer", "undelivered")]);
    post(&session, pending.clone()).await.unwrap();
    let release_gate = compaction_gate.clone();
    tokio::spawn(async move {
        tokio::task::yield_now().await;
        release_gate.add_permits(1);
    });
    session.set_auto_start(false).await;
    session.cancel_run().await;
    session.wait_until_idle().await;

    let TurnOutcome::Stopped { .. } = task.await.unwrap().unwrap() else {
        panic!("取消应产生 Stopped 终态");
    };
    // 输入属于会话而不属于某一次运行：收尾时它们留在信箱里，不交还给调用方。
    assert_eq!(session.mailbox_snapshot().await, vec![pending.clone()]);
    assert!(request_rx.try_recv().is_err());
    assert!(compaction_rx.try_recv().is_err());
}

#[tokio::test]
async fn invalid_compaction_output_falls_back_to_the_closed_conversation() {
    let (request_tx, mut request_rx) = mpsc::unbounded_channel();
    let (response_tx, response_rx) = mpsc::unbounded_channel();
    let (batch_tx, _batch_rx) = mpsc::unbounded_channel();
    let compaction = Arc::new(DanglingCompaction::default());
    let session = Arc::new(AgentSession::new(
        Arc::new(StaticPrompt),
        Arc::new(GatedBatchRuntime {
            batches: batch_tx,
            gate: Arc::new(Semaphore::new(0)),
            dispatches: AtomicUsize::new(0),
        }),
        compaction.clone(),
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
    ));

    let running = session.clone();
    let task = tokio::spawn(async move {
        run_once(&running, vec![input("custom-role", "keep original")]).await
    });
    request_rx.recv().await.unwrap();
    response_tx
        .send(Ok(model_response(ModelOutput::text("closed answer"))))
        .unwrap();
    assert!(matches!(
        task.await.unwrap().unwrap(),
        TurnOutcome::Completed { .. }
    ));
    assert_eq!(compaction.compactions.load(Ordering::SeqCst), 1);

    let conversation = session.conversation().await;
    assert_eq!(
        conversation,
        vec![
            input("custom-role", "keep original"),
            TranscriptItem::ModelOutput(ModelOutput::text("closed answer")),
        ]
    );
    assert!(tool_call_ids(&conversation).is_empty());
}

#[tokio::test]
async fn late_input_during_finalization_is_kept_for_the_next_run() {
    let (request_tx, mut request_rx) = mpsc::unbounded_channel();
    let (response_tx, response_rx) = mpsc::unbounded_channel();
    let (batch_tx, _batch_rx) = mpsc::unbounded_channel();
    let tool_gate = Arc::new(Semaphore::new(0));
    let (compaction_tx, mut compaction_rx) = mpsc::unbounded_channel();
    let compaction_gate = Arc::new(Semaphore::new(0));
    let session = Arc::new(AgentSession::new(
        Arc::new(StaticPrompt),
        Arc::new(GatedBatchRuntime {
            batches: batch_tx,
            gate: tool_gate,
            dispatches: AtomicUsize::new(0),
        }),
        Arc::new(GatedCompaction {
            started: compaction_tx,
            gate: compaction_gate.clone(),
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
    ));

    let running = session.clone();
    let task =
        tokio::spawn(async move { run_once(&running, vec![input("operator", "finish")]).await });
    request_rx.recv().await.unwrap();
    response_tx
        .send(Ok(model_response(ModelOutput::text("done"))))
        .unwrap();
    compaction_rx.recv().await.unwrap();

    // 收尾期间到达的输入照收：它不属于这一轮，但也不会被丢掉。
    let late = MailboxInput::next_model_request(vec![input("operator", "too late")]);
    post(&session, late.clone()).await.unwrap();

    compaction_gate.add_permits(1);
    assert!(matches!(
        task.await.unwrap().unwrap(),
        TurnOutcome::Completed { .. }
    ));
}
