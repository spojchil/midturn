//! 增量交付与提前结算：交付、封口、上报、commit 严格终止。

use super::super::*;

#[tokio::test]
async fn incremental_runtime_receives_ready_calls_and_seal_before_outer_response_finishes() {
    let observer = Arc::new(RecordingStreamObserver::default());
    let mut fixture = streaming_fixture(
        0,
        false,
        SessionConfig::default(),
        Arc::new(NoCompaction),
        observer.clone(),
    );
    let session = fixture.session.clone();
    let task =
        tokio::spawn(async move { run_once(&session, vec![input("operator", "stream")]).await });
    fixture.requests.recv().await.unwrap();

    let call_a = ToolCall::new("stream-a", "read", json!({"path": "a"}));
    let call_b = ToolCall::new("stream-b", "read", json!({"path": "b"}));
    fixture
        .commands
        .send(StreamCommand::Event(ModelStreamEvent::TextDelta {
            part_index: 0,
            delta: "计划".to_owned(),
        }))
        .unwrap();
    fixture
        .commands
        .send(StreamCommand::Event(ModelStreamEvent::ToolCallReady {
            slot: ToolCallSlot::new(0),
            call: call_a.clone(),
        }))
        .unwrap();
    assert!(matches!(
        fixture.updates.recv().await.unwrap(),
        IncrementalUpdate::Began(id) if id.ends_with("/run/1/model/1/tools")
    ));
    assert_eq!(
        fixture.updates.recv().await.unwrap(),
        IncrementalUpdate::Submitted(ToolCallSlot::new(0), "stream-a".to_owned())
    );

    fixture
        .commands
        .send(StreamCommand::Event(ModelStreamEvent::ToolCallReady {
            slot: ToolCallSlot::new(1),
            call: call_b.clone(),
        }))
        .unwrap();
    assert_eq!(
        fixture.updates.recv().await.unwrap(),
        IncrementalUpdate::Submitted(ToolCallSlot::new(1), "stream-b".to_owned())
    );
    fixture
        .commands
        .send(StreamCommand::Event(ModelStreamEvent::ToolCallsSealed {
            call_count: 2,
        }))
        .unwrap();
    assert_eq!(
        fixture.updates.recv().await.unwrap(),
        IncrementalUpdate::CallsSealed(2)
    );
    assert!(fixture.requests.try_recv().is_err());

    fixture
        .commands
        .send(StreamCommand::Complete(model_response(ModelOutput {
            content: vec![ContentPart::text("计划")],
            tool_calls: vec![call_a, call_b],
            provider_data: Default::default(),
        })))
        .unwrap();
    assert_eq!(
        fixture.updates.recv().await.unwrap(),
        IncrementalUpdate::Committed
    );

    let continuation = fixture.requests.recv().await.unwrap();
    assert_eq!(
        tool_call_ids(&continuation.transcript),
        ["stream-a", "stream-b"]
    );
    let results = continuation
        .transcript
        .iter()
        .find_map(|item| match item {
            TranscriptItem::ToolResults(results) => Some(results),
            _ => None,
        })
        .unwrap();
    assert_eq!(results.results[0].call_id.as_str(), "stream-a");
    assert_eq!(results.results[1].call_id.as_str(), "stream-b");
    assert_eq!(fixture.runtime.legacy_dispatches.load(Ordering::SeqCst), 0);

    fixture
        .commands
        .send(StreamCommand::Complete(model_response(ModelOutput::text(
            "done",
        ))))
        .unwrap();
    assert!(matches!(
        task.await.unwrap().unwrap(),
        TurnOutcome::Completed { output, .. } if output.text_content() == "done"
    ));

    let observations = observer.events.lock().unwrap();
    assert!(observations.iter().any(|event| matches!(
        event.payload,
        ModelStreamObservation::Delta(ModelStreamEvent::TextDelta { .. })
    )));
    assert!(observations
        .iter()
        .any(|event| matches!(event.payload, ModelStreamObservation::AttemptCommitted)));
}

/// 提前上报把「本来可知的结果」从未知态里救出来。
///
/// 场景：批内 slot 0 早早跑完并主动上报，slot 1 还在跑时断流。运行时的 abort 报告
/// 用 `settled_slots = 0` 配置成**全部报「确定未开始」**——这模拟一个只按自己视角
/// 回答的实现。内核已经把 slot 0 的结算落了盘，那是一条不可撤回的终态事实，
/// 因此回执里 slot 0 必须仍然是 `Settled`。
///
/// 这不是在提防运行时说谎，而是与「`OutcomeUnknown` 不得被降级」同一条原则：
/// 已经记下的完成事实无法被抹掉。
#[tokio::test]
async fn a_reported_settlement_survives_an_abort_report_that_omits_it() {
    let (request_tx, request_rx) = mpsc::unbounded_channel();
    let (command_tx, command_rx) = mpsc::unbounded_channel();
    let (update_tx, mut update_rx) = mpsc::unbounded_channel();
    let runtime = Arc::new(EagerIncrementalRuntime {
        report_slot_on_submit: Some(0),
        ..EagerIncrementalRuntime::new(update_tx, /*settled_slots*/ 0)
    });
    let session = Arc::new(AgentSession::new(
        Arc::new(StaticPrompt),
        runtime.clone(),
        Arc::new(NoCompaction),
        Arc::new(CommandStreamModel {
            requests: request_tx,
            commands: Mutex::new(command_rx),
        }),
        SessionConfig {
            interrupted_tool_receipt_role: "recovery_fact".to_owned(),
            ..SessionConfig::default()
        },
    ));
    let mut requests = request_rx;

    let driver = session.clone();
    let task =
        tokio::spawn(async move { run_once(&driver, vec![input("operator", "早报")]).await });
    requests.recv().await.unwrap();

    command_tx
        .send(StreamCommand::Event(ModelStreamEvent::ToolCallReady {
            slot: ToolCallSlot::new(0),
            call: ToolCall::new("fast", "read", json!({"path": "a"})),
        }))
        .unwrap();
    command_tx
        .send(StreamCommand::Event(ModelStreamEvent::ToolCallReady {
            slot: ToolCallSlot::new(1),
            call: ToolCall::new("slow", "read", json!({"path": "b"})),
        }))
        .unwrap();
    // Began + 两次 Submitted
    update_rx.recv().await.unwrap();
    update_rx.recv().await.unwrap();
    update_rx.recv().await.unwrap();

    command_tx
        .send(StreamCommand::Fail(AgentError::new(
            AgentErrorKind::Model,
            "stream disconnected",
        )))
        .unwrap();
    assert_eq!(
        update_rx.recv().await.unwrap(),
        IncrementalUpdate::Aborted(ToolBatchAbortReason::ModelStreamInterrupted)
    );

    // 用超时而不是裸 await：合并逻辑一旦没了，回执为空、根本不会有下一次请求，
    // 裸 await 会挂死而不是给出失败。
    let retry = tokio::time::timeout(std::time::Duration::from_secs(2), requests.recv())
        .await
        .expect("已上报的结算应当产生恢复回执并触发下一次模型请求")
        .unwrap();
    let receipt = retry
        .transcript
        .iter()
        .find(|item| is_recovery_receipt(item))
        .expect("已上报的结算必须进入恢复回执");
    let TranscriptItem::Input(receipt) = receipt else {
        unreachable!()
    };
    let body = receipt
        .content
        .iter()
        .find_map(|part| match part {
            ContentPart::Json { value } => Some(value.clone()),
            _ => None,
        })
        .expect("回执是 JSON");
    let calls = body["calls"].as_array().expect("回执含调用列表");
    // slot 1 从未上报，运行时说它确定未开始——那种调用没有副作用，不进回执。
    assert_eq!(calls.len(), 1, "只有已上报的那一项该出现：{body}");
    assert_eq!(calls[0]["call_id"], json!("fast"));
    assert_eq!(
        calls[0]["outcome"]["status"],
        json!("settled"),
        "运行时报的是「确定未开始」，但内核已经落盘了结算，不能被抹掉：{body}"
    );

    task.abort();
}

#[tokio::test]
async fn interrupted_stream_records_settled_fact_before_mailbox() {
    let observer = Arc::new(RecordingStreamObserver::default());
    let mut fixture = streaming_fixture(
        1,
        false,
        SessionConfig {
            interrupted_tool_receipt_role: "recovery_fact".to_owned(),
            ..SessionConfig::default()
        },
        Arc::new(NoCompaction),
        observer.clone(),
    );
    let session = fixture.session.clone();
    let task =
        tokio::spawn(async move { run_once(&session, vec![input("operator", "recover")]).await });
    fixture.requests.recv().await.unwrap();

    fixture
        .commands
        .send(StreamCommand::Event(ModelStreamEvent::TextDelta {
            part_index: 0,
            delta: "未提交正文".to_owned(),
        }))
        .unwrap();
    fixture
        .commands
        .send(StreamCommand::Event(ModelStreamEvent::ToolCallReady {
            slot: ToolCallSlot::new(0),
            call: ToolCall::new("executed", "read", json!({"path": "done"})),
        }))
        .unwrap();
    fixture.updates.recv().await.unwrap();
    fixture.updates.recv().await.unwrap();

    post(
        &fixture.session,
        MailboxInput::next_model_request(vec![input("developer", "mailbox-after-receipt")]),
    )
    .await
    .unwrap();
    fixture
        .commands
        .send(StreamCommand::Fail(AgentError::new(
            AgentErrorKind::Model,
            "stream disconnected",
        )))
        .unwrap();
    assert_eq!(
        fixture.updates.recv().await.unwrap(),
        IncrementalUpdate::Aborted(ToolBatchAbortReason::ModelStreamInterrupted)
    );

    let retry = fixture.requests.recv().await.unwrap();
    assert!(tool_call_ids(&retry.transcript).is_empty());
    let receipt_index = retry
        .transcript
        .iter()
        .position(is_recovery_receipt)
        .expect("恢复请求必须包含执行事实回执");
    let mailbox_index = retry
        .transcript
        .iter()
        .position(|item| {
            matches!(
                item,
                TranscriptItem::Input(message)
                    if message.content == vec![ContentPart::text("mailbox-after-receipt")]
            )
        })
        .unwrap();
    assert!(receipt_index < mailbox_index);

    let TranscriptItem::Input(receipt) = &retry.transcript[receipt_index] else {
        unreachable!()
    };
    assert_eq!(receipt.role, "recovery_fact");
    let ContentPart::Json { value } = &receipt.content[0] else {
        panic!("恢复回执应为结构化 JSON")
    };
    assert_eq!(
        value["kind"],
        DurableFactKind::InterruptedToolBatch.as_wire()
    );
    assert_eq!(value["aborted"], true);
    assert_eq!(value["calls"][0]["slot"], 0);
    assert_eq!(value["calls"][0]["arguments"]["path"], "done");
    assert_eq!(value["calls"][0]["outcome"]["status"], "settled");
    assert!(!value.to_string().contains("DO_NOT_SEND_TO_MODEL"));

    fixture
        .commands
        .send(StreamCommand::Complete(model_response(ModelOutput::text(
            "recovered",
        ))))
        .unwrap();
    assert!(matches!(
        task.await.unwrap().unwrap(),
        TurnOutcome::Completed { output, .. } if output.text_content() == "recovered"
    ));

    let conversation = fixture.session.conversation().await;
    assert!(conversation.iter().any(is_recovery_receipt));
    assert!(conversation.iter().all(|item| {
        !matches!(item, TranscriptItem::ModelOutput(output) if output.tool_calls.iter().any(|call| call.id.as_str() == "executed"))
    }));
    assert!(observer.events.lock().unwrap().iter().any(|event| matches!(
        event.payload,
        ModelStreamObservation::AttemptAborted {
            error_kind: AgentErrorKind::Model
        }
    )));
}

#[tokio::test]
async fn compaction_cannot_drop_an_interrupted_tool_recovery_receipt() {
    let mut fixture = streaming_fixture(
        1,
        false,
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
        Arc::new(DroppingRecoveryCompaction),
        Arc::new(NoopStreamObserver),
    );
    let session = fixture.session.clone();
    let task = tokio::spawn(async move {
        run_once(&session, vec![input("operator", "forget recovery fact")]).await
    });
    fixture.requests.recv().await.unwrap();

    fixture
        .commands
        .send(StreamCommand::Event(ModelStreamEvent::ToolCallReady {
            slot: ToolCallSlot::new(0),
            call: ToolCall::new("executed-and-forgotten", "read", json!({})),
        }))
        .unwrap();
    fixture.updates.recv().await.unwrap();
    fixture.updates.recv().await.unwrap();
    fixture
        .commands
        .send(StreamCommand::Fail(AgentError::new(
            AgentErrorKind::Model,
            "stream disconnected",
        )))
        .unwrap();
    assert_eq!(
        fixture.updates.recv().await.unwrap(),
        IncrementalUpdate::Aborted(ToolBatchAbortReason::ModelStreamInterrupted)
    );

    let retry = fixture.requests.recv().await.unwrap();
    assert!(
        retry.transcript.iter().any(is_recovery_receipt),
        "恢复回执是不可丢弃的持久事实，即使压缩器省略它也必须保留"
    );
    assert!(tool_call_ids(&retry.transcript).is_empty());

    fixture
        .commands
        .send(StreamCommand::Complete(model_response(ModelOutput::text(
            "recovered",
        ))))
        .unwrap();
    assert!(matches!(
        task.await.unwrap().unwrap(),
        TurnOutcome::Completed { output, .. } if output.text_content() == "recovered"
    ));
    assert!(fixture
        .session
        .conversation()
        .await
        .iter()
        .any(is_recovery_receipt));
}

#[tokio::test]
async fn incremental_commit_error_is_strictly_fatal_and_is_not_fabricated_as_recovery() {
    let mut fixture = streaming_fixture(
        1,
        true,
        SessionConfig::default(),
        Arc::new(NoCompaction),
        Arc::new(NoopStreamObserver),
    );
    let session = fixture.session.clone();
    let task =
        tokio::spawn(async move { run_once(&session, vec![input("operator", "commit")]).await });
    fixture.requests.recv().await.unwrap();
    let call = ToolCall::new("commit-fail", "read", json!({}));
    fixture
        .commands
        .send(StreamCommand::Event(ModelStreamEvent::ToolCallReady {
            slot: ToolCallSlot::new(0),
            call: call.clone(),
        }))
        .unwrap();
    fixture.updates.recv().await.unwrap();
    fixture.updates.recv().await.unwrap();
    fixture
        .commands
        .send(StreamCommand::Event(ModelStreamEvent::ToolCallsSealed {
            call_count: 1,
        }))
        .unwrap();
    fixture.updates.recv().await.unwrap();
    fixture
        .commands
        .send(StreamCommand::Complete(model_response(ModelOutput::calls(
            vec![call],
        ))))
        .unwrap();
    assert_eq!(
        fixture.updates.recv().await.unwrap(),
        IncrementalUpdate::Committed
    );

    assert!(matches!(
        task.await.unwrap().unwrap(),
        TurnOutcome::Failed { error, .. }
            if error.kind == AgentErrorKind::ReconciliationRequired
                && error.summary == "tool_batch_outcome_requires_reconciliation:incremental commit failed"
    ));
    let conversation = fixture.session.conversation().await;
    // 严格终止 ≠ 抹掉事实：交出去的那一槽以「结果未知」的回执留在历史里，本轮照样失败、
    // 不自动继续；下一轮开跑前必须对账。
    let receipt = conversation
        .iter()
        .find(|item| is_recovery_receipt(item))
        .expect("在途的槽必须有一个下一轮看得见的出口");
    let TranscriptItem::Input(receipt) = receipt else {
        unreachable!()
    };
    let ContentPart::Json { value } = &receipt.content[0] else {
        panic!("回执应为结构化 JSON")
    };
    assert_eq!(value["calls"][0]["outcome"]["status"], "outcome_unknown");
    assert!(
        tool_call_ids(&conversation).is_empty(),
        "不伪造成正常工具轮"
    );
    assert_eq!(reconcile_all_as_cancelled(&fixture.session).await, 1);
    assert!(fixture.session.pending_reconciliation().await.is_empty());
}
