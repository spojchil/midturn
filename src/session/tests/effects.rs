//! 副作用账本与持久化：对账、部分结算、确认丢失与隔离。

use super::*;

#[tokio::test]
async fn live_in_flight_effect_is_not_reconcilable_by_an_operator() {
    let mut fixture = fixture();
    let running = fixture.session.clone();
    let run =
        tokio::spawn(
            async move { run_once(&running, vec![input("operator", "active effect")]).await },
        );
    fixture.requests.recv().await.unwrap();
    fixture
        .responses
        .send(Ok(model_response(ModelOutput::calls(vec![ToolCall::new(
            "active-call",
            "read",
            json!({}),
        )]))))
        .unwrap();
    fixture.batches.recv().await.unwrap();

    let effect_id = {
        let effects = fixture.session.effect_ledger().await;
        let (id, record) = effects.iter().next().expect("effect must be durable");
        assert!(matches!(record.state, EffectState::InFlight { .. }));
        id.clone()
    };
    assert!(fixture.session.pending_reconciliation().await.is_empty());
    let error = fixture
        .session
        .reconcile_effect(&effect_id, DurableEffectOutcome::CancelledBeforeStart)
        .await
        .unwrap_err();
    assert_eq!(error.kind, AgentErrorKind::InvalidState);
    assert_eq!(error.summary, "effect_is_owned_by_active_driver");

    fixture.gate.add_permits(1);
    fixture.requests.recv().await.unwrap();
    fixture
        .responses
        .send(Ok(model_response(ModelOutput::text("done"))))
        .unwrap();
    assert!(matches!(
        run.await.unwrap().unwrap(),
        TurnOutcome::Completed { output, .. } if output.text_content() == "done"
    ));
    assert!(fixture.session.pending_reconciliation().await.is_empty());
}

#[tokio::test]
async fn invalid_tool_result_batch_never_partially_settles_effects() {
    let (request_tx, mut request_rx) = mpsc::unbounded_channel();
    let (response_tx, response_rx) = mpsc::unbounded_channel();
    let session = Arc::new(AgentSession::new(
        Arc::new(StaticPrompt),
        Arc::new(PartiallyInvalidBatchRuntime),
        Arc::new(NoCompaction),
        Arc::new(ChannelModel {
            requests: request_tx,
            responses: Mutex::new(response_rx),
        }),
        SessionConfig::default(),
    ));
    let running = session.clone();
    let run = tokio::spawn(async move {
        run_once(&running, vec![input("operator", "invalid runtime results")]).await
    });
    request_rx.recv().await.unwrap();
    response_tx
        .send(Ok(model_response(ModelOutput::calls(vec![
            ToolCall::new("valid-result", "read", json!({})),
            ToolCall::new("missing-result", "read", json!({})),
        ]))))
        .unwrap();

    assert!(matches!(
        run.await.unwrap().unwrap(),
        TurnOutcome::Failed { error, .. }
            if error.kind == AgentErrorKind::ReconciliationRequired
                && error.summary == "invalid_tool_result_batch_requires_reconciliation"
    ));
    let pending = session.pending_reconciliation().await;
    assert_eq!(pending.len(), 2);
    let effects = session.effect_ledger().await;
    assert_eq!(effects.len(), 2);
    assert!(effects.iter().all(|(_, record)| matches!(
        &record.state,
        EffectState::Outcome {
            outcome: DurableEffectOutcome::OutcomeUnknown { .. }
        }
    )));
    assert!(tool_call_ids(&session.conversation().await).is_empty());
}

#[tokio::test]
async fn ack_lost_writer_is_fenced_and_reopen_uses_the_committed_candidate() {
    let session_id = SessionId::new("ack-loss/writer-fence").unwrap();
    let store = Arc::new(AckLossCheckpointStore::default());
    let (first_request_tx, _first_request_rx) = mpsc::unbounded_channel();
    let (_first_response_tx, first_response_rx) = mpsc::unbounded_channel();
    let first = Arc::new(
        AgentSession::open(
            session_id.clone(),
            store.clone(),
            Arc::new(StaticPrompt),
            Arc::new(PanickingBatchRuntime),
            Arc::new(NoCompaction),
            Arc::new(ChannelModel {
                requests: first_request_tx,
                responses: Mutex::new(first_response_rx),
            }),
            SessionConfig::default(),
        )
        .await
        .unwrap(),
    );

    let initial = vec![input("operator", "committed start candidate")];
    store.arm();
    let rejected = match start(&first, initial.clone()).await {
        Ok(_) => panic!("确认丢失必须隔离 writer"),
        Err(rejected) => rejected,
    };
    assert_eq!(
        rejected.reason,
        MailboxRejectedReason::PersistenceCommitUnknown
    );
    assert_eq!(rejected.input.items, initial);
    assert_eq!(
        first.persistence_fence_reason().await,
        Some(AgentErrorKind::PersistenceCommitUnknown)
    );
    let writes_after_start_ack_loss = store.committed_writes();
    let rejected = match start(
        &first,
        vec![input("operator", "must not retry on fenced writer")],
    )
    .await
    {
        Ok(_) => panic!("被隔离的 writer 必须拒绝下一次投递"),
        Err(rejected) => rejected,
    };
    assert_eq!(
        rejected.reason,
        MailboxRejectedReason::PersistenceCommitUnknown
    );
    assert_eq!(store.committed_writes(), writes_after_start_ack_loss);

    let (second_request_tx, mut second_request_rx) = mpsc::unbounded_channel();
    let (_second_response_tx, second_response_rx) = mpsc::unbounded_channel();
    let second = Arc::new(
        AgentSession::open(
            session_id.clone(),
            store.clone(),
            Arc::new(StaticPrompt),
            Arc::new(PanickingBatchRuntime),
            Arc::new(NoCompaction),
            Arc::new(ChannelModel {
                requests: second_request_tx,
                responses: Mutex::new(second_response_rx),
            }),
            SessionConfig::default(),
        )
        .await
        .unwrap(),
    );
    assert_eq!(second.run_sequence().await, 1);
    let resumed = second
        .try_resume()
        .await
        .unwrap()
        .expect("the committed start candidate must be resumed, not recreated");
    assert!(resumed.run_id().as_str().ends_with("/run/1"));
    let request = second_request_rx.recv().await.unwrap();
    assert_eq!(
        visible_texts(&request.transcript),
        ["base", "base-constraint", "committed start candidate"]
    );

    let envelope =
        MailboxInput::next_model_request(vec![input("developer", "committed enqueue candidate")]);
    store.arm();
    let rejected = post(&second, envelope.clone()).await.unwrap_err();
    assert_eq!(
        rejected.reason,
        MailboxRejectedReason::PersistenceCommitUnknown
    );
    assert_eq!(rejected.input, envelope);
    assert_eq!(
        second.persistence_fence_reason().await,
        Some(AgentErrorKind::PersistenceCommitUnknown)
    );
    let writes_after_enqueue_ack_loss = store.committed_writes();
    let rejected = post(
        &second,
        MailboxInput::next_model_request(vec![input(
            "developer",
            "must not retry enqueue on fenced writer",
        )]),
    )
    .await
    .unwrap_err();
    assert_eq!(
        rejected.reason,
        MailboxRejectedReason::PersistenceCommitUnknown
    );
    assert_eq!(store.committed_writes(), writes_after_enqueue_ack_loss);

    resumed.cancel();
    let cancelled = tokio::time::timeout(std::time::Duration::from_secs(1), resumed.join())
        .await
        .expect("fenced driver must still release ownership when cancelled");
    assert!(matches!(
        cancelled,
        TurnOutcome::Failed { error, .. }
            if error.kind == AgentErrorKind::PersistenceCommitUnknown
    ));
    assert_eq!(store.committed_writes(), writes_after_enqueue_ack_loss);

    let (third_request_tx, mut third_request_rx) = mpsc::unbounded_channel();
    let (third_response_tx, third_response_rx) = mpsc::unbounded_channel();
    let third = Arc::new(
        AgentSession::open(
            session_id,
            store.clone(),
            Arc::new(StaticPrompt),
            Arc::new(PanickingBatchRuntime),
            Arc::new(NoCompaction),
            Arc::new(ChannelModel {
                requests: third_request_tx,
                responses: Mutex::new(third_response_rx),
            }),
            SessionConfig::default(),
        )
        .await
        .unwrap(),
    );
    assert_eq!(third.mailbox_snapshot().await, vec![envelope]);
    let resumed = third
        .try_resume()
        .await
        .unwrap()
        .expect("the committed enqueue candidate must survive reopening");
    assert!(resumed.run_id().as_str().ends_with("/run/1"));
    let request = third_request_rx.recv().await.unwrap();
    let visible = visible_texts(&request.transcript);
    assert_eq!(
        visible,
        [
            "base",
            "base-constraint",
            "committed start candidate",
            "committed enqueue candidate",
        ]
    );
    assert_eq!(
        visible
            .iter()
            .filter(|text| text.as_str() == "committed start candidate")
            .count(),
        1
    );
    third_response_tx
        .send(Ok(model_response(ModelOutput::text("recovered once"))))
        .unwrap();
    assert!(matches!(
        resumed.join().await,
        TurnOutcome::Completed { output, .. } if output.text_content() == "recovered once"
    ));
}

#[tokio::test]
async fn finalization_ack_loss_does_not_return_an_ambiguously_committed_envelope() {
    let session_id = SessionId::new("ack-loss/finalization").unwrap();
    let store = Arc::new(AckLossCheckpointStore::default());
    let (request_tx, mut request_rx) = mpsc::unbounded_channel();
    let (response_tx, response_rx) = mpsc::unbounded_channel();
    let session = Arc::new(
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
    let run = start(
        &session,
        vec![input("operator", "before ambiguous finalization")],
    )
    .await
    .unwrap();
    request_rx.recv().await.unwrap();
    let envelope =
        MailboxInput::next_model_request(vec![input("developer", "must not have two owners")]);
    post(&session, envelope.clone()).await.unwrap();

    store.arm();
    response_tx
        .send(Err(AgentError::new(
            AgentErrorKind::Model,
            "force finalization",
        )))
        .unwrap();
    let outcome = run.join().await;
    let TurnOutcome::Failed { stage, error, .. } = outcome else {
        panic!("finalization ack loss must fail the run")
    };
    assert_eq!(stage, RunStage::Boundary);
    assert_eq!(error.kind, AgentErrorKind::PersistenceCommitUnknown);
    assert_eq!(
        session.persistence_fence_reason().await,
        Some(AgentErrorKind::PersistenceCommitUnknown)
    );

    let stored = store.load(&session_id).await.unwrap().unwrap();
    assert!(stored.checkpoint.active_run.is_none());
    // 信箱在收尾时不搬家，因此这次 CAS 到底提交没有都不影响它：
    // 新旧两个版本的 checkpoint 持有的是同一份未投递输入。
    assert_eq!(stored.checkpoint.mailbox, vec![envelope.clone()]);
    assert_eq!(
        visible_texts(&stored.checkpoint.conversation),
        ["before ambiguous finalization"]
    );

    // 重新打开：没有半截运行；未投递输入仍在信箱里，于是它自己把新运行叫起来。
    let (reopened_request_tx, mut reopened_request_rx) = mpsc::unbounded_channel();
    let (reopened_response_tx, reopened_response_rx) = mpsc::unbounded_channel();
    let reopened = Arc::new(
        AgentSession::open(
            session_id,
            store,
            Arc::new(StaticPrompt),
            Arc::new(PanickingBatchRuntime),
            Arc::new(NoCompaction),
            Arc::new(ChannelModel {
                requests: reopened_request_tx,
                responses: Mutex::new(reopened_response_rx),
            }),
            SessionConfig::default(),
        )
        .await
        .unwrap(),
    );
    assert!(reopened.try_resume().await.unwrap().is_none());
    let request = reopened_request_rx.recv().await.unwrap();
    assert_eq!(
        visible_texts(&request.transcript),
        [
            "base",
            "base-constraint",
            "before ambiguous finalization",
            "must not have two owners",
        ],
        "重新打开后未投递输入只有一个主人：新运行"
    );
    reopened_response_tx
        .send(Ok(model_response(ModelOutput::text("owned once"))))
        .unwrap();
    reopened.wait_until_idle().await;
    assert!(reopened.mailbox_snapshot().await.is_empty());
}
