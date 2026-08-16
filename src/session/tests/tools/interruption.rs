//! 中断与分类：断流、取消、运行时拒绝、确认丢失、恢复次数上限。

use super::super::*;

#[tokio::test]
async fn cancelled_incremental_calls_do_not_create_a_recovery_receipt() {
    let mut fixture = streaming_fixture(
        0,
        false,
        SessionConfig::default(),
        Arc::new(NoCompaction),
        Arc::new(NoopStreamObserver),
    );
    let session = fixture.session.clone();
    let task =
        tokio::spawn(async move { run_once(&session, vec![input("operator", "cancel")]).await });
    fixture.requests.recv().await.unwrap();
    fixture
        .commands
        .send(StreamCommand::Event(ModelStreamEvent::ToolCallReady {
            slot: ToolCallSlot::new(0),
            call: ToolCall::new("cancelled", "read", json!({})),
        }))
        .unwrap();
    fixture.updates.recv().await.unwrap();
    fixture.updates.recv().await.unwrap();
    fixture
        .commands
        .send(StreamCommand::Fail(AgentError::new(
            AgentErrorKind::Model,
            "interrupted",
        )))
        .unwrap();
    fixture.updates.recv().await.unwrap();

    assert!(matches!(
        task.await.unwrap().unwrap(),
        TurnOutcome::Failed { error, .. } if error.summary == "interrupted"
    ));
    assert!(!fixture
        .session
        .conversation()
        .await
        .iter()
        .any(is_recovery_receipt));
}

#[tokio::test]
async fn legacy_runtime_does_not_dispatch_ready_calls_from_an_interrupted_stream() {
    let (request_tx, mut request_rx) = mpsc::unbounded_channel();
    let (command_tx, command_rx) = mpsc::unbounded_channel();
    let (batch_tx, mut batch_rx) = mpsc::unbounded_channel();
    let runtime = Arc::new(GatedBatchRuntime {
        batches: batch_tx,
        gate: Arc::new(Semaphore::new(0)),
        dispatches: AtomicUsize::new(0),
    });
    let session = Arc::new(AgentSession::new(
        Arc::new(StaticPrompt),
        runtime.clone(),
        Arc::new(NoCompaction),
        Arc::new(CommandStreamModel {
            requests: request_tx,
            commands: Mutex::new(command_rx),
        }),
        SessionConfig::default(),
    ));
    let running = session.clone();
    let task =
        tokio::spawn(async move { run_once(&running, vec![input("operator", "legacy")]).await });
    request_rx.recv().await.unwrap();
    command_tx
        .send(StreamCommand::Event(ModelStreamEvent::ToolCallReady {
            slot: ToolCallSlot::new(0),
            call: ToolCall::new("not-dispatched", "read", json!({})),
        }))
        .unwrap();
    command_tx
        .send(StreamCommand::Fail(AgentError::new(
            AgentErrorKind::Model,
            "legacy stream interrupted",
        )))
        .unwrap();

    assert!(matches!(
        task.await.unwrap().unwrap(),
        TurnOutcome::Failed { error, .. } if error.summary == "legacy stream interrupted"
    ));
    assert_eq!(runtime.dispatches.load(Ordering::SeqCst), 0);
    assert!(batch_rx.try_recv().is_err());
    assert!(tool_call_ids(&session.conversation().await).is_empty());
}

#[tokio::test]
async fn invalid_final_tool_array_is_validated_before_any_legacy_dispatch() {
    let mut fixture = fixture();
    let session = fixture.session.clone();
    let task = tokio::spawn(async move {
        run_once(&session, vec![input("operator", "invalid final tools")]).await
    });
    fixture.requests.recv().await.unwrap();
    fixture
        .responses
        .send(Ok(model_response(ModelOutput::calls(vec![
            ToolCall::new("duplicate", "read", json!({"slot": 0})),
            ToolCall::new("duplicate", "read", json!({"slot": 1})),
        ]))))
        .unwrap();

    assert!(matches!(
        task.await.unwrap().unwrap(),
        TurnOutcome::Failed { error, .. }
            if error.kind == AgentErrorKind::InvalidToolBatch
                && error.summary == "duplicate_tool_call_id"
    ));
    assert_eq!(fixture.runtime.dispatches.load(Ordering::SeqCst), 0);
    assert!(fixture.batches.try_recv().is_err());
    assert!(tool_call_ids(&fixture.session.conversation().await).is_empty());
}

#[tokio::test]
async fn interrupted_tool_recovery_limit_stops_repeated_side_effect_loops_but_keeps_facts() {
    let mut fixture = streaming_fixture(
        10,
        false,
        SessionConfig {
            max_interrupted_tool_recoveries: 1,
            ..SessionConfig::default()
        },
        Arc::new(NoCompaction),
        Arc::new(NoopStreamObserver),
    );
    let session = fixture.session.clone();
    let task =
        tokio::spawn(async move { run_once(&session, vec![input("operator", "loop")]).await });

    for attempt in 0..2 {
        fixture.requests.recv().await.unwrap();
        fixture
            .commands
            .send(StreamCommand::Event(ModelStreamEvent::ToolCallReady {
                slot: ToolCallSlot::new(0),
                call: ToolCall::new(format!("call-{attempt}"), "read", json!({})),
            }))
            .unwrap();
        fixture.updates.recv().await.unwrap();
        fixture.updates.recv().await.unwrap();
        fixture
            .commands
            .send(StreamCommand::Fail(AgentError::new(
                AgentErrorKind::Model,
                "repeat disconnect",
            )))
            .unwrap();
        fixture.updates.recv().await.unwrap();
    }

    assert!(matches!(
        task.await.unwrap().unwrap(),
        TurnOutcome::Failed { error, .. }
            if error.summary == "interrupted_tool_recovery_limit_exceeded"
    ));
    assert_eq!(
        fixture
            .session
            .conversation()
            .await
            .iter()
            .filter(|item| is_recovery_receipt(item))
            .count(),
        2
    );
}

struct AmbiguousSubmitRuntime;

impl ToolRuntime for AmbiguousSubmitRuntime {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition::new(
            "write",
            json!({"type": "object", "additionalProperties": true}),
        )]
    }

    fn dispatch<'a>(
        &'a self,
        _batch: ToolCallBatch,
    ) -> PortFuture<'a, Result<ToolResultBatch, AgentError>> {
        Box::pin(async { panic!("确认不确定测试不应走兼容 dispatch") })
    }

    fn begin_incremental<'a>(
        &'a self,
        _batch: ToolBatchStart,
        _reporter: Arc<dyn ToolCallReporter>,
    ) -> PortFuture<'a, Result<Option<Box<dyn IncrementalToolBatch + 'a>>, AgentError>> {
        Box::pin(async move {
            Ok(Some(Box::new(AmbiguousSubmitBatch { offered: None })
                as Box<dyn IncrementalToolBatch + 'a>))
        })
    }
}

struct AmbiguousSubmitBatch {
    offered: Option<IncrementalToolCall>,
}

struct SwallowingSinkErrorModel;

impl Model for SwallowingSinkErrorModel {
    fn complete<'a>(
        &'a self,
        _request: ModelRequest,
    ) -> PortFuture<'a, Result<ModelResponse, AgentError>> {
        Box::pin(async { panic!("吞错模型测试不应走 one-shot complete") })
    }

    fn complete_stream<'a>(
        &'a self,
        _request: ModelRequest,
        sink: &'a mut dyn ModelStreamSink,
    ) -> PortFuture<'a, Result<ModelResponse, AgentError>> {
        Box::pin(async move {
            let call = ToolCall::new("possibly-written", "write", json!({"value": 1}));
            // 故意违反端口契约：忽略 sink 的失败并谎报完整模型响应。
            let _ = sink
                .emit(ModelStreamEvent::ToolCallReady {
                    slot: ToolCallSlot::new(0),
                    call: call.clone(),
                })
                .await;
            Ok(model_response(ModelOutput::calls(vec![call])))
        })
    }
}

impl IncrementalToolBatch for AmbiguousSubmitBatch {
    fn submit<'a>(
        &'a mut self,
        call: IncrementalToolCall,
    ) -> PortFuture<'a, Result<(), AgentError>> {
        Box::pin(async move {
            self.offered = Some(call);
            // 端口实现可以返回任意错误种类；核心必须按调用来源识别这是 runtime
            // 拒绝，不能因为它写成 Model 就误走模型流自动恢复。
            Err(AgentError::new(AgentErrorKind::Model, "submit_ack_lost"))
        })
    }

    fn calls_sealed<'a>(&'a mut self, _call_count: u32) -> PortFuture<'a, Result<(), AgentError>> {
        Box::pin(async { panic!("submit 已失败，不应再封口") })
    }

    fn commit<'a>(self: Box<Self>) -> PortFuture<'a, Result<(), AgentError>>
    where
        Self: 'a,
    {
        Box::pin(async { panic!("submit 已失败，不应提交") })
    }

    fn abort<'a>(
        self: Box<Self>,
        _reason: ToolBatchAbortReason,
    ) -> PortFuture<'a, Result<AbortClassification, AgentError>>
    where
        Self: 'a,
    {
        Box::pin(async move {
            let offered = self.offered.expect("submit 应先收到调用");
            Ok(AbortClassification::new()
                .outcome_unknown(offered.slot, "远端可能已经执行，但确认丢失"))
        })
    }
}

#[tokio::test]
async fn submit_ack_loss_is_preserved_as_an_unknown_outcome() {
    let (request_tx, mut request_rx) = mpsc::unbounded_channel();
    let (command_tx, command_rx) = mpsc::unbounded_channel();
    let session = Arc::new(AgentSession::new(
        Arc::new(StaticPrompt),
        Arc::new(AmbiguousSubmitRuntime),
        Arc::new(NoCompaction),
        Arc::new(CommandStreamModel {
            requests: request_tx,
            commands: Mutex::new(command_rx),
        }),
        SessionConfig::default(),
    ));
    let running = session.clone();
    let task = tokio::spawn(async move {
        run_once(&running, vec![input("operator", "ambiguous submit")]).await
    });
    request_rx.recv().await.unwrap();
    command_tx
        .send(StreamCommand::Event(ModelStreamEvent::ToolCallReady {
            slot: ToolCallSlot::new(0),
            call: ToolCall::new("possibly-written", "write", json!({"value": 1})),
        }))
        .unwrap();

    assert!(matches!(
        tokio::time::timeout(std::time::Duration::from_secs(1), task)
            .await
            .expect("runtime 拒绝必须严格终止，不能等待下一次模型响应")
            .unwrap()
            .unwrap(),
        TurnOutcome::Failed { error, .. }
            if error.kind == AgentErrorKind::ReconciliationRequired
                && error.summary == "interrupted_tool_outcome_requires_reconciliation"
    ));
    let conversation = session.conversation().await;
    assert!(tool_call_ids(&conversation).is_empty());
    let receipt = conversation
        .iter()
        .find(|item| is_recovery_receipt(item))
        .expect("确认不确定必须留下恢复事实");
    let TranscriptItem::Input(receipt) = receipt else {
        unreachable!()
    };
    let ContentPart::Json { value } = &receipt.content[0] else {
        panic!("恢复事实应为结构化 JSON")
    };
    assert_eq!(value["calls"][0]["call_id"], "possibly-written");
    assert_eq!(value["calls"][0]["outcome"]["status"], "outcome_unknown");
    assert_eq!(reconcile_all_as_cancelled(&session).await, 1);
    assert!(session.pending_reconciliation().await.is_empty());
}

#[tokio::test]
async fn adapter_cannot_swallow_a_sink_failure_and_commit_the_attempt() {
    let session = Arc::new(AgentSession::new(
        Arc::new(StaticPrompt),
        Arc::new(AmbiguousSubmitRuntime),
        Arc::new(NoCompaction),
        Arc::new(SwallowingSinkErrorModel),
        SessionConfig::default(),
    ));

    assert!(matches!(
        run_once(&session, vec![input("operator", "swallow sink error")]).await
            .unwrap(),
        TurnOutcome::Failed { error, .. }
            if error.kind == AgentErrorKind::ReconciliationRequired
                && error.summary == "interrupted_tool_outcome_requires_reconciliation"
    ));
    let conversation = session.conversation().await;
    assert!(conversation.iter().any(is_recovery_receipt));
    assert!(tool_call_ids(&conversation).is_empty());
    assert_eq!(reconcile_all_as_cancelled(&session).await, 1);
}
