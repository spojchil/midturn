//! 监督：等待方取消、运行时 / 工具目录 / 压缩 panic。

use super::*;

#[tokio::test]
async fn cancelling_the_waiter_does_not_cancel_the_owned_driver() {
    let mut fixture = fixture();
    let running = fixture.session.clone();
    let waiter =
        tokio::spawn(async move { run_once(&running, vec![input("operator", "first")]).await });
    fixture.requests.recv().await.unwrap();

    // 丢弃调用方等待任务；session 内部的 driver 仍应完成并解除 Busy。
    waiter.abort();
    let _ = waiter.await;
    fixture
        .responses
        .send(Ok(model_response(ModelOutput::text("first done"))))
        .unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        loop {
            let completed = fixture.session.conversation().await.iter().any(|item| {
                matches!(item, TranscriptItem::ModelOutput(output) if output.text_content() == "first done")
            });
            if completed {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();

    let running = fixture.session.clone();
    let second =
        tokio::spawn(async move { run_once(&running, vec![input("operator", "second")]).await });
    fixture.requests.recv().await.unwrap();
    fixture
        .responses
        .send(Ok(model_response(ModelOutput::text("second done"))))
        .unwrap();
    assert!(matches!(
        second.await.unwrap().unwrap(),
        TurnOutcome::Completed { .. }
    ));
}

#[tokio::test]
async fn cancelled_waiter_does_not_orphan_a_later_runtime_panic() {
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
    let waiter =
        tokio::spawn(async move { run_once(&running, vec![input("operator", "first")]).await });
    request_rx.recv().await.unwrap();

    // 这个信封到 panic 时尚未投递。即使原等待方已经取消，也不能在收尾时丢失。
    post(
        &session,
        MailboxInput::next_model_request(vec![input("operator", "preserved after panic")]),
    )
    .await
    .unwrap();
    waiter.abort();
    let _ = waiter.await;

    response_tx
        .send(Ok(model_response(ModelOutput::calls(vec![ToolCall::new(
            "panic-after-cancel",
            "read",
            json!({}),
        )]))))
        .unwrap();

    // 丢掉句柄不会遗弃运行：它照样跑到失败，照样释放。
    tokio::time::timeout(std::time::Duration::from_secs(1), session.wait_until_idle())
        .await
        .expect("supervisor must release the run after the runtime panics");
    assert!(!session.has_active_run().await);

    assert!(matches!(
        session
            .enqueue(MailboxInput::next_model_request(vec![input(
                "operator",
                "cannot bypass panic recovery"
            )]))
            .await
            .unwrap(),
        Enqueued::Held(HoldReason::ReconciliationRequired)
    ));
    assert_eq!(reconcile_all_as_cancelled(&session).await, 1);

    // 对账期间被挡住的两条输入照样收下了；对账完成的那一刻自动开跑，第一个边界把它们排空。
    let resumed_request =
        tokio::time::timeout(std::time::Duration::from_secs(1), request_rx.recv())
            .await
            .expect("对账完成后应发起模型请求")
            .unwrap();
    assert_eq!(
        visible_texts(&resumed_request.transcript),
        vec![
            "base",
            "base-constraint",
            "first",
            "preserved after panic",
            "cannot bypass panic recovery"
        ]
    );
    assert!(resumed_request
        .transcript
        .iter()
        .any(|item| has_json_fact_kind(item, "effect_reconciliation_receipt")));
    response_tx
        .send(Ok(model_response(ModelOutput::text("recovered"))))
        .unwrap();
    session.wait_until_idle().await;

    let running = session.clone();
    let next_run =
        tokio::spawn(async move { run_once(&running, vec![input("operator", "second")]).await });
    let next_request = request_rx.recv().await.unwrap();
    assert_eq!(
        visible_texts(&next_request.transcript),
        vec![
            "base",
            "base-constraint",
            "first",
            "preserved after panic",
            "cannot bypass panic recovery",
            "recovered",
            "second"
        ]
    );
    response_tx
        .send(Ok(model_response(ModelOutput::text("second done"))))
        .unwrap();
    assert!(matches!(
        next_run.await.unwrap().unwrap(),
        TurnOutcome::Completed { output, .. } if output.text_content() == "second done"
    ));

    session.set_auto_start(false).await;
    tokio::time::timeout(std::time::Duration::from_secs(1), session.wait_until_idle())
        .await
        .expect("恢复过的会话仍应能安静下来");
}

#[tokio::test]
async fn definitions_panic_fails_the_run_and_keeps_the_drained_input() {
    let (request_tx, mut request_rx) = mpsc::unbounded_channel();
    let (response_tx, response_rx) = mpsc::unbounded_channel();
    let runtime = Arc::new(PanicOnceDefinitionsRuntime::default());
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

    // 工具目录都拿不到就不去问模型：那会得到一个静默错误的回答。
    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        run_once(
            &session,
            vec![input("operator", "durable before definitions")],
        ),
    )
    .await
    .expect("definitions panic must be contained and finish the run")
    .unwrap();
    let TurnOutcome::Failed { error, stage, .. } = outcome else {
        panic!("definitions panic 应让运行失败")
    };
    assert_eq!(stage, RunStage::Model);
    assert_eq!(error.kind, AgentErrorKind::InvalidState);
    assert_eq!(error.summary, "tool_definitions_panicked");
    assert!(request_rx.try_recv().is_err());
    assert!(session.pending_reconciliation().await.is_empty());
    assert!(session.try_resume().await.unwrap().is_none());
    // 已经排空进对话的输入不丢：它属于会话，不属于失败的那一次运行。
    assert_eq!(
        visible_texts(&session.conversation().await),
        ["durable before definitions"]
    );

    let running = session.clone();
    let next = tokio::spawn(async move {
        run_once(&running, vec![input("operator", "after definitions")]).await
    });
    let request = tokio::time::timeout(std::time::Duration::from_secs(1), request_rx.recv())
        .await
        .expect("the next run must reach the model once definitions succeeds")
        .unwrap();
    assert_eq!(
        visible_texts(&request.transcript),
        [
            "base",
            "base-constraint",
            "durable before definitions",
            "after definitions"
        ]
    );
    assert_eq!(request.function_tools.len(), 1);
    response_tx
        .send(Ok(model_response(ModelOutput::text("resumed"))))
        .unwrap();
    assert!(matches!(
        next.await.unwrap().unwrap(),
        TurnOutcome::Completed { output, .. } if output.text_content() == "resumed"
    ));
    assert!(runtime.panicked.load(Ordering::SeqCst));
}

#[tokio::test]
async fn final_compaction_panic_fails_the_run_and_keeps_the_queued_envelope() {
    let (request_tx, mut request_rx) = mpsc::unbounded_channel();
    let (response_tx, response_rx) = mpsc::unbounded_channel();
    let compaction = Arc::new(PanicOnceCompaction::default());
    let session = Arc::new(AgentSession::new(
        Arc::new(StaticPrompt),
        Arc::new(PanicOnceDefinitionsRuntime {
            // 此场景只验证压缩，因此 definitions 不能 panic。
            panicked: AtomicBool::new(true),
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
    let panicking_run = tokio::spawn(async move {
        run_once(&running, vec![input("operator", "before final compaction")]).await
    });
    request_rx.recv().await.unwrap();
    // 被动投递不会在完成边界把运行重开，于是它跨过收尾压缩留在信箱里。
    let envelope = MailboxInput::passive(vec![input(
        "developer",
        "queued across final compaction panic",
    )]);
    post(&session, envelope.clone()).await.unwrap();
    response_tx
        .send(Ok(model_response(ModelOutput::text("done"))))
        .unwrap();

    let outcome = tokio::time::timeout(std::time::Duration::from_secs(1), panicking_run)
        .await
        .expect("final compaction panic must be supervised")
        .unwrap()
        .unwrap();
    let TurnOutcome::Failed { error, stage, .. } = outcome else {
        panic!("收尾压缩 panic 应让运行失败，而不是丢下半截运行")
    };
    assert_eq!(stage, RunStage::Compaction);
    assert_eq!(error.summary, "compaction_panicked");
    assert!(compaction.panicked.load(Ordering::SeqCst));
    // 对话沿用原记录；信封还在信箱里。
    assert_eq!(
        visible_texts(&session.conversation().await),
        ["before final compaction", "done"]
    );
    assert_eq!(session.mailbox_snapshot().await, vec![envelope]);
    assert!(session.try_resume().await.unwrap().is_none());

    let running = session.clone();
    let next =
        tokio::spawn(
            async move { run_once(&running, vec![input("operator", "after panic")]).await },
        );
    let request = tokio::time::timeout(std::time::Duration::from_secs(1), request_rx.recv())
        .await
        .expect("the next run must deliver the preserved envelope")
        .unwrap();
    assert_eq!(
        visible_texts(&request.transcript),
        [
            "base",
            "base-constraint",
            "before final compaction",
            "done",
            "after panic",
            "queued across final compaction panic",
        ]
    );
    response_tx
        .send(Ok(model_response(ModelOutput::text("recovered"))))
        .unwrap();
    assert!(matches!(
        next.await.unwrap().unwrap(),
        TurnOutcome::Completed { output, .. } if output.text_content() == "recovered"
    ));
}

// ── 增量运行时在五个端口方法里各 panic 一次 ─────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
enum PanicAt {
    Begin,
    Submit,
    Seal,
    Commit,
    Abort,
}

struct PanickingIncrementalRuntime {
    at: PanicAt,
    reached: mpsc::UnboundedSender<PanicAt>,
}

impl ToolRuntime for PanickingIncrementalRuntime {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition::new(
            "read",
            json!({"type": "object", "additionalProperties": true}),
        )]
    }

    fn dispatch<'a>(
        &'a self,
        _batch: ToolCallBatch,
    ) -> PortFuture<'a, Result<ToolResultBatch, AgentError>> {
        Box::pin(async { panic!("增量场景不该走 dispatch") })
    }

    fn begin_incremental<'a>(
        &'a self,
        _batch: ToolBatchStart,
        _reporter: Arc<dyn ToolCallReporter>,
    ) -> PortFuture<'a, Result<Option<Box<dyn IncrementalToolBatch + 'a>>, AgentError>> {
        Box::pin(async move {
            let _ = self.reached.send(PanicAt::Begin);
            if self.at == PanicAt::Begin {
                panic!("begin_incremental bug")
            }
            Ok(Some(Box::new(PanickingBatch {
                at: self.at,
                reached: self.reached.clone(),
                slots: Vec::new(),
            }) as Box<dyn IncrementalToolBatch + 'a>))
        })
    }
}

struct PanickingBatch {
    at: PanicAt,
    reached: mpsc::UnboundedSender<PanicAt>,
    slots: Vec<ToolCallSlot>,
}

impl IncrementalToolBatch for PanickingBatch {
    fn submit<'a>(
        &'a mut self,
        call: IncrementalToolCall,
    ) -> PortFuture<'a, Result<(), AgentError>> {
        Box::pin(async move {
            let _ = self.reached.send(PanicAt::Submit);
            // 先记下「收到了」再炸：abort 时才能如实说它没开始。
            self.slots.push(call.slot);
            if self.at == PanicAt::Submit {
                panic!("submit bug")
            }
            Ok(())
        })
    }

    fn calls_sealed<'a>(&'a mut self, _count: u32) -> PortFuture<'a, Result<(), AgentError>> {
        Box::pin(async move {
            let _ = self.reached.send(PanicAt::Seal);
            if self.at == PanicAt::Seal {
                panic!("calls_sealed bug")
            }
            Ok(())
        })
    }

    fn commit<'a>(self: Box<Self>) -> PortFuture<'a, Result<(), AgentError>>
    where
        Self: 'a,
    {
        Box::pin(async move {
            let _ = self.reached.send(PanicAt::Commit);
            if self.at == PanicAt::Commit {
                panic!("commit bug")
            }
            Ok(())
        })
    }

    fn abort<'a>(
        self: Box<Self>,
        _reason: ToolBatchAbortReason,
    ) -> PortFuture<'a, Result<AbortClassification, AgentError>>
    where
        Self: 'a,
    {
        Box::pin(async move {
            let _ = self.reached.send(PanicAt::Abort);
            if self.at == PanicAt::Abort {
                panic!("abort bug")
            }
            Ok(self
                .slots
                .iter()
                .fold(AbortClassification::new(), |report, slot| {
                    report.cancelled_before_start(*slot)
                }))
        })
    }
}

/// 五个端口方法里任何一处 panic 都不许把运行挂住：每一处都变成那一步的失败回执，
/// 运行以稳定的终态结束。
#[tokio::test]
async fn a_panic_in_any_incremental_port_method_fails_the_run_instead_of_hanging_it() {
    for at in [
        PanicAt::Begin,
        PanicAt::Submit,
        PanicAt::Seal,
        PanicAt::Commit,
        PanicAt::Abort,
    ] {
        let (request_tx, mut requests) = mpsc::unbounded_channel();
        let (commands, command_rx) = mpsc::unbounded_channel();
        let (reached_tx, mut reached) = mpsc::unbounded_channel();
        let session = Arc::new(AgentSession::new(
            Arc::new(StaticPrompt),
            Arc::new(PanickingIncrementalRuntime {
                at,
                reached: reached_tx,
            }),
            Arc::new(NoCompaction),
            Arc::new(CommandStreamModel {
                requests: request_tx,
                commands: Mutex::new(command_rx),
            }),
            SessionConfig {
                // 关掉超时：挂住就是真的挂住，不许靠计时器兜底。
                tool_timeout: None,
                ..SessionConfig::default()
            },
        ));
        let running = session.clone();
        let task = tokio::spawn(async move {
            run_once(
                &running,
                vec![input("operator", format!("panic at {at:?}").as_str())],
            )
            .await
        });
        requests.recv().await.unwrap();
        let call = ToolCall::new("boom", "read", json!({}));
        commands
            .send(StreamCommand::Event(ModelStreamEvent::ToolCallReady {
                slot: ToolCallSlot::new(0),
                call: call.clone(),
            }))
            .unwrap();
        // 走到目标那一步为止再往下推流。
        assert_eq!(reached.recv().await, Some(PanicAt::Begin));
        if at != PanicAt::Begin {
            assert_eq!(reached.recv().await, Some(PanicAt::Submit));
        }
        match at {
            PanicAt::Begin | PanicAt::Submit => {}
            PanicAt::Seal | PanicAt::Commit => {
                commands
                    .send(StreamCommand::Event(ModelStreamEvent::ToolCallsSealed {
                        call_count: 1,
                    }))
                    .unwrap();
                assert_eq!(reached.recv().await, Some(PanicAt::Seal));
                if at == PanicAt::Commit {
                    commands
                        .send(StreamCommand::Complete(model_response(ModelOutput::calls(
                            vec![call.clone()],
                        ))))
                        .unwrap();
                    assert_eq!(reached.recv().await, Some(PanicAt::Commit));
                }
            }
            PanicAt::Abort => {
                commands
                    .send(StreamCommand::Fail(AgentError::new(
                        AgentErrorKind::Model,
                        "stream dropped",
                    )))
                    .unwrap();
                assert_eq!(reached.recv().await, Some(PanicAt::Abort));
            }
        }
        // 模型侧的流还开着（Begin/Submit/Seal 场景）：内核中止后会丢掉模型任务，
        // 这里不需要再发任何东西。运行必须在有限时间内给出终态。
        let outcome = tokio::time::timeout(std::time::Duration::from_secs(2), task)
            .await
            .unwrap_or_else(|_| panic!("{at:?} 处的 panic 让运行挂住了"))
            .unwrap()
            .unwrap();
        let TurnOutcome::Failed { error, stage, .. } = outcome else {
            panic!("{at:?}: 运行时 panic 应让运行失败，实际 {outcome:?}")
        };
        assert_eq!(stage, RunStage::Tools, "{at:?}");
        match at {
            // 什么都没交出去 / 所有者活着答了「没开始」：一次严格的运行时失败。
            PanicAt::Begin | PanicAt::Submit | PanicAt::Seal => {
                assert_eq!(
                    error.kind,
                    AgentErrorKind::ToolDispatch,
                    "{at:?}: {error:?}"
                );
                assert_eq!(error.summary, "tool_runtime_panicked", "{at:?}");
                assert!(session.pending_reconciliation().await.is_empty(), "{at:?}");
            }
            // commit 消费了批次对象；abort 连报告都给不出：在途的一律未知。
            PanicAt::Commit | PanicAt::Abort => {
                assert_eq!(
                    error.kind,
                    AgentErrorKind::ReconciliationRequired,
                    "{at:?}: {error:?}"
                );
                assert_eq!(session.pending_reconciliation().await.len(), 1, "{at:?}");
            }
        }
        assert!(!session.has_active_run().await, "{at:?}");
    }
}

// ── Release 必须真的撤掉所有者：卡在 submit 里的运行时、超时之后 ─────────

struct StuckSubmitRuntime {
    gate: Arc<Semaphore>,
    dropped: Arc<AtomicUsize>,
    later_calls: Arc<AtomicUsize>,
}

impl ToolRuntime for StuckSubmitRuntime {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition::new(
            "read",
            json!({"type": "object", "additionalProperties": true}),
        )]
    }

    fn dispatch<'a>(
        &'a self,
        _batch: ToolCallBatch,
    ) -> PortFuture<'a, Result<ToolResultBatch, AgentError>> {
        Box::pin(async { panic!("增量场景不该走 dispatch") })
    }

    fn begin_incremental<'a>(
        &'a self,
        _batch: ToolBatchStart,
        _reporter: Arc<dyn ToolCallReporter>,
    ) -> PortFuture<'a, Result<Option<Box<dyn IncrementalToolBatch + 'a>>, AgentError>> {
        Box::pin(async move {
            Ok(Some(Box::new(StuckBatch {
                gate: self.gate.clone(),
                dropped: self.dropped.clone(),
                later_calls: self.later_calls.clone(),
            }) as Box<dyn IncrementalToolBatch + 'a>))
        })
    }
}

struct StuckBatch {
    gate: Arc<Semaphore>,
    dropped: Arc<AtomicUsize>,
    later_calls: Arc<AtomicUsize>,
}

impl Drop for StuckBatch {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

impl IncrementalToolBatch for StuckBatch {
    fn submit<'a>(
        &'a mut self,
        _call: IncrementalToolCall,
    ) -> PortFuture<'a, Result<(), AgentError>> {
        Box::pin(async move {
            // 永远等不到的闸。
            let _permit = self.gate.acquire().await;
            Ok(())
        })
    }

    fn calls_sealed<'a>(&'a mut self, _count: u32) -> PortFuture<'a, Result<(), AgentError>> {
        self.later_calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok(()) })
    }

    fn commit<'a>(self: Box<Self>) -> PortFuture<'a, Result<(), AgentError>>
    where
        Self: 'a,
    {
        self.later_calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok(()) })
    }

    fn abort<'a>(
        self: Box<Self>,
        _reason: ToolBatchAbortReason,
    ) -> PortFuture<'a, Result<AbortClassification, AgentError>>
    where
        Self: 'a,
    {
        Box::pin(async { Ok(AbortClassification::new()) })
    }
}

/// 超时之后 `Release` 要连所有者任务一起撤：卡在 submit 里的 future 被丢、Box 被 drop、
/// 排在后面的 Seal / Commit 不再交给运行时——关通道是不够的（关闭的 receiver 会把缓冲排空）。
#[tokio::test]
async fn releasing_a_stuck_owner_drops_the_batch_and_never_delivers_the_queued_calls() {
    let gate = Arc::new(Semaphore::new(0));
    let dropped = Arc::new(AtomicUsize::new(0));
    let later_calls = Arc::new(AtomicUsize::new(0));
    let (request_tx, mut requests) = mpsc::unbounded_channel();
    let (commands, command_rx) = mpsc::unbounded_channel();
    let session = Arc::new(AgentSession::new(
        Arc::new(StaticPrompt),
        Arc::new(StuckSubmitRuntime {
            gate: gate.clone(),
            dropped: dropped.clone(),
            later_calls: later_calls.clone(),
        }),
        Arc::new(NoCompaction),
        Arc::new(CommandStreamModel {
            requests: request_tx,
            commands: Mutex::new(command_rx),
        }),
        SessionConfig {
            tool_timeout: Some(std::time::Duration::from_millis(100)),
            ..SessionConfig::default()
        },
    ));
    let running = session.clone();
    let task =
        tokio::spawn(async move { run_once(&running, vec![input("operator", "stuck")]).await });
    requests.recv().await.unwrap();
    let call = ToolCall::new("stuck", "read", json!({}));
    // 非流式形态：模型一次说完带调用；ModelDone 时才问运行时，Begin(true) 之后同一步排 Submit+Seal+Commit。
    commands
        .send(StreamCommand::Complete(model_response(ModelOutput::calls(
            vec![call],
        ))))
        .unwrap();
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(3), task)
        .await
        .expect("批次超时必须收口运行")
        .unwrap()
        .unwrap();
    assert!(matches!(
        outcome,
        TurnOutcome::Failed { ref error, .. } if error.kind == AgentErrorKind::ReconciliationRequired
    ));
    // Box 在有限时间内被 drop（任务被撤，不是等它自己回来）。
    for _ in 0..500 {
        if dropped.load(Ordering::SeqCst) == 1 {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(
        dropped.load(Ordering::SeqCst),
        1,
        "所有者任务没被撤，Box 还活着"
    );
    // 放闸也无济于事：缓冲里的 Seal / Commit 不会再交给运行时。
    gate.add_permits(10);
    for _ in 0..50 {
        tokio::task::yield_now().await;
    }
    assert_eq!(later_calls.load(Ordering::SeqCst), 0);
}

// ── store panic 不许带走循环；丢弃外壳会关掉循环 ─────────────────────

struct PanickingStore {
    inner: InMemoryCheckpointStore,
    writes: AtomicUsize,
    panic_at: usize,
}

impl CheckpointStore for PanickingStore {
    fn load<'a>(
        &'a self,
        session_id: &'a SessionId,
    ) -> crate::persistence::PersistenceFuture<
        'a,
        Result<Option<crate::persistence::StoredCheckpoint>, PersistenceError>,
    > {
        self.inner.load(session_id)
    }

    fn compare_and_swap<'a>(
        &'a self,
        expected: Option<CheckpointRevision>,
        checkpoint: SessionCheckpoint,
    ) -> crate::persistence::PersistenceFuture<
        'a,
        Result<crate::persistence::StoredCheckpoint, PersistenceError>,
    > {
        let n = self.writes.fetch_add(1, Ordering::SeqCst) + 1;
        if n == self.panic_at {
            panic!("store bug on write {n}")
        }
        self.inner.compare_and_swap(expected, checkpoint)
    }
}

/// store 是用户代码：它 panic 不许带走唯一的循环。不知道 CAS 有没有提交，按「已提交但确认丢失」
/// 隔离——之后每个公共方法都还能回答，答案是「被隔离了，重开」。
#[tokio::test]
async fn a_panicking_store_fences_the_session_instead_of_killing_the_loop() {
    let store = Arc::new(PanickingStore {
        inner: InMemoryCheckpointStore::new(),
        writes: AtomicUsize::new(0),
        // 第 1 次是 open 时创建；第 2 次是第一条 enqueue 的 Persist。
        panic_at: 2,
    });
    let (request_tx, _requests) = mpsc::unbounded_channel();
    let (_responses, response_rx) = mpsc::unbounded_channel();
    let session = Arc::new(
        AgentSession::open(
            SessionId::new("supervision/panicking-store").unwrap(),
            store,
            Arc::new(StaticPrompt),
            Arc::new(FailingBatchRuntime::default()),
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
    let rejected = session
        .enqueue(MailboxInput::next_model_request(vec![input(
            "operator", "boom",
        )]))
        .await
        .unwrap_err();
    assert_eq!(
        rejected.reason,
        MailboxRejectedReason::PersistenceCommitUnknown
    );
    // 循环还活着，而且知道自己被隔离了。
    assert_eq!(
        session.persistence_fence_reason().await,
        Some(AgentErrorKind::PersistenceCommitUnknown)
    );
    let error = session
        .reschedule(Delivery::Passive, Delivery::NextModelRequest)
        .await
        .unwrap_err();
    assert_eq!(error.kind, AgentErrorKind::PersistenceCommitUnknown);
    assert!(session.conversation().await.is_empty());
}

/// 丢弃外壳 = 关掉会话：循环退出，还握着的运行句柄得到 session_loop_closed。
#[tokio::test]
async fn dropping_the_session_shuts_the_loop_down() {
    let (request_tx, mut requests) = mpsc::unbounded_channel();
    let (_responses, response_rx) = mpsc::unbounded_channel();
    let session = Arc::new(AgentSession::new(
        Arc::new(StaticPrompt),
        Arc::new(FailingBatchRuntime::default()),
        Arc::new(NoCompaction),
        Arc::new(ChannelModel {
            requests: request_tx,
            responses: Mutex::new(response_rx),
        }),
        SessionConfig::default(),
    ));
    let handle = start(&session, vec![input("operator", "bye")])
        .await
        .unwrap();
    requests.recv().await.unwrap();
    drop(session);
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(2), handle.join())
        .await
        .expect("循环退出后句柄必须返回");
    assert!(matches!(
        outcome,
        TurnOutcome::Failed { error, .. }
            if error.kind == AgentErrorKind::InvalidState && error.summary == "session_loop_closed"
    ));
}
