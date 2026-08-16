//! 观测：事件、内容观察端、观察端 panic 隔离。

use super::*;
use crate::events::EventKind;

#[test]
fn content_ir_keeps_non_text_values_available_to_adapters() {
    let value = Value::String("raw".to_owned());
    assert_eq!(
        ContentPart::json(value.clone()),
        ContentPart::Json { value }
    );
    assert_eq!(Delivery::NextModelRequest, Delivery::NextModelRequest);
    assert_eq!(ToolCallId::new("x").as_str(), "x");
}

struct PanicEnabledObserver;

impl Observer for PanicEnabledObserver {
    fn enabled(&self, _metadata: EventMetadata) -> bool {
        panic!("enabled panic")
    }

    fn observe(&self, _event: &AgentEvent) {
        unreachable!()
    }
}

struct PanicObserveObserver;

impl Observer for PanicObserveObserver {
    fn observe(&self, _event: &AgentEvent) {
        panic!("observe panic")
    }
}

struct PanicEnabledStreamObserver;

impl StreamObserver for PanicEnabledStreamObserver {
    fn enabled(&self) -> bool {
        panic!("stream enabled panic")
    }

    fn observe(&self, _event: &ObservedModelStreamEvent) {
        unreachable!()
    }
}

struct PanicObserveStreamObserver;

impl StreamObserver for PanicObserveStreamObserver {
    fn observe(&self, _event: &ObservedModelStreamEvent) {
        panic!("stream observe panic")
    }
}

/// 观察端的 `enabled` 或 `observe` panic 都不许影响运行：前者当作「关掉」，
/// 后者只丢它自己那一条。
#[tokio::test]
async fn observer_panics_are_isolated_from_the_run() {
    let pairs: Vec<(Arc<dyn Observer>, Arc<dyn StreamObserver>)> = vec![
        (
            Arc::new(PanicEnabledObserver),
            Arc::new(PanicEnabledStreamObserver),
        ),
        (
            Arc::new(PanicObserveObserver),
            Arc::new(PanicObserveStreamObserver),
        ),
    ];
    for (observer, stream) in pairs {
        let (request_tx, mut request_rx) = mpsc::unbounded_channel();
        let (response_tx, response_rx) = mpsc::unbounded_channel();
        let session = Arc::new(
            AgentSession::new(
                Arc::new(StaticPrompt),
                Arc::new(FailingBatchRuntime::default()),
                Arc::new(NoCompaction),
                Arc::new(ChannelModel {
                    requests: request_tx,
                    responses: Mutex::new(response_rx),
                }),
                SessionConfig::default(),
            )
            .with_observer(observer)
            .with_stream_observer(stream),
        );
        let running = session.clone();
        let task =
            tokio::spawn(async move { run_once(&running, vec![input("operator", "hi")]).await });
        request_rx.recv().await.unwrap();
        response_tx
            .send(Ok(model_response(ModelOutput::text("ok"))))
            .unwrap();
        assert!(matches!(
            task.await.unwrap().unwrap(),
            TurnOutcome::Completed { output, .. } if output.text_content() == "ok"
        ));
    }
}

/// 收下完整事件，供载荷用例检查内容与顺序。
#[derive(Default)]
struct EventRecordingObserver {
    events: StdMutex<Vec<AgentEvent>>,
}

impl Observer for EventRecordingObserver {
    fn observe(&self, event: &AgentEvent) {
        self.events.lock().unwrap().push(event.clone());
    }
}

#[derive(Default)]
struct ContentRecordingObserver {
    events: StdMutex<Vec<ContentEvent>>,
}

impl ContentObserver for ContentRecordingObserver {
    fn observe(&self, event: &ContentEvent) {
        self.events.lock().unwrap().push(event.clone());
    }
}

/// 跑一轮「模型要调工具 → 工具返回 → 模型收尾」，可选地装上内容观察端。
async fn run_two_request_turn(
    content: Option<Arc<ContentRecordingObserver>>,
) -> Arc<EventRecordingObserver> {
    let (request_tx, mut request_rx) = mpsc::unbounded_channel();
    let (response_tx, response_rx) = mpsc::unbounded_channel();
    let (batch_tx, _batch_rx) = mpsc::unbounded_channel();
    let recorder = Arc::new(EventRecordingObserver::default());
    // 故意用最高级别：证明「拿不到内容」不是被等级挡住的，而是根本不在这个端口上。
    let mut session = AgentSession::new(
        Arc::new(StaticPrompt),
        Arc::new(GatedBatchRuntime {
            batches: batch_tx,
            gate: Arc::new(Semaphore::new(8)),
            dispatches: AtomicUsize::new(0),
        }),
        Arc::new(NoCompaction),
        Arc::new(ChannelModel {
            requests: request_tx,
            responses: Mutex::new(response_rx),
        }),
        SessionConfig::default(),
    )
    .with_observer(Arc::new(FilteredObserver::new(
        recorder.clone(),
        LevelFilter::Trace,
    )));
    if let Some(content) = content {
        session = session.with_content_observer(content);
    }
    let session = Arc::new(session);

    let handle = start(&session, vec![input("operator", "载荷用例")])
        .await
        .unwrap();
    request_rx.recv().await.unwrap();
    response_tx
        .send(Ok(model_response(ModelOutput::calls(vec![ToolCall::new(
            "call-1",
            "read",
            json!({"path": "a"}),
        )]))))
        .unwrap();
    request_rx.recv().await.unwrap();
    response_tx
        .send(Ok(model_response(ModelOutput::text("好了"))))
        .unwrap();
    handle.join().await;
    recorder
}

/// 内容不是靠等级挡住的，是根本不在 `Observer` 这个端口上。
///
/// 这条用例守的边界：载荷若是 `AgentEvent` 的一个 Trace 变体，只要消费者自己实现
/// `Observer`（`enabled` 默认 `true`）或把等级调到 Trace，提示词和工具结果就会
/// 静默进日志。等级是「有多啰嗦」，不是「有没有凭据」。
#[tokio::test]
async fn observer_never_carries_content_even_at_trace() {
    let events = run_two_request_turn(None).await;
    let events = events.events.lock().unwrap().clone();

    // 元数据照常，而且是最高级别下的全量。
    assert!(events
        .iter()
        .any(|event| matches!(event, AgentEvent::ToolBatchFinished { .. })));
    assert!(events
        .iter()
        .any(|event| matches!(event, AgentEvent::MailboxEnqueued { .. })));

    // `{:?}` 是消费者最常见的写法；正文一个字都不该出现在里面。
    let dumped = events
        .iter()
        .map(|event| format!("{event:?}"))
        .collect::<String>();
    for leaked in ["载荷用例", "好了", "path", "tool-result"] {
        assert!(
            !dumped.contains(leaked),
            "元数据端口泄漏了内容 {leaked:?}：{dumped}"
        );
    }
}

/// 装上 `ContentObserver` 才拿得到内容，且必须能和配对的元数据事件对上号。
#[tokio::test]
async fn content_observer_carries_full_content_and_pairs_by_identifiers() {
    let content = Arc::new(ContentRecordingObserver::default());
    let meta = run_two_request_turn(Some(content.clone())).await;
    let meta = meta.events.lock().unwrap().clone();
    let content = content.events.lock().unwrap().clone();

    let transcripts: Vec<_> = content
        .iter()
        .filter_map(|event| match event {
            ContentEvent::ModelRequestTranscript {
                request_index,
                transcript,
                function_tools,
                ..
            } => Some((*request_index, transcript, function_tools)),
            _ => None,
        })
        .collect();
    let outputs: Vec<_> = content
        .iter()
        .filter_map(|event| match event {
            ContentEvent::ModelResponseOutput {
                request_index,
                output,
                ..
            } => Some((*request_index, output)),
            _ => None,
        })
        .collect();
    let results: Vec<_> = content
        .iter()
        .filter_map(|event| match event {
            ContentEvent::ToolBatchResults {
                batch_id, results, ..
            } => Some((batch_id, results)),
            _ => None,
        })
        .collect();

    assert_eq!(transcripts.len(), 2, "两次模型请求各配一条完整对话");
    assert_eq!(visible_texts(transcripts[0].1).last().unwrap(), "载荷用例");
    assert!(!transcripts[0].2.is_empty(), "工具声明也要给全");
    assert_eq!(outputs.len(), 2, "每一轮模型输出都拿得到");
    assert_eq!(outputs[1].1.text_content(), "好了");
    assert_eq!(results.len(), 1, "工具批的真实结果");
    assert_eq!(results[0].1.results[0].call_id.as_str(), "call-1");

    // 配对靠标识符，不靠「挨着」。request_index 被写死成常量就会在这里失败。
    let meta_request_indices: Vec<u64> = meta
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ModelRequestStarted { request_index, .. } => Some(*request_index),
            _ => None,
        })
        .collect();
    assert_eq!(meta_request_indices, vec![1, 2]);
    assert_eq!(
        transcripts.iter().map(|entry| entry.0).collect::<Vec<_>>(),
        meta_request_indices,
        "内容事件的 request_index 必须与配对的元数据事件一致"
    );
    assert_eq!(
        outputs.iter().map(|entry| entry.0).collect::<Vec<_>>(),
        meta_request_indices,
    );
    let meta_batch_id = meta
        .iter()
        .find_map(|event| match event {
            AgentEvent::ToolBatchFinished { batch_id, .. } => Some(batch_id.clone()),
            _ => None,
        })
        .expect("工具批完成事件");
    assert_eq!(results[0].0, &meta_batch_id);

    // 两个端口共用一个计数器：合并后必须严格递增且无空洞。
    let mut sequences: Vec<u64> = meta
        .iter()
        .map(AgentEvent::sequence)
        .chain(content.iter().map(ContentEvent::sequence))
        .collect();
    sequences.sort_unstable();
    assert_eq!(
        sequences,
        (1..=sequences.len() as u64).collect::<Vec<_>>(),
        "被投递的事件应当占满 1..=n，没有空洞"
    );
}

#[derive(Default)]
struct DebugRecordingObserver {
    lines: StdMutex<Vec<String>>,
}

impl Observer for DebugRecordingObserver {
    fn observe(&self, event: &AgentEvent) {
        self.lines.lock().unwrap().push(format!("{event:?}"));
    }
}

#[tokio::test]
async fn core_failure_event_does_not_copy_provider_error_body() {
    let sensitive = "SENSITIVE_PROVIDER_BODY_SENTINEL";
    let (request_tx, mut request_rx) = mpsc::unbounded_channel();
    let (response_tx, response_rx) = mpsc::unbounded_channel();
    let (batch_tx, _batch_rx) = mpsc::unbounded_channel();
    let observer = Arc::new(DebugRecordingObserver::default());
    let session = Arc::new(
        AgentSession::new(
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
        .with_observer(observer.clone()),
    );

    let running = session.clone();
    let task =
        tokio::spawn(async move { run_once(&running, vec![input("user", "request")]).await });
    request_rx.recv().await.unwrap();
    response_tx
        .send(Err(AgentError::new(AgentErrorKind::Model, sensitive)))
        .unwrap();
    let outcome = task.await.unwrap().unwrap();
    assert!(matches!(outcome, TurnOutcome::Failed { error, .. } if error.summary == sensitive));
    assert!(observer
        .lines
        .lock()
        .unwrap()
        .iter()
        .all(|line| !line.contains(sensitive)));
}

/// `reschedule` 有自己的元数据事件：不带内容，只说「哪一类、改成什么、几封」。
/// 什么都没动就没有事件——也没有落盘。
#[tokio::test]
async fn reschedule_is_observable_without_content() {
    let recorder = Arc::new(EventRecordingObserver::default());
    let (request_tx, _request_rx) = mpsc::unbounded_channel();
    let (_response_tx, response_rx) = mpsc::unbounded_channel();
    let session = Arc::new(
        AgentSession::new(
            Arc::new(StaticPrompt),
            Arc::new(FailingBatchRuntime::default()),
            Arc::new(NoCompaction),
            Arc::new(ChannelModel {
                requests: request_tx,
                responses: Mutex::new(response_rx),
            }),
            SessionConfig::default(),
        )
        .with_observer(Arc::new(FilteredObserver::new(
            recorder.clone(),
            LevelFilter::Trace,
        ))),
    );
    session.set_auto_start(false).await;
    session
        .enqueue(MailboxInput::next_model_request(vec![input(
            "user",
            "先别急",
        )]))
        .await
        .unwrap();
    session
        .enqueue(MailboxInput::next_model_request(vec![input(
            "user",
            "这条也是",
        )]))
        .await
        .unwrap();

    let moved = session
        .reschedule(Delivery::NextModelRequest, Delivery::WhenIdle)
        .await
        .unwrap();
    assert_eq!(moved.envelopes, 2);
    let nothing = session
        .reschedule(Delivery::NextModelRequest, Delivery::WhenIdle)
        .await
        .unwrap();
    assert_eq!(nothing.envelopes, 0);
    assert!(matches!(nothing.outcome, RescheduleOutcome::Unchanged));

    // 事件在 drainer 里异步投递；等会话空闲不够（它本来就空闲），给 drainer 一个让步窗口。
    for _ in 0..50 {
        if recorder
            .events
            .lock()
            .unwrap()
            .iter()
            .any(|event| matches!(event, AgentEvent::MailboxRescheduled { .. }))
        {
            break;
        }
        tokio::task::yield_now().await;
    }
    let events = recorder.events.lock().unwrap();
    let rescheduled: Vec<_> = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::MailboxRescheduled { .. }))
        .collect();
    assert_eq!(rescheduled.len(), 1, "没动的那次不发事件");
    assert!(matches!(
        rescheduled[0],
        AgentEvent::MailboxRescheduled {
            run_id: None,
            from: Delivery::NextModelRequest,
            to: Delivery::WhenIdle,
            envelopes: 2,
            ..
        }
    ));
    let sequences: Vec<u64> = events.iter().map(AgentEvent::sequence).collect();
    assert!(sequences.windows(2).all(|pair| pair[0] < pair[1]));
}

/// `enabled` 返回 false 或 panic 都不取号；`observe` panic 取了号——那才是唯一的洞。
struct HoleObserver {
    events: StdMutex<Vec<AgentEvent>>,
}

impl Observer for HoleObserver {
    fn enabled(&self, metadata: EventMetadata) -> bool {
        match metadata.kind {
            // 过滤掉：不许耗号。
            EventKind::MailboxEnqueued => false,
            // enabled 里炸：当作过滤掉，也不许耗号。
            EventKind::BoundaryDrained => panic!("enabled bug"),
            _ => true,
        }
    }

    fn observe(&self, event: &AgentEvent) {
        // observe 里炸：号已经取了，后面的人会看到一个洞。
        if event.kind() == EventKind::ModelRequestStarted {
            panic!("observe bug")
        }
        self.events.lock().unwrap().push(event.clone());
    }
}

#[tokio::test]
async fn filtering_never_leaves_a_hole_but_an_observe_panic_does() {
    let observer = Arc::new(HoleObserver {
        events: StdMutex::new(Vec::new()),
    });
    let (request_tx, mut request_rx) = mpsc::unbounded_channel();
    let (response_tx, response_rx) = mpsc::unbounded_channel();
    let session = Arc::new(
        AgentSession::new(
            Arc::new(StaticPrompt),
            Arc::new(FailingBatchRuntime::default()),
            Arc::new(NoCompaction),
            Arc::new(ChannelModel {
                requests: request_tx,
                responses: Mutex::new(response_rx),
            }),
            SessionConfig::default(),
        )
        .with_observer(observer.clone()),
    );
    let running = session.clone();
    let task = tokio::spawn(async move { run_once(&running, vec![input("operator", "hi")]).await });
    request_rx.recv().await.unwrap();
    response_tx
        .send(Ok(model_response(ModelOutput::text("ok"))))
        .unwrap();
    task.await.unwrap().unwrap();
    // 投递是异步的：等 RunCompleted 到达。
    for _ in 0..200 {
        if observer
            .events
            .lock()
            .unwrap()
            .iter()
            .any(|event| matches!(event, AgentEvent::RunCompleted { .. }))
        {
            break;
        }
        tokio::task::yield_now().await;
    }
    let events = observer.events.lock().unwrap();
    let kinds: Vec<EventKind> = events.iter().map(AgentEvent::kind).collect();
    assert!(!kinds.contains(&EventKind::MailboxEnqueued));
    assert!(!kinds.contains(&EventKind::BoundaryDrained));
    assert!(!kinds.contains(&EventKind::ModelRequestStarted));
    let sequences: Vec<u64> = events.iter().map(AgentEvent::sequence).collect();
    // RunStarted 是第一条被观察到的：MailboxEnqueued（过滤）不耗号，所以它是 1。
    assert_eq!(sequences.first(), Some(&1), "{sequences:?}");
    // 唯一的洞：ModelRequestStarted 取了 2 号然后在 observe 里炸了。
    let mut expected = 1u64;
    let mut holes = 0;
    for sequence in &sequences {
        while *sequence > expected {
            holes += 1;
            expected += 1;
        }
        assert_eq!(*sequence, expected, "{sequences:?}");
        expected += 1;
    }
    assert_eq!(holes, 1, "只有 observe panic 会造洞：{sequences:?}");
    assert!(
        sequences.contains(&3),
        "BoundaryDrained（enabled panic）没耗号：{sequences:?}"
    );
}
