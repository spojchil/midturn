//! 生命周期：空闲开跑、忙时搭车、自动开始开关、open 时不开跑。

use super::*;

#[tokio::test]
async fn input_arriving_after_a_run_ends_starts_the_next_one_by_itself() {
    let mut fixture = fixture();
    let handle = start(&fixture.session, vec![input("operator", "first")])
        .await
        .unwrap();
    fixture.requests.recv().await.unwrap();
    fixture
        .responses
        .send(Ok(model_response(ModelOutput::text("first done"))))
        .unwrap();
    handle.join().await;
    fixture.session.wait_until_idle().await;

    // 空闲之后再投递：没有人调用任何「启动」接口，会话自己醒过来。
    let started = fixture
        .session
        .enqueue(MailboxInput::next_model_request(vec![input(
            "tool",
            "late result arrives while idle",
        )]))
        .await
        .unwrap();
    assert!(matches!(started, Enqueued::Started(_)));
    let second = fixture.requests.recv().await.unwrap();
    assert_eq!(
        visible_texts(&second.transcript).last().map(String::as_str),
        Some("late result arrives while idle")
    );
    fixture
        .responses
        .send(Ok(model_response(ModelOutput::text("second done"))))
        .unwrap();
    fixture.session.wait_until_idle().await;
}

#[tokio::test]
async fn disabling_auto_start_holds_input_until_it_is_armed_again() {
    let mut fixture = fixture();
    fixture.session.set_auto_start(false).await;

    let held = fixture
        .session
        .enqueue(MailboxInput::next_model_request(vec![input(
            "operator", "queued",
        )]))
        .await
        .unwrap();
    assert!(matches!(
        held,
        Enqueued::Held(HoldReason::AutoStartDisabled)
    ));
    assert!(fixture.requests.try_recv().is_err());

    // 重新武装时立刻兑现攒下来的输入。
    fixture.session.set_auto_start(true).await;
    let request = fixture.requests.recv().await.unwrap();
    assert_eq!(
        visible_texts(&request.transcript)
            .last()
            .map(String::as_str),
        Some("queued")
    );
    fixture
        .responses
        .send(Ok(model_response(ModelOutput::text("done"))))
        .unwrap();
    fixture.session.wait_until_idle().await;
}

#[tokio::test]
async fn idle_starts_a_run_and_busy_joins_the_active_one() {
    let mut fixture = fixture();

    // 空闲不是拒绝的理由，而是开跑的理由。
    let started = fixture
        .session
        .enqueue(MailboxInput::next_model_request(vec![input(
            "custom", "idle",
        )]))
        .await
        .unwrap();
    let Enqueued::Started(handle) = started else {
        panic!("空闲会话应因这次投递开跑，实际是 {started:?}");
    };
    fixture.requests.recv().await.unwrap();

    // 已经在跑时投递同样不拒，内容会在下一个边界进入本轮。
    assert!(matches!(
        fixture
            .session
            .enqueue(MailboxInput::next_model_request(vec![input(
                "operator", "steer"
            )]))
            .await
            .unwrap(),
        Enqueued::Pending
    ));

    fixture
        .responses
        .send(Ok(model_response(ModelOutput::text("first"))))
        .unwrap();
    // 插话把本该结束的运行又拉回一次模型请求。
    let second = fixture.requests.recv().await.unwrap();
    assert_eq!(
        visible_texts(&second.transcript).last().map(String::as_str),
        Some("steer")
    );
    fixture
        .responses
        .send(Ok(model_response(ModelOutput::text("done"))))
        .unwrap();
    assert!(matches!(
        handle.join().await,
        TurnOutcome::Completed { output, .. } if output.text_content() == "done"
    ));
}

/// 只想打开 checkpoint 看看的程序：`auto_start: false` 让 `Boot` 不开跑。
#[tokio::test]
async fn opening_with_auto_start_off_does_not_run_on_boot() {
    let store = Arc::new(InMemoryCheckpointStore::new());
    let session_id = SessionId::new("mailbox/open-quiet").unwrap();
    let (request_tx, mut request_rx) = mpsc::unbounded_channel();
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
            SessionConfig {
                auto_start: false,
                ..SessionConfig::default()
            },
        )
        .await
        .unwrap(),
    );
    assert!(matches!(
        first
            .enqueue(MailboxInput::next_model_request(vec![input(
                "user", "later"
            )]))
            .await
            .unwrap(),
        Enqueued::Held(HoldReason::AutoStartDisabled)
    ));
    drop(first);

    let (request_tx, mut quiet_rx) = mpsc::unbounded_channel();
    let (_response_tx, response_rx) = mpsc::unbounded_channel();
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
            SessionConfig {
                auto_start: false,
                ..SessionConfig::default()
            },
        )
        .await
        .unwrap(),
    );
    // 第一次异步调用会 spawn 循环并 Boot：信箱有触发内容，但开关关着。
    assert_eq!(reopened.mailbox_snapshot().await.len(), 1);
    assert!(quiet_rx.try_recv().is_err(), "关着的会话不该发模型请求");
    assert!(request_rx.try_recv().is_err());
    // 打开开关：立刻兑现。
    reopened.set_auto_start(true).await;
    let request = quiet_rx.recv().await.unwrap();
    assert!(visible_texts(&request.transcript).contains(&"later".to_owned()));
}
