//! 改期：跨重开保留、超字节上限回滚。

use super::*;

/// 改主意必须落盘，否则崩溃一次就退回原来的交付语义。
///
/// 「先别急，等它跑完再说」如果只改内存，进程重开后那句话又变回下一次请求前就进模型——
/// 用户明确表达过的意图被一次崩溃抹掉了。
#[tokio::test]
async fn rescheduling_survives_reopen() {
    let store = Arc::new(InMemoryCheckpointStore::new());
    let session_id = SessionId::new("mailbox/reschedule").unwrap();
    // 第一个会话关掉了自动开始，所以它不会发出任何请求；接收端只需保持存活。
    let (request_tx, _request_rx) = mpsc::unbounded_channel();
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
    // 关掉自动开始，这样内容只是躺在信箱里，不会被跑掉。
    first.set_auto_start(false).await;
    first
        .enqueue(MailboxInput::next_model_request(vec![input(
            "user",
            "先别急",
        )]))
        .await
        .unwrap();
    let moved = first
        .reschedule(Delivery::NextModelRequest, Delivery::WhenIdle)
        .await
        .unwrap();
    assert_eq!(moved.envelopes, 1);

    // 盘上那句话必须仍然是「等它跑完再说」，而不是被退回成「下一次请求就带上」。
    let stored = store.load(&session_id).await.unwrap().unwrap();
    assert_eq!(stored.checkpoint.mailbox.len(), 1);
    assert_eq!(stored.checkpoint.mailbox[0].delivery, Delivery::WhenIdle);
    drop(first);

    // 同一个 store 重开：会话空闲、信箱有触发内容、自动开始默认打开——那句话
    // 自己就把运行叫起来了。「等它跑完再说」在没有「它」的时候就是「现在」。
    let (second_request_tx, mut second_request_rx) = mpsc::unbounded_channel();
    let (second_response_tx, second_response_rx) = mpsc::unbounded_channel();
    let reopened = Arc::new(
        AgentSession::open(
            session_id,
            store,
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
    assert!(reopened.try_resume().await.unwrap().is_none());
    let first_request = second_request_rx.recv().await.unwrap();
    let texts = visible_texts(&first_request.transcript);
    assert!(
        texts.contains(&"先别急".to_owned()),
        "改期不是丢弃：重开后空闲，它就是这次运行的理由：{texts:?}"
    );
    second_response_tx
        .send(Ok(model_response(ModelOutput::text("回答一"))))
        .unwrap();
    reopened.wait_until_idle().await;
    assert!(reopened.mailbox_snapshot().await.is_empty());
}

/// 改期推过字节上限时的回滚必须还原三个队列的原貌——反向搬一次会把目标队列的原住民一起搬走。
#[tokio::test]
async fn reschedule_over_byte_budget_restores_all_three_queues() {
    // 三类各放一封；上限卡在「改期前刚好够、改期后差一点」。
    let next = MailboxInput::next_model_request(vec![input("user", "next-1")]);
    let idle = MailboxInput::when_idle(vec![input("user", "idle-1")]);
    let passive = MailboxInput::passive(vec![input("user", "passive-1")]);
    let bytes = |input: &MailboxInput| serde_json::to_vec(input).unwrap().len();
    let before = bytes(&next) + bytes(&idle) + bytes(&passive);
    let fixture = fixture_with_config(SessionConfig {
        max_mailbox_bytes: before + 4,
        ..SessionConfig::default()
    });
    fixture.session.set_auto_start(false).await;
    for envelope in [next.clone(), idle.clone(), passive.clone()] {
        post(&fixture.session, envelope).await.unwrap();
    }
    // passive(7 字节标签) → next_model_request(18 字节标签)：+11 字节，超线。
    let error = fixture
        .session
        .reschedule(Delivery::Passive, Delivery::NextModelRequest)
        .await
        .unwrap_err();
    assert_eq!(error.summary, "reschedule_exceeds_mailbox_byte_budget");
    assert_eq!(
        fixture.session.mailbox_snapshot().await,
        vec![next, idle, passive],
        "失败的改期不能改变任何一类的成员或顺序"
    );
}
