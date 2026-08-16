//! 预算：累计预算跨运行、跨重开；被动内容也计容量。

use super::*;

#[tokio::test]
async fn cumulative_budget_spans_runs_and_disarms_auto_start() {
    // 运行可以自动开始，按轮计数每轮都归零；真正要封口的是整个会话消耗了多少。
    let mut fixture = fixture_with_config(SessionConfig {
        budget: SessionBudget {
            cumulative: CumulativeBudget {
                max_model_requests: Some(2),
                ..CumulativeBudget::default()
            },
            ..SessionBudget::default()
        },
        ..SessionConfig::default()
    });

    // 第一轮用掉一次。
    let handle = start(&fixture.session, vec![input("operator", "第一轮")])
        .await
        .unwrap();
    fixture.requests.recv().await.unwrap();
    fixture
        .responses
        .send(Ok(model_response(ModelOutput::text("一"))))
        .unwrap();
    handle.join().await;
    fixture.session.wait_until_idle().await;

    // 第二轮用掉第二次——如果计数按轮归零，这里和下一轮都不会撞上限。
    let handle = start(&fixture.session, vec![input("operator", "第二轮")])
        .await
        .unwrap();
    fixture.requests.recv().await.unwrap();
    fixture
        .responses
        .send(Ok(model_response(ModelOutput::text("二"))))
        .unwrap();
    handle.join().await;
    fixture.session.wait_until_idle().await;

    // 第三轮：额度已经用完，这一轮必须失败。
    let handle = start(&fixture.session, vec![input("operator", "第三轮")])
        .await
        .unwrap();
    let outcome = handle.join().await;
    let TurnOutcome::Failed { error, .. } = outcome else {
        panic!("累计额度用完后不该还能发请求");
    };
    assert_eq!(error.kind, AgentErrorKind::BudgetExceeded);
    assert_eq!(error.summary, "session_model_request_budget_exhausted");
    assert!(
        fixture.requests.try_recv().is_err(),
        "不应真的发出第三次请求"
    );

    // 而且自动启动要被关掉：否则之后每一条进信箱的消息都会触发一轮注定失败的运行。
    assert!(matches!(
        fixture
            .session
            .enqueue(MailboxInput::next_model_request(vec![input(
                "operator",
                "还想再问"
            )]))
            .await
            .unwrap(),
        Enqueued::Held(HoldReason::AutoStartDisabled)
    ));
}

#[tokio::test]
async fn consumption_survives_reopen_so_the_limit_cannot_be_reset() {
    // 不持久化的话，重开一次进程就能把限额清零，这个预算也就形同虚设。
    let store = Arc::new(InMemoryCheckpointStore::new());
    let session_id = SessionId::new("budget/persistence").unwrap();
    let (request_tx, mut request_rx) = mpsc::unbounded_channel();
    let (response_tx, response_rx) = mpsc::unbounded_channel();
    let config = SessionConfig {
        budget: SessionBudget {
            cumulative: CumulativeBudget {
                max_model_requests: Some(1),
                ..CumulativeBudget::default()
            },
            ..SessionBudget::default()
        },
        ..SessionConfig::default()
    };

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
            config.clone(),
        )
        .await
        .unwrap(),
    );
    let handle = start(&first, vec![input("operator", "用掉唯一一次")])
        .await
        .unwrap();
    request_rx.recv().await.unwrap();
    response_tx
        .send(Ok(model_response(ModelOutput::text("好"))))
        .unwrap();
    handle.join().await;
    first.wait_until_idle().await;

    // 用同一个 store 重开：消耗量必须跟着回来。
    let (second_request_tx, mut second_request_rx) = mpsc::unbounded_channel();
    let (_second_response_tx, second_response_rx) = mpsc::unbounded_channel();
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
            config,
        )
        .await
        .unwrap(),
    );
    let handle = start(&reopened, vec![input("operator", "重开之后再问")])
        .await
        .unwrap();
    let TurnOutcome::Failed { error, .. } = handle.join().await else {
        panic!("重开不该把额度清零");
    };
    assert_eq!(error.kind, AgentErrorKind::BudgetExceeded);
    assert!(second_request_rx.try_recv().is_err());
}

#[tokio::test]
async fn passive_delivery_counts_against_mailbox_capacity() {
    // 容量上限是整个信箱的，不是某一类投递的：漏算 passive 会让「不叫醒模型」的
    // 内容无界增长，而迟到的工具结果正是它的主要生产者。
    let fixture = fixture_with_config(SessionConfig {
        max_mailbox_items: 2,
        ..SessionConfig::default()
    });
    for _ in 0..2 {
        assert!(matches!(
            fixture
                .session
                .enqueue(MailboxInput::passive(vec![input("tool", "late")]))
                .await
                .unwrap(),
            Enqueued::Held(HoldReason::Passive)
        ));
    }
    let rejected = fixture
        .session
        .enqueue(MailboxInput::passive(vec![input("tool", "overflow")]))
        .await
        .unwrap_err();
    assert_eq!(rejected.reason, MailboxRejectedReason::Capacity);
}
