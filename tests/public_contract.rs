//! 从外部使用者视角验证最重要的公开状态机契约。

use midturn::persistence::{InMemoryCheckpointStore, SessionCheckpoint, SessionId};
use midturn::{
    InputMessage, MailboxInput, ModelOutput, ModelRequest, ModelResponse, RequestBoundaryKind,
    RunHandle, RunId, ToolCall, ToolCallId, ToolResult, ToolResultBatch, TranscriptItem, Turn,
    TurnStep,
};
use serde_json::json;

#[test]
fn public_api_exposes_one_complete_tool_batch_and_ordered_results() {
    let mut turn = Turn::new(RunId::new("public-run"), Vec::new());
    assert_eq!(
        turn.next_step().unwrap(),
        TurnStep::RequestBoundary {
            kind: RequestBoundaryKind::BeforeModelRequest,
        }
    );
    turn.resume_boundary(vec![InputMessage::text("operator", "开始").into()])
        .unwrap();
    assert!(matches!(
        turn.next_step().unwrap(),
        TurnStep::CallModel { .. }
    ));

    turn.model_response(ModelResponse {
        output: ModelOutput::calls(vec![
            ToolCall::new("call-a", "read", json!({"path": "a"})),
            ToolCall::new("call-b", "read", json!({"path": "b"})),
        ]),
        ..ModelResponse::default()
    })
    .unwrap();

    let TurnStep::DispatchTools { batch } = turn.next_step().unwrap() else {
        panic!("应一次公开完整工具批");
    };
    assert_eq!(batch.calls.len(), 2);

    turn.tool_results(ToolResultBatch {
        results: vec![
            ToolResult::success_json(ToolCallId::new("call-b"), json!(2)),
            ToolResult::success_json(ToolCallId::new("call-a"), json!(1)),
        ],
    })
    .unwrap();

    let TranscriptItem::ToolResults(results) = turn.transcript().last().unwrap() else {
        panic!("结果批应写入公开 transcript");
    };
    assert_eq!(results.results[0].call_id.as_str(), "call-a");
    assert_eq!(results.results[1].call_id.as_str(), "call-b");
}

#[test]
fn public_api_exposes_stable_session_and_checkpoint_types() {
    let session_id = SessionId::new("public-session").unwrap();
    let checkpoint = SessionCheckpoint::empty(session_id.clone());
    checkpoint.validate().unwrap();
    let _store = InMemoryCheckpointStore::new();

    let request = ModelRequest::new(
        session_id,
        RunId::new("public-session/run/1"),
        1,
        vec![InputMessage::text("user", "hello").into()],
        Vec::new(),
    );
    assert_eq!(request.request_index, 1);

    // 未投递输入留在会话信箱里，没有需要调用方确认的交接对象。
    let queued = MailboxInput::when_idle(vec![InputMessage::text("user", "later").into()]);
    assert_eq!(queued.items.len(), 1);

    // 编译期确认句柄可由公开路径命名，并能跨 Tokio task 移动。
    fn assert_send<T: Send>() {}
    assert_send::<RunHandle>();
}

/// `Compaction` 实现方必须能认出哪些记录不许动。
///
/// 内核要求「耐久事实逐项原样保留」，就必须给出识别手段——否则实现方只能去
/// 硬编码 `kind` 字符串，而内核加一种回执时那份硬编码不会有任何提示。
#[test]
fn compaction_implementors_can_identify_protected_records() {
    use midturn::{durable_fact_kind, ContentPart, DurableFactKind, InputMessage, TranscriptItem};

    // 普通输入不受保护，可以自由概括。
    let ordinary = TranscriptItem::from(InputMessage::text("user", "这句可以压缩"));
    assert_eq!(durable_fact_kind(&ordinary), None);

    // 内核会产生的每一种回执都必须被认出来。少登记一种，这里就会失败。
    for kind in DurableFactKind::ALL {
        let receipt = TranscriptItem::from(InputMessage::new(
            "user".to_owned(),
            vec![ContentPart::json(serde_json::json!({
                "kind": kind.as_wire(),
                "detail": "任意载荷",
            }))],
        ));
        assert_eq!(
            durable_fact_kind(&receipt),
            Some(*kind),
            "{} 没有被识别为受保护记录",
            kind.as_wire()
        );
    }

    // 字符串与变体一一对应，不能有两个变体撞同一个线路值。
    let mut wires: Vec<&str> = DurableFactKind::ALL.iter().map(|k| k.as_wire()).collect();
    wires.sort_unstable();
    let count = wires.len();
    wires.dedup();
    assert_eq!(wires.len(), count, "线路值必须互不相同");
}
