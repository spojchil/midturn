use midturn::persistence::{SessionCheckpoint, SessionId};
use midturn::{
    ContentPart, ImageSource, InputMessage, ModelOutput, ModelResponse, RunId, ToolCall,
    ToolCallId, ToolResult, ToolResultBatch, TranscriptItem, Turn, TurnStep,
};
use serde_json::json;

fn image() -> ContentPart {
    ContentPart::image_base64("image/png", "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC")
}

fn image_checkpoint() -> SessionCheckpoint {
    let mut checkpoint = SessionCheckpoint::empty(SessionId::new("images-public").unwrap());
    checkpoint.conversation = vec![
        InputMessage::new("user", vec![ContentPart::text("describe"), image()]).into(),
        TranscriptItem::ModelOutput(ModelOutput::calls(vec![ToolCall::new(
            "read",
            "read_image",
            json!({}),
        )])),
        TranscriptItem::ToolResults(ToolResultBatch {
            results: vec![ToolResult::success(ToolCallId::new("read"), vec![image()])],
        }),
    ]
    .into();
    checkpoint
}

#[test]
fn image_public_api_and_checkpoint_format_round_trip() {
    let part = ContentPart::Image {
        source: ImageSource::Url {
            url: "https://example.com/a.png".into(),
        },
    };
    assert_eq!(part, ContentPart::image_url("https://example.com/a.png"));
    assert_eq!(
        serde_json::to_value(&part).unwrap(),
        json!({
            "type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}
        })
    );
    let checkpoint = image_checkpoint();
    let decoded: SessionCheckpoint =
        serde_json::from_slice(&serde_json::to_vec(&checkpoint).unwrap()).unwrap();
    decoded.validate().unwrap();
    assert_eq!(checkpoint, decoded);
    // 新增变体不改变旧文本的持久化格式。
    assert_eq!(
        serde_json::from_value::<ContentPart>(json!({"type": "text", "text": "old"})).unwrap(),
        ContentPart::text("old")
    );
    let output = ModelOutput {
        content: vec![ContentPart::text("visible"), image()],
        ..Default::default()
    };
    assert_eq!(output.text_content(), "visible");
}

#[test]
fn turn_carries_images_through_input_and_tool_boundaries() {
    let mut turn = Turn::new(RunId::new("images/run/1"), vec![]);
    assert!(matches!(
        turn.next_step().unwrap(),
        TurnStep::RequestBoundary { .. }
    ));
    turn.resume_boundary(vec![InputMessage::new("user", vec![image()]).into()])
        .unwrap();
    assert!(matches!(
        turn.next_step().unwrap(),
        TurnStep::CallModel { .. }
    ));
    turn.model_response(ModelResponse {
        output: ModelOutput::calls(vec![ToolCall::new("read", "read_image", json!({}))]),
        ..Default::default()
    })
    .unwrap();
    assert!(matches!(
        turn.next_step().unwrap(),
        TurnStep::DispatchTools { .. }
    ));
    turn.tool_results(ToolResultBatch {
        results: vec![ToolResult::success(ToolCallId::new("read"), vec![image()])],
    })
    .unwrap();
    assert!(matches!(
        turn.next_step().unwrap(),
        TurnStep::RequestBoundary { .. }
    ));
    turn.resume_boundary(vec![]).unwrap();
    assert!(matches!(
        turn.next_step().unwrap(),
        TurnStep::CallModel { .. }
    ));
    let TranscriptItem::Input(input) = &turn.transcript()[0] else {
        panic!("missing input")
    };
    assert_eq!(input.content, vec![image()]);
    let TranscriptItem::ToolResults(results) = &turn.transcript()[2] else {
        panic!("missing result")
    };
    assert_eq!(results.results[0].content, vec![image()]);
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn sqlite_preserves_image_inputs_and_tool_results() {
    use midturn::persistence::{CheckpointStore, SqliteCheckpointStore};
    let store = SqliteCheckpointStore::open_in_memory().await.unwrap();
    let checkpoint = image_checkpoint();
    store
        .compare_and_swap(None, checkpoint.clone())
        .await
        .unwrap();
    let restored = store.load(&checkpoint.session_id).await.unwrap().unwrap();
    assert_eq!(restored.checkpoint, checkpoint);
}
