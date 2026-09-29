//! 图片的真实 wire 结构、错误边界与持久化后的重放；不调用付费 API。

use serde_json::{json, Value};

use super::http::WireCodec;
use crate::{
    ContentPart, InputMessage, ModelOutput, ModelRequest, ToolCall, ToolCallId, ToolResult,
    ToolResultBatch, TranscriptItem,
};

const PNG: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";
const URL: &str = "https://example.com/image.png";

fn codecs() -> Vec<(&'static str, Box<dyn WireCodec>)> {
    vec![
        #[cfg(feature = "openai")]
        (
            "chat",
            Box::new(super::openai::chat::OpenAiChatCodec::new(Default::default())),
        ),
        #[cfg(feature = "openai")]
        (
            "responses",
            Box::new(super::openai::responses::OpenAiResponsesCodec::new(
                Default::default(),
            )),
        ),
        #[cfg(feature = "anthropic")]
        (
            "anthropic",
            Box::new(super::anthropic::messages::AnthropicMessagesCodec::new(
                Default::default(),
            )),
        ),
    ]
}

fn parts() -> Vec<ContentPart> {
    vec![
        ContentPart::text("before"),
        ContentPart::image_url(URL),
        ContentPart::json(json!({"page": 1})),
        ContentPart::image_base64("image/png", PNG),
        ContentPart::text("after"),
    ]
}

fn expected_parts(protocol: &str) -> Value {
    match protocol {
        "chat" => json!([
            {"type": "text", "text": "before"},
            {"type": "image_url", "image_url": {"url": URL}},
            {"type": "text", "text": "{\"page\":1}"},
            {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{PNG}")}},
            {"type": "text", "text": "after"},
        ]),
        "responses" => json!([
            {"type": "input_text", "text": "before"},
            {"type": "input_image", "image_url": URL},
            {"type": "input_text", "text": "{\"page\":1}"},
            {"type": "input_image", "image_url": format!("data:image/png;base64,{PNG}")},
            {"type": "input_text", "text": "after"},
        ]),
        "anthropic" => json!([
            {"type": "text", "text": "before"},
            {"type": "image", "source": {"type": "url", "url": URL}},
            {"type": "text", "text": "{\"page\":1}"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}},
            {"type": "text", "text": "after"},
        ]),
        _ => unreachable!(),
    }
}

#[test]
fn image_inputs_keep_order_and_survive_checkpoint_replay() {
    use crate::persistence::{SessionCheckpoint, SessionId};
    let mut checkpoint = SessionCheckpoint::empty(SessionId::new("images").unwrap());
    checkpoint.conversation = vec![
        InputMessage::new("user", parts()).into(),
        TranscriptItem::ModelOutput(ModelOutput::text("seen")),
        InputMessage::text("user", "compare again").into(),
    ]
    .into();
    let restored: SessionCheckpoint =
        serde_json::from_str(&serde_json::to_string(&checkpoint).unwrap()).unwrap();
    restored.validate().unwrap();
    assert_eq!(restored, checkpoint);
    for (name, codec) in codecs() {
        let body = codec
            .encode_request(
                &ModelRequest::test(restored.conversation.as_ref().clone(), vec![]),
                "vision",
            )
            .unwrap();
        let messages = if name == "responses" {
            &body["input"]
        } else {
            &body["messages"]
        };
        assert_eq!(messages[0]["content"], expected_parts(name), "{name}");
        assert_eq!(messages.as_array().unwrap().len(), 3);
    }
}

#[test]
fn images_are_allowed_without_text() {
    for (name, codec) in codecs() {
        let body = codec
            .encode_request(
                &ModelRequest::test(
                    vec![InputMessage::new("user", vec![ContentPart::image_url(URL)]).into()],
                    vec![],
                ),
                "vision",
            )
            .unwrap();
        let messages = if name == "responses" {
            &body["input"]
        } else {
            &body["messages"]
        };
        assert_eq!(
            messages[0]["content"].as_array().unwrap().len(),
            1,
            "{name}"
        );
    }
}

#[test]
fn image_tool_results_keep_call_identity_and_batch_order() {
    let transcript = vec![
        InputMessage::text("user", "read images").into(),
        TranscriptItem::ModelOutput(ModelOutput::calls(vec![
            ToolCall::new("image-call", "read_image", json!({})),
            ToolCall::new("text-call", "read_text", json!({})),
        ])),
        TranscriptItem::ToolResults(ToolResultBatch {
            results: vec![
                ToolResult::success(ToolCallId::new("image-call"), parts()),
                ToolResult::failure(ToolCallId::new("text-call"), "missing"),
            ],
        }),
        InputMessage::text("user", "explain").into(),
    ];
    crate::validate_transcript(&transcript).unwrap();
    let replay: Vec<TranscriptItem> =
        serde_json::from_value(serde_json::to_value(&transcript).unwrap()).unwrap();
    assert_eq!(replay, transcript);
    for (name, codec) in codecs() {
        let result = codec.encode_request(&ModelRequest::test(replay.clone(), vec![]), "vision");
        match name {
            "chat" => assert_eq!(
                result.unwrap_err().summary,
                "openai_chat_tool_result_images_unsupported:use_responses_or_anthropic"
            ),
            "responses" => {
                let body = result.unwrap();
                assert_eq!(body["input"][3]["call_id"], "image-call");
                assert_eq!(body["input"][3]["output"], expected_parts(name));
                assert_eq!(body["input"][4]["call_id"], "text-call");
                assert_eq!(body["input"][4]["output"], "missing");
                assert_eq!(body["input"][5]["content"], "explain");
            }
            "anthropic" => {
                let body = result.unwrap();
                let blocks = &body["messages"][2]["content"];
                assert_eq!(blocks[0]["tool_use_id"], "image-call");
                assert_eq!(blocks[0]["content"], expected_parts(name));
                assert_eq!(blocks[0]["is_error"], false);
                assert_eq!(blocks[1]["tool_use_id"], "text-call");
                assert_eq!(blocks[1]["is_error"], true);
                assert_eq!(blocks[2]["text"], "explain");
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn invalid_image_sources_fail_without_echoing_image_data() {
    let cases = [
        (
            ContentPart::image_url("file:///private/image.png"),
            "image_url_requires_http_or_https",
        ),
        (
            ContentPart::image_url("not-a-url-secret"),
            "image_invalid_url",
        ),
        (
            ContentPart::image_url(format!("data:image/png;base64,{PNG}")),
            "image_url_requires_http_or_https",
        ),
        (
            ContentPart::image_base64("image/svg+xml", PNG),
            "image_unsupported_media_type",
        ),
        (
            ContentPart::image_base64("image/png", ""),
            "image_invalid_base64",
        ),
        (
            ContentPart::image_base64("image/png", "not-base64-secret"),
            "image_invalid_base64",
        ),
        (
            ContentPart::image_base64("image/png", "data:image/png;base64,YQ=="),
            "image_invalid_base64",
        ),
    ];
    for (name, codec) in codecs() {
        for (part, error) in &cases {
            let request = ModelRequest::test(
                vec![InputMessage::new("user", vec![part.clone()]).into()],
                vec![],
            );
            assert_eq!(
                codec
                    .encode_request(&request, "vision")
                    .unwrap_err()
                    .summary,
                *error,
                "{name}"
            );
        }
    }
}

#[test]
fn unsupported_image_roles_and_model_outputs_fail_instead_of_stringifying() {
    for (name, codec) in codecs() {
        for role in ["assistant", "system", "developer"] {
            // Responses 使用统一的输入内容 schema；具体模型仍可能限制 system/developer 图片。
            if name == "responses" && role != "assistant" {
                continue;
            }
            let request = ModelRequest::test(vec![InputMessage::new(role, parts()).into()], vec![]);
            assert!(
                codec.encode_request(&request, "vision").is_err(),
                "{name}/{role}"
            );
        }
        let request = ModelRequest::test(
            vec![TranscriptItem::ModelOutput(ModelOutput {
                content: parts(),
                ..Default::default()
            })],
            vec![],
        );
        assert!(codec.encode_request(&request, "vision").is_err(), "{name}");
    }
}

#[test]
fn image_role_validation_uses_the_mapped_role() {
    #[cfg(feature = "openai")]
    let codec: Box<dyn WireCodec> = Box::new(super::openai::chat::OpenAiChatCodec::new(
        super::openai::chat::RequestOptions::default().with_role_mapping("operator", "user"),
    ));
    #[cfg(not(feature = "openai"))]
    let codec: Box<dyn WireCodec> =
        Box::new(super::anthropic::messages::AnthropicMessagesCodec::new(
            super::anthropic::messages::RequestOptions::default()
                .with_role_mapping("operator", "user"),
        ));
    let request = ModelRequest::test(vec![InputMessage::new("operator", parts()).into()], vec![]);
    assert!(codec.encode_request(&request, "vision").is_ok());
}
