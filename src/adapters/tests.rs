//! 只验证 wire 投影，不访问环境变量或网络。

use serde_json::json;

use crate::{
    InputMessage, ModelRequest, ModelStreamEvent, ToolCallId, ToolDefinition, ToolResult,
    ToolResultBatch, TranscriptItem,
};

#[cfg(feature = "anthropic")]
use super::anthropic::messages::{AnthropicMessagesCodec, RequestOptions as AnthropicOptions};
use super::http::{SseEvent, WireCodec};
#[cfg(feature = "openai")]
use super::openai::{
    chat::{OpenAiChatCodec, RequestOptions as OpenAiChatOptions},
    responses::{OpenAiResponsesCodec, RequestOptions as OpenAiResponsesOptions},
};

#[cfg(feature = "openai")]
#[test]
fn chat_uses_tool_call_id_as_the_canonical_id() {
    let codec = OpenAiChatCodec::new(OpenAiChatOptions::default());
    let response = codec
        .decode_response(json!({
            "id": "chat-response",
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_chat_1",
                        "type": "function",
                        "function": {"name": "add", "arguments": "{\"left\":2,\"right\":3}"}
                    }],
                    "compatible_endpoint_extension": {"opaque": true}
                }
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
        }))
        .unwrap();

    assert_eq!(response.output.tool_calls[0].id.as_str(), "call_chat_1");
    assert_eq!(response.output.tool_calls[0].arguments["left"], 2);

    let request = ModelRequest::test(
        vec![
            InputMessage::text("developer", "只按协议执行").into(),
            TranscriptItem::ModelOutput(response.output),
        ],
        Vec::new(),
    );
    let body = codec.encode_request(&request, "test-model").unwrap();
    assert_eq!(body["model"], "test-model");
    assert!(body.get("thinking").is_none());
    assert_eq!(body["messages"][0]["role"], "developer");
    assert_eq!(body["messages"][1]["tool_calls"][0]["id"], "call_chat_1");
    assert_eq!(
        body["messages"][1]["compatible_endpoint_extension"]["opaque"],
        true
    );
}

#[cfg(feature = "openai")]
#[test]
fn responses_preserves_output_items_and_correlates_with_call_id() {
    let codec = OpenAiResponsesCodec::new(OpenAiResponsesOptions::default());
    let response = codec
        .decode_response(json!({
            "id": "resp_1",
            "status": "completed",
            "output": [
                {"id": "rs_1", "type": "reasoning", "content": []},
                {
                    "id": "fc_1",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": "call_responses_1",
                    "name": "add",
                    "arguments": "{\"left\":2,\"right\":3}"
                },
                {
                    "id": "fc_2",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": "call_responses_2",
                    "name": "add",
                    "arguments": "{\"left\":5,\"right\":7}"
                }
            ],
            "usage": {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28}
        }))
        .unwrap();
    assert_eq!(
        response.output.tool_calls[0].id.as_str(),
        "call_responses_1"
    );
    assert_eq!(
        response.output.tool_calls[0].provider_data["openai.responses.item_id"],
        "fc_1"
    );

    let request = continuation_request(response.output);
    let body = codec.encode_request(&request, "test-model").unwrap();
    assert_eq!(body["model"], "test-model");
    let input = body["input"].as_array().unwrap();
    assert_eq!(input[1]["id"], "rs_1");
    assert_eq!(input[2]["id"], "fc_1");
    assert_eq!(input[2]["call_id"], "call_responses_1");

    let outputs = input
        .iter()
        .filter(|item| item["type"] == "function_call_output")
        .collect::<Vec<_>>();
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0]["call_id"], "call_responses_1");
    assert_eq!(outputs[1]["call_id"], "call_responses_2");
    assert_eq!(input.last().unwrap()["content"], "结果齐全后简短回答");
}

#[cfg(feature = "anthropic")]
#[test]
fn anthropic_groups_the_complete_result_batch_before_steering_text() {
    let codec = AnthropicMessagesCodec::new(AnthropicOptions::default());
    let response = codec
        .decode_response(json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "stop_reason": "tool_use",
            "content": [
                {"type": "thinking", "thinking": "opaque", "signature": "sig"},
                {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"left": 2, "right": 3}},
                {"type": "tool_use", "id": "toolu_2", "name": "add", "input": {"left": 5, "right": 7}}
            ],
            "usage": {"input_tokens": 30, "output_tokens": 10}
        }))
        .unwrap();
    assert_eq!(response.output.tool_calls[0].id.as_str(), "toolu_1");
    assert_eq!(response.usage.unwrap().total_tokens, Some(40));

    let request = continuation_request(response.output);
    let body = codec.encode_request(&request, "test-model").unwrap();
    assert_eq!(body["model"], "test-model");
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[1]["content"][0]["signature"], "sig");

    let final_user = messages[2]["content"].as_array().unwrap();
    assert_eq!(
        final_user
            .iter()
            .map(|block| block["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["tool_result", "tool_result", "text"]
    );
    assert_eq!(final_user[0]["tool_use_id"], "toolu_1");
    assert_eq!(final_user[1]["tool_use_id"], "toolu_2");
    assert_eq!(final_user[2]["text"], "结果齐全后简短回答");
}

#[cfg(feature = "anthropic")]
#[test]
fn anthropic_extracts_initial_system_context_and_rejects_late_system_input() {
    let codec = AnthropicMessagesCodec::new(AnthropicOptions::default());
    let request = ModelRequest::test(
        vec![
            InputMessage::text("system", "基础约束").into(),
            InputMessage::text("developer", "本轮约束").into(),
            InputMessage::text("user", "开始").into(),
        ],
        Vec::new(),
    );
    let body = codec.encode_request(&request, "test-model").unwrap();
    assert_eq!(body["system"].as_array().unwrap().len(), 2);
    assert_eq!(body["messages"][0]["role"], "user");

    let late = ModelRequest::test(
        vec![
            InputMessage::text("user", "开始").into(),
            InputMessage::text("system", "过晚约束").into(),
        ],
        Vec::new(),
    );
    assert!(codec.encode_request(&late, "test-model").is_err());
}

#[cfg(any(feature = "openai", feature = "anthropic"))]
fn continuation_request(output: crate::ModelOutput) -> ModelRequest {
    let calls = output.tool_calls.clone();
    ModelRequest::test(
        vec![
            InputMessage::text("user", "调用两次 add").into(),
            TranscriptItem::ModelOutput(output),
            TranscriptItem::ToolResults(ToolResultBatch {
                results: calls
                    .iter()
                    .enumerate()
                    .map(|(index, call)| {
                        ToolResult::success_json(
                            ToolCallId::new(call.id.as_str()),
                            json!({"sum": if index == 0 { 5 } else { 12 }}),
                        )
                    })
                    .collect(),
            }),
            InputMessage::text("user", "结果齐全后简短回答").into(),
        ],
        vec![ToolDefinition::new(
            "add",
            json!({"type": "object", "properties": {}}),
        )],
    )
}

#[cfg(feature = "openai")]
#[test]
fn responses_rejects_an_item_id_without_a_call_id() {
    let codec = OpenAiResponsesCodec::new(OpenAiResponsesOptions::default());
    let error = codec
        .decode_response(json!({
            "status": "completed",
            "output": [{
                "id": "fc_only",
                "type": "function_call",
                "name": "add",
                "arguments": "{}"
            }]
        }))
        .unwrap_err();
    assert_eq!(error.kind, crate::AgentErrorKind::Model);
}

#[cfg(feature = "openai")]
#[test]
fn responses_rejects_an_incomplete_response() {
    let codec = OpenAiResponsesCodec::new(OpenAiResponsesOptions::default());
    let error = codec
        .decode_response(json!({
            "status": "incomplete",
            "output": []
        }))
        .unwrap_err();
    assert_eq!(error.summary, "openai_responses_not_completed:incomplete");
}

#[cfg(feature = "openai")]
#[test]
fn chat_rejects_unsuccessful_or_mismatched_finish_reasons() {
    let codec = OpenAiChatCodec::new(OpenAiChatOptions::default());
    for reason in ["length", "content_filter", "insufficient_system_resource"] {
        let error = codec
            .decode_response(json!({
                "choices": [{
                    "finish_reason": reason,
                    "message": {"role": "assistant", "content": "partial"}
                }]
            }))
            .unwrap_err();
        assert_eq!(error.kind, crate::AgentErrorKind::Model);
    }

    let mismatched = codec
        .decode_response(json!({
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_partial",
                        "type": "function",
                        "function": {"name": "add", "arguments": "{}"}
                    }]
                }
            }]
        }))
        .unwrap_err();
    assert_eq!(
        mismatched.summary,
        "openai_chat_tool_calls_without_tool_calls_finish"
    );

    let missing_calls = codec
        .decode_response(json!({
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {"role": "assistant", "content": null}
            }]
        }))
        .unwrap_err();
    assert_eq!(
        missing_calls.summary,
        "openai_chat_tool_calls_finish_without_tool_calls"
    );
}

#[cfg(feature = "anthropic")]
#[test]
fn anthropic_rejects_unsuccessful_or_mismatched_stop_reasons() {
    let codec = AnthropicMessagesCodec::new(AnthropicOptions::default());
    for reason in ["max_tokens", "model_context_window_exceeded"] {
        let error = codec
            .decode_response(json!({
                "id": "msg_partial",
                "stop_reason": reason,
                "content": [{"type": "text", "text": "partial"}]
            }))
            .unwrap_err();
        assert_eq!(error.kind, crate::AgentErrorKind::Model);
    }

    let mismatched = codec
        .decode_response(json!({
            "id": "msg_partial_tool",
            "stop_reason": "end_turn",
            "content": [{
                "type": "tool_use",
                "id": "toolu_partial",
                "name": "add",
                "input": {}
            }]
        }))
        .unwrap_err();
    assert_eq!(
        mismatched.summary,
        "anthropic_messages_tool_calls_without_tool_use_stop"
    );

    let missing_calls = codec
        .decode_response(json!({
            "id": "msg_missing_tool",
            "stop_reason": "tool_use",
            "content": [{"type": "text", "text": "没有工具块"}]
        }))
        .unwrap_err();
    assert_eq!(
        missing_calls.summary,
        "anthropic_messages_tool_use_stop_without_tool_calls"
    );
}

#[cfg(feature = "openai")]
#[test]
fn responses_rejects_an_explicitly_incomplete_function_call_item() {
    let codec = OpenAiResponsesCodec::new(OpenAiResponsesOptions::default());
    let error = codec
        .decode_response(json!({
            "id": "resp_partial_call",
            "status": "completed",
            "output": [{
                "id": "fc_partial",
                "type": "function_call",
                "status": "incomplete",
                "call_id": "call_partial",
                "name": "add",
                "arguments": "{}"
            }]
        }))
        .unwrap_err();
    assert_eq!(
        error.summary,
        "openai_responses_function_call_not_completed"
    );
}

#[cfg(any(feature = "openai", feature = "anthropic"))]
fn sse(value: serde_json::Value) -> SseEvent {
    SseEvent {
        event: "message".to_owned(),
        data: value.to_string(),
        id: "".into(),
    }
}

#[cfg(feature = "openai")]
fn sse_done() -> SseEvent {
    SseEvent {
        event: "message".to_owned(),
        data: "[DONE]".to_owned(),
        id: "".into(),
    }
}

#[cfg(feature = "openai")]
#[test]
fn chat_publishes_a_complete_object_before_a_later_stream_failure() {
    let codec = OpenAiChatCodec::new(OpenAiChatOptions::default());
    let mut decoder = codec.stream_decoder();
    let first = decoder
        .push(sse(json!({
            "id": "chat_stream_1",
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0,
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "add", "arguments": "{\"left\":1"}
                }]},
                "finish_reason": null
            }]
        })))
        .unwrap();
    assert!(first.events.is_empty());
    let complete_first = decoder
        .push(sse(json!({
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0,
                    "function": {"arguments": ",\"right\":2}"}
                }]},
                "finish_reason": null
            }]
        })))
        .unwrap();
    assert!(matches!(
        &complete_first.events[..],
        [ModelStreamEvent::ToolCallReady { slot, call }]
            if slot.get() == 0 && call.id.as_str() == "call_0"
    ));

    let partial_second = decoder
        .push(sse(json!({
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 1,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "add", "arguments": "{\"left\":3"}
                }]},
                "finish_reason": null
            }]
        })))
        .unwrap();
    assert!(partial_second.events.is_empty());
    assert_eq!(
        decoder.finish_eof().unwrap_err().summary,
        "openai_chat_stream_missing_done"
    );
}

#[cfg(feature = "openai")]
#[test]
fn chat_publishes_the_complete_ordered_batch_at_done() {
    let codec = OpenAiChatCodec::new(OpenAiChatOptions::default());
    let mut decoder = codec.stream_decoder();
    let early = decoder
        .push(sse(json!({
            "id": "chat_stream_2",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "先计算。",
                    "reasoning_content": "思考一",
                    "tool_calls": [
                        {"index": 0, "id": "call_0", "type": "function", "function": {"name": "add", "arguments": "{\"x\":1}"}},
                        {"index": 1, "id": "call_1", "type": "function", "function": {"name": "add", "arguments": "{\"x\":2}"}}
                    ]
                },
                "finish_reason": null
            }]
        })))
        .unwrap();
    // 正文与推理各自开段，然后才是两个已经完整的工具调用。
    assert_eq!(early.events.len(), 6);
    assert!(matches!(
        &early.events[0],
        ModelStreamEvent::TextStart { .. }
    ));
    assert!(matches!(
        &early.events[1],
        ModelStreamEvent::TextDelta { delta, .. } if delta == "先计算。"
    ));
    assert!(matches!(
        &early.events[2],
        ModelStreamEvent::ReasoningStart { .. }
    ));
    assert!(matches!(
        &early.events[3],
        ModelStreamEvent::ReasoningDelta { delta, .. } if delta == "思考一"
    ));
    assert!(matches!(
        &early.events[4],
        ModelStreamEvent::ToolCallReady { slot, call }
            if slot.get() == 0 && call.id.as_str() == "call_0"
    ));
    assert!(matches!(
        &early.events[5],
        ModelStreamEvent::ToolCallReady { slot, call }
            if slot.get() == 1 && call.id.as_str() == "call_1"
    ));
    decoder
        .push(sse(json!({
            "id": "chat_stream_2",
            "choices": [{
                "index": 0,
                "delta": {"reasoning_content": "思考二"},
                "finish_reason": "tool_calls"
            }]
        })))
        .unwrap();
    let done = decoder.push(sse_done()).unwrap();
    let response = done.response.as_ref().unwrap();
    assert_eq!(
        response.output.provider_data["openai.chat.message"]["reasoning_content"],
        "思考一思考二"
    );
    assert_eq!(
        response.output.provider_data["openai.chat.message"]["role"],
        "assistant"
    );
    // Chat 没有段结束信号，两段都在流终止时由适配器封口，然后才是工具数组封口。
    assert_eq!(
        done.events,
        vec![
            ModelStreamEvent::ReasoningEnd { part_index: 1 },
            ModelStreamEvent::TextEnd { part_index: 0 },
            ModelStreamEvent::ToolCallsSealed { call_count: 2 },
        ]
    );

    let continuation = ModelRequest::test(
        vec![TranscriptItem::ModelOutput(response.output.clone())],
        Vec::new(),
    );
    let replay = codec.encode_request(&continuation, "test-model").unwrap();
    assert_eq!(replay["messages"][0]["reasoning_content"], "思考一思考二");
}

#[cfg(feature = "openai")]
#[test]
fn chat_detects_a_complex_tool_object_across_arbitrary_character_deltas() {
    let codec = OpenAiChatCodec::new(OpenAiChatOptions::default());
    let mut decoder = codec.stream_decoder();
    decoder
        .push(sse(json!({
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0,
                    "id": "call_complex",
                    "type": "function",
                    "function": {"name": "inspect"}
                }]},
                "finish_reason": null
            }]
        })))
        .unwrap();

    let raw =
        r#"{"nested":[{"text":"quote: \" slash: \\"}],"unicode":"你好😀","escaped":"\u4f60"}"#;
    let characters = raw.chars().collect::<Vec<_>>();
    for (position, character) in characters.iter().enumerate() {
        let decoded = decoder
            .push(sse(json!({
                "choices": [{
                    "index": 0,
                    "delta": {"tool_calls": [{
                        "index": 0,
                        "function": {"arguments": character.to_string()}
                    }]},
                    "finish_reason": null
                }]
            })))
            .unwrap();
        if position + 1 == characters.len() {
            assert!(matches!(
                &decoded.events[..],
                [ModelStreamEvent::ToolCallReady { slot, call }]
                    if slot.get() == 0
                        && call.id.as_str() == "call_complex"
                        && call.arguments["nested"][0]["text"] == "quote: \" slash: \\"
                        && call.arguments["unicode"] == "你好😀"
                        && call.arguments["escaped"] == "你"
            ));
        } else {
            assert!(decoded.events.is_empty(), "position={position}");
        }
    }
}

#[cfg(feature = "openai")]
#[test]
fn chat_rejects_substantive_arguments_after_an_early_ready() {
    let codec = OpenAiChatCodec::new(OpenAiChatOptions::default());
    let mut decoder = codec.stream_decoder();
    let ready = decoder
        .push(sse(json!({
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0,
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "add", "arguments": "{}"}
                }]},
                "finish_reason": null
            }]
        })))
        .unwrap();
    assert!(matches!(
        &ready.events[..],
        [ModelStreamEvent::ToolCallReady { .. }]
    ));

    let whitespace = decoder
        .push(sse(json!({
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0,
                    "id": "call_0",
                    "function": {"name": "add", "arguments": " \r\n\t"}
                }]},
                "finish_reason": null
            }]
        })))
        .unwrap();
    assert!(whitespace.events.is_empty());

    let error = decoder
        .push(sse(json!({
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0,
                    "function": {"arguments": "x"}
                }]},
                "finish_reason": null
            }]
        })))
        .unwrap_err();
    assert_eq!(
        error.summary,
        "openai_chat_stream_tool_arguments_after_complete"
    );

    let mut changed_identity = codec.stream_decoder();
    changed_identity
        .push(sse(json!({
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0,
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "add", "arguments": "{}"}
                }]},
                "finish_reason": null
            }]
        })))
        .unwrap();
    let error = changed_identity
        .push(sse(json!({
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0,
                    "id": "call_changed",
                    "function": {"name": "add", "arguments": " "}
                }]},
                "finish_reason": null
            }]
        })))
        .unwrap_err();
    assert_eq!(error.summary, "openai_chat_stream_tool_id_changed");
}

#[cfg(feature = "openai")]
#[test]
fn chat_stream_requires_finish_reason_and_calls_to_match_both_ways() {
    let codec = OpenAiChatCodec::new(OpenAiChatOptions::default());
    let mut missing_calls = codec.stream_decoder();
    missing_calls
        .push(sse(json!({
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": "tool_calls"
            }]
        })))
        .unwrap();
    assert_eq!(
        missing_calls.push(sse_done()).unwrap_err().summary,
        "openai_chat_stream_tool_calls_finish_without_tool_calls"
    );

    let mut unexpected_calls = codec.stream_decoder();
    unexpected_calls
        .push(sse(json!({
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0,
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "add", "arguments": "{}"}
                }]},
                "finish_reason": "stop"
            }]
        })))
        .unwrap();
    assert_eq!(
        unexpected_calls.push(sse_done()).unwrap_err().summary,
        "openai_chat_stream_tool_calls_without_tool_calls_finish"
    );
}

#[cfg(feature = "openai")]
#[test]
fn chat_stream_accepts_choice_scoped_usage_and_rejects_conflicts() {
    let codec = OpenAiChatCodec::new(OpenAiChatOptions::default());
    let mut decoder = codec.stream_decoder();
    decoder
        .push(sse(json!({
            "choices": [{
                "index": 0,
                "delta": {"content": "ok"},
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
            }]
        })))
        .unwrap();
    let response = decoder.push(sse_done()).unwrap().response.unwrap();
    let usage = response.usage.unwrap();
    assert_eq!(usage.input_tokens, Some(7));
    assert_eq!(usage.output_tokens, Some(3));
    assert_eq!(usage.total_tokens, Some(10));

    let mut conflicting = codec.stream_decoder();
    let error = conflicting
        .push(sse(json!({
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            "choices": [{
                "index": 0,
                "delta": {"content": "ok"},
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}
            }]
        })))
        .unwrap_err();
    assert_eq!(error.summary, "openai_chat_stream_usage_changed");
}

#[cfg(feature = "openai")]
#[test]
fn chat_requires_unknown_non_string_delta_fields_to_remain_stable() {
    let codec = OpenAiChatCodec::new(OpenAiChatOptions::default());
    let mut decoder = codec.stream_decoder();
    for value in [
        json!({"choices": [{"index": 0, "delta": {"compatible_flag": 1, "ignored": null}, "finish_reason": null}]}),
        json!({"choices": [{"index": 0, "delta": {"compatible_flag": 1}, "finish_reason": null}]}),
    ] {
        decoder.push(sse(value)).unwrap();
    }
    let error = decoder
        .push(sse(json!({
            "choices": [{"index": 0, "delta": {"compatible_flag": 2}, "finish_reason": null}]
        })))
        .unwrap_err();
    assert_eq!(
        error.summary,
        "openai_chat_stream_message_extension_changed_type_or_value"
    );
}

#[cfg(feature = "openai")]
#[test]
fn responses_publishes_item_done_but_does_not_seal_a_truncated_response() {
    let codec = OpenAiResponsesCodec::new(OpenAiResponsesOptions::default());
    let mut decoder = codec.stream_decoder();
    decoder
        .push(sse(json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "id": "fc_0", "call_id": "call_0", "name": "add", "arguments": ""}
        })))
        .unwrap();
    let ready = decoder
        .push(sse(json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "function_call", "status": "completed", "id": "fc_0", "call_id": "call_0", "name": "add", "arguments": "{\"x\":1}"}
        })))
        .unwrap();
    assert!(matches!(
        &ready.events[..],
        [ModelStreamEvent::ToolCallReady { slot, call }]
            if slot.get() == 0 && call.id.as_str() == "call_0"
    ));
    decoder
        .push(sse(json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 1,
            "item_id": "fc_1",
            "delta": "{\"x\":"
        })))
        .unwrap();
    assert_eq!(
        decoder.finish_eof().unwrap_err().summary,
        "openai_responses_stream_missing_terminal"
    );
}

#[cfg(feature = "openai")]
#[test]
fn responses_seals_only_from_response_completed() {
    let codec = OpenAiResponsesCodec::new(OpenAiResponsesOptions::default());
    let mut decoder = codec.stream_decoder();
    decoder
        .push(sse(json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "id": "fc_0", "call_id": "call_0", "name": "add", "arguments": ""}
        })))
        .unwrap();
    decoder
        .push(sse(json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "function_call", "status": "completed", "id": "fc_0", "call_id": "call_0", "name": "add", "arguments": "{\"x\":1}"}
        })))
        .unwrap();
    let completed = decoder
        .push(sse(json!({
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "status": "completed",
                "output": [{"type": "function_call", "status": "completed", "id": "fc_0", "call_id": "call_0", "name": "add", "arguments": "{\"x\":1}"}]
            }
        })))
        .unwrap();
    assert_eq!(
        completed.events,
        vec![ModelStreamEvent::ToolCallsSealed { call_count: 1 }]
    );
    assert!(completed.response.is_some());
}

#[cfg(feature = "openai")]
#[test]
fn responses_completed_event_does_not_backfill_an_incomplete_function_item() {
    let codec = OpenAiResponsesCodec::new(OpenAiResponsesOptions::default());
    let mut decoder = codec.stream_decoder();
    let error = decoder
        .push(sse(json!({
            "type": "response.completed",
            "response": {
                "id": "resp_partial_call",
                "status": "completed",
                "output": [{
                    "id": "fc_partial",
                    "type": "function_call",
                    "status": "incomplete",
                    "call_id": "call_partial",
                    "name": "add",
                    "arguments": "{}"
                }]
            }
        })))
        .unwrap_err();
    assert_eq!(
        error.summary,
        "openai_responses_function_call_not_completed"
    );
}

#[cfg(feature = "anthropic")]
#[test]
fn anthropic_publishes_a_stopped_tool_block_before_a_later_stream_failure() {
    let codec = AnthropicMessagesCodec::new(AnthropicOptions::default());
    let mut decoder = codec.stream_decoder();
    decoder
        .push(sse(json!({
            "type": "message_start",
            "message": {"id": "msg_stream", "type": "message", "role": "assistant", "content": [], "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 5, "output_tokens": 0}}
        })))
        .unwrap();
    decoder
        .push(sse(json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_0", "name": "add", "input": {}}
        })))
        .unwrap();
    decoder
        .push(sse(json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": "{\"x\":1}"}
        })))
        .unwrap();
    let ready = decoder
        .push(sse(json!({"type": "content_block_stop", "index": 0})))
        .unwrap();
    assert!(matches!(
        &ready.events[..],
        [ModelStreamEvent::ToolCallReady { slot, call }]
            if slot.get() == 0 && call.id.as_str() == "toolu_0"
    ));

    decoder
        .push(sse(json!({
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {}}
        })))
        .unwrap();
    decoder
        .push(sse(json!({
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": "{\"x\":"}
        })))
        .unwrap();
    assert_eq!(
        decoder.finish_eof().unwrap_err().summary,
        "anthropic_messages_stream_missing_message_stop"
    );
}

#[cfg(feature = "anthropic")]
#[test]
fn anthropic_seals_at_message_stop_after_the_tool_block_is_closed() {
    let codec = AnthropicMessagesCodec::new(AnthropicOptions::default());
    let mut decoder = codec.stream_decoder();
    for value in [
        json!({
            "type": "message_start",
            "message": {"id": "msg_stream", "type": "message", "role": "assistant", "content": [], "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 5, "output_tokens": 0}}
        }),
        json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_0", "name": "add", "input": {}}
        }),
        json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": "{\"x\":1}"}
        }),
        json!({"type": "content_block_stop", "index": 0}),
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": null},
            "usage": {"output_tokens": 8}
        }),
    ] {
        decoder.push(sse(value)).unwrap();
    }
    let completed = decoder.push(sse(json!({"type": "message_stop"}))).unwrap();
    assert_eq!(
        completed.events,
        vec![ModelStreamEvent::ToolCallsSealed { call_count: 1 }]
    );
    assert!(completed.response.is_some());
}

#[cfg(feature = "anthropic")]
#[test]
fn anthropic_stream_publishes_reasoning_and_text_as_separate_segments() {
    let codec = AnthropicMessagesCodec::new(AnthropicOptions::default());
    let mut decoder = codec.stream_decoder();
    decoder
        .push(sse(json!({
            "type": "message_start",
            "message": {"id": "msg_think", "type": "message", "role": "assistant", "content": [], "stop_reason": null, "stop_sequence": null}
        })))
        .unwrap();

    // 先推理：块 0。
    let opened = decoder
        .push(sse(json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""}
        })))
        .unwrap();
    assert_eq!(
        opened.events,
        vec![ModelStreamEvent::ReasoningStart { part_index: 0 }]
    );
    let thinking = decoder
        .push(sse(json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "先想一下"}
        })))
        .unwrap();
    assert_eq!(
        thinking.events,
        vec![ModelStreamEvent::ReasoningDelta {
            part_index: 0,
            delta: "先想一下".to_owned()
        }]
    );
    let closed = decoder
        .push(sse(json!({"type": "content_block_stop", "index": 0})))
        .unwrap();
    assert_eq!(
        closed.events,
        vec![ModelStreamEvent::ReasoningEnd { part_index: 0 }]
    );

    // 再回答：块 1。推理与正文用不同的 part_index，消费方可以分开渲染。
    let opened = decoder
        .push(sse(json!({
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""}
        })))
        .unwrap();
    assert_eq!(
        opened.events,
        vec![ModelStreamEvent::TextStart { part_index: 1 }]
    );
    let text = decoder
        .push(sse(json!({
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "答案"}
        })))
        .unwrap();
    assert_eq!(
        text.events,
        vec![ModelStreamEvent::TextDelta {
            part_index: 1,
            delta: "答案".to_owned()
        }]
    );
    let closed = decoder
        .push(sse(json!({"type": "content_block_stop", "index": 1})))
        .unwrap();
    assert_eq!(
        closed.events,
        vec![ModelStreamEvent::TextEnd { part_index: 1 }]
    );

    decoder
        .push(sse(json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null}
        })))
        .unwrap();
    let done = decoder.push(sse(json!({"type": "message_stop"}))).unwrap();
    let response = done.response.expect("message_stop 应产出完整响应");
    // 推理不进规范正文，但原始块要留在 provider_data 里供回放。
    assert_eq!(response.output.text_content(), "答案");
    let raw = response.output.provider_data["anthropic.messages.content"]
        .as_array()
        .unwrap();
    assert_eq!(raw[0]["type"], "thinking");
    assert_eq!(raw[0]["thinking"], "先想一下");
}

#[cfg(feature = "anthropic")]
#[test]
fn anthropic_stream_requires_stop_reason_and_calls_to_match_both_ways() {
    let codec = AnthropicMessagesCodec::new(AnthropicOptions::default());
    let mut missing_calls = codec.stream_decoder();
    for value in [
        json!({
            "type": "message_start",
            "message": {"id": "msg_missing", "type": "message", "role": "assistant", "content": [], "stop_reason": null, "stop_sequence": null}
        }),
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": null}
        }),
    ] {
        missing_calls.push(sse(value)).unwrap();
    }
    let error = missing_calls
        .push(sse(json!({"type": "message_stop"})))
        .unwrap_err();
    assert_eq!(
        error.summary,
        "anthropic_messages_stream_tool_use_stop_without_tool_calls"
    );

    let mut unexpected_calls = codec.stream_decoder();
    for value in [
        json!({
            "type": "message_start",
            "message": {"id": "msg_unexpected", "type": "message", "role": "assistant", "content": [], "stop_reason": null, "stop_sequence": null}
        }),
        json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_0", "name": "add", "input": {}}
        }),
        json!({"type": "content_block_stop", "index": 0}),
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null}
        }),
    ] {
        unexpected_calls.push(sse(value)).unwrap();
    }
    let error = unexpected_calls
        .push(sse(json!({"type": "message_stop"})))
        .unwrap_err();
    assert_eq!(
        error.summary,
        "anthropic_messages_stream_tool_calls_without_tool_use_stop"
    );
}

#[cfg(any(feature = "openai", feature = "anthropic"))]
#[test]
fn custom_canonical_roles_require_an_explicit_wire_mapping() {
    let request = || {
        ModelRequest::test(
            vec![InputMessage::text("interrupted_tool_receipt", "已执行").into()],
            Vec::new(),
        )
    };

    #[cfg(feature = "openai")]
    {
        let plain = OpenAiChatCodec::new(OpenAiChatOptions::default());
        assert!(plain.encode_request(&request(), "test-model").is_err());
        let mapped = OpenAiChatCodec::new(
            OpenAiChatOptions::default().with_role_mapping("interrupted_tool_receipt", "user"),
        );
        assert_eq!(
            mapped.encode_request(&request(), "test-model").unwrap()["messages"][0]["role"],
            "user"
        );

        let responses = OpenAiResponsesCodec::new(
            OpenAiResponsesOptions::default().with_role_mapping("interrupted_tool_receipt", "user"),
        );
        assert_eq!(
            responses.encode_request(&request(), "test-model").unwrap()["input"][0]["role"],
            "user"
        );
    }

    #[cfg(feature = "anthropic")]
    {
        let mapped = AnthropicMessagesCodec::new(
            AnthropicOptions::default().with_role_mapping("interrupted_tool_receipt", "user"),
        );
        assert_eq!(
            mapped.encode_request(&request(), "test-model").unwrap()["messages"][0]["role"],
            "user"
        );
    }
}

#[cfg(feature = "openai")]
#[test]
fn refusals_are_exposed_as_model_text_in_chat_and_responses() {
    let chat = OpenAiChatCodec::new(OpenAiChatOptions::default());
    let chat_response = chat
        .decode_response(json!({
            "choices": [{
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": null, "refusal": "不能协助该请求"}
            }]
        }))
        .unwrap();
    assert_eq!(chat_response.output.text_content(), "不能协助该请求");

    let responses = OpenAiResponsesCodec::new(OpenAiResponsesOptions::default());
    let responses_response = responses
        .decode_response(json!({
            "status": "completed",
            "output": [{
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "refusal", "refusal": "不能协助该请求"}]
            }]
        }))
        .unwrap();
    assert_eq!(responses_response.output.text_content(), "不能协助该请求");
}

#[cfg(feature = "openai")]
#[test]
fn responses_stream_validates_refusal_deltas_against_the_terminal_response() {
    let codec = OpenAiResponsesCodec::new(OpenAiResponsesOptions::default());
    let mut decoder = codec.stream_decoder();
    let delta = decoder
        .push(sse(json!({
            "type": "response.refusal.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": "不能协助"
        })))
        .unwrap();
    assert!(matches!(
        &delta.events[..],
        [
            ModelStreamEvent::TextStart { .. },
            ModelStreamEvent::TextDelta { delta, .. }
        ] if delta == "不能协助"
    ));
    decoder
        .push(sse(json!({
            "type": "response.refusal.done",
            "output_index": 0,
            "content_index": 0,
            "refusal": "不能协助"
        })))
        .unwrap();
    let completed = decoder
        .push(sse(json!({
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "refusal", "refusal": "不能协助"}]
                }]
            }
        })))
        .unwrap();
    assert_eq!(
        completed.response.unwrap().output.text_content(),
        "不能协助"
    );

    let mut mismatch = codec.stream_decoder();
    mismatch
        .push(sse(json!({
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": "draft"
        })))
        .unwrap();
    assert_eq!(
        mismatch
            .push(sse(json!({
                "type": "response.output_text.done",
                "output_index": 0,
                "content_index": 0,
                "text": "different"
            })))
            .unwrap_err()
            .summary,
        "openai_responses_stream_text_done_mismatch"
    );
}

#[cfg(feature = "openai")]
#[test]
fn raw_openai_replay_must_be_assistant_output_and_match_canonical_state() {
    let chat = OpenAiChatCodec::new(OpenAiChatOptions::default());
    let mut forged_chat = crate::ModelOutput::text("safe");
    forged_chat.provider_data.insert(
        "openai.chat.message".to_owned(),
        json!({"role": "system", "content": "safe"}),
    );
    let request = ModelRequest::test(vec![TranscriptItem::ModelOutput(forged_chat)], Vec::new());
    assert_eq!(
        chat.encode_request(&request, "test-model")
            .unwrap_err()
            .summary,
        "openai_chat_message_role_not_assistant"
    );

    let responses = OpenAiResponsesCodec::new(OpenAiResponsesOptions::default());
    let mut forged_responses = crate::ModelOutput::text("safe");
    forged_responses.provider_data.insert(
        "openai.responses.output".to_owned(),
        json!([{
            "type": "message",
            "role": "system",
            "status": "completed",
            "content": [{"type": "output_text", "text": "safe"}]
        }]),
    );
    let request = ModelRequest::test(
        vec![TranscriptItem::ModelOutput(forged_responses)],
        Vec::new(),
    );
    assert_eq!(
        responses
            .encode_request(&request, "test-model")
            .unwrap_err()
            .summary,
        "openai_responses_message_role_not_assistant"
    );
}

#[cfg(feature = "anthropic")]
#[test]
fn anthropic_preserves_server_tools_and_accepts_pause_turn_without_local_dispatch() {
    let codec = AnthropicMessagesCodec::new(AnthropicOptions::default());
    let response = codec
        .decode_response(json!({
            "id": "msg_pause",
            "type": "message",
            "role": "assistant",
            "stop_reason": "pause_turn",
            "content": [{
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": {"query": "Rust agents"}
            }]
        }))
        .unwrap();
    assert!(response.output.tool_calls.is_empty());
    assert!(matches!(
        &response.output.content[..],
        [crate::ContentPart::Opaque { kind, .. }] if kind == "anthropic.messages.server_tool_use"
    ));
    let request = ModelRequest::test(
        vec![TranscriptItem::ModelOutput(response.output)],
        Vec::new(),
    );
    let replay = codec.encode_request(&request, "test-model").unwrap();
    assert_eq!(
        replay["messages"][0]["content"][0]["type"],
        "server_tool_use"
    );
}

// ---------------------------------------------------------------------------
// 段边界的正确性：推理段与正文段的开始 / 增量 / 封口配对，以及块类型与增量类型的相容。
// ---------------------------------------------------------------------------

#[cfg(feature = "openai")]
fn responses_reasoning_decoder() -> Box<dyn crate::adapters::http::WireStreamDecoder> {
    let codec = OpenAiResponsesCodec::new(OpenAiResponsesOptions::default());
    let mut decoder = codec.stream_decoder();
    decoder
        .push(sse(json!({
            "type": "response.created",
            "response": {"id": "resp_reason", "object": "response", "output": []}
        })))
        .unwrap();
    decoder
}

/// `done` 之后再来增量必须报错，否则 `ReasoningEnd` 就成了一句空话。
#[cfg(feature = "openai")]
#[test]
fn responses_stream_rejects_reasoning_delta_after_done() {
    let mut decoder = responses_reasoning_decoder();
    decoder
        .push(sse(json!({
            "type": "response.reasoning_text.delta",
            "output_index": 0, "content_index": 0, "delta": "想"
        })))
        .unwrap();
    decoder
        .push(sse(json!({
            "type": "response.reasoning_text.done",
            "output_index": 0, "content_index": 0, "text": "想"
        })))
        .unwrap();
    let error = decoder
        .push(sse(json!({
            "type": "response.reasoning_text.delta",
            "output_index": 0, "content_index": 0, "delta": "又想"
        })))
        .expect_err("封口后的增量必须被拒绝");
    assert!(
        error.summary.contains("reasoning_delta_after_done"),
        "{error:?}"
    );
}

/// 重复的 `done` 会发出第二个不成对的 `ReasoningEnd`。
#[cfg(feature = "openai")]
#[test]
fn responses_stream_rejects_duplicate_reasoning_done() {
    let mut decoder = responses_reasoning_decoder();
    decoder
        .push(sse(json!({
            "type": "response.reasoning_text.done",
            "output_index": 0, "content_index": 0, "text": "一次"
        })))
        .unwrap();
    let error = decoder
        .push(sse(json!({
            "type": "response.reasoning_text.done",
            "output_index": 0, "content_index": 0, "text": "一次"
        })))
        .expect_err("重复封口必须被拒绝");
    assert!(
        error.summary.contains("duplicate_reasoning_done"),
        "{error:?}"
    );
}

/// 只有 `done`、没有增量时，整段推理内容不能从观测里凭空消失。
///
/// 推理不进入规范输出，所以这里若只发一对空的开始/结束，内容就再也拿不到了——
/// 这正是它与正文段处理方式不同的原因。
#[cfg(feature = "openai")]
#[test]
fn responses_stream_done_only_reasoning_still_carries_its_text() {
    let mut decoder = responses_reasoning_decoder();
    let decoded = decoder
        .push(sse(json!({
            "type": "response.reasoning_text.done",
            "output_index": 0, "content_index": 0, "text": "一步到位的推理"
        })))
        .unwrap();
    assert_eq!(
        decoded.events,
        vec![
            ModelStreamEvent::ReasoningStart { part_index: 0 },
            ModelStreamEvent::ReasoningDelta {
                part_index: 0,
                delta: "一步到位的推理".to_owned()
            },
            ModelStreamEvent::ReasoningEnd { part_index: 0 },
        ]
    );
}

/// `done` 报的全文与累积增量对不上，说明中间丢过内容。
#[cfg(feature = "openai")]
#[test]
fn responses_stream_rejects_reasoning_done_text_mismatch() {
    let mut decoder = responses_reasoning_decoder();
    decoder
        .push(sse(json!({
            "type": "response.reasoning_text.delta",
            "output_index": 0, "content_index": 0, "delta": "前半"
        })))
        .unwrap();
    let error = decoder
        .push(sse(json!({
            "type": "response.reasoning_text.done",
            "output_index": 0, "content_index": 0, "text": "前半后半"
        })))
        .expect_err("累积值与 done 全文不一致必须被拒绝");
    assert!(
        error.summary.contains("reasoning_done_mismatch"),
        "{error:?}"
    );
}

/// `text` 是官方 schema 的必填字段：缺了就是丢内容，不能当观测缺口放过。
#[cfg(feature = "openai")]
#[test]
fn responses_stream_rejects_reasoning_done_without_text() {
    let mut decoder = responses_reasoning_decoder();
    let error = decoder
        .push(sse(json!({
            "type": "response.reasoning_text.done",
            "output_index": 0, "content_index": 0
        })))
        .expect_err("缺 text 的 reasoning done 必须被拒绝");
    assert!(
        error.summary.contains("reasoning_done_missing_text"),
        "{error:?}"
    );
}

/// 流式 `citations_delta` 必须写进 text 块，否则 `message_stop` 重建的 content 丢引用。
#[cfg(feature = "anthropic")]
#[test]
fn anthropic_stream_accumulates_citations_into_the_text_block() {
    let codec = AnthropicMessagesCodec::new(AnthropicOptions::default());
    let mut decoder = codec.stream_decoder();
    decoder
        .push(sse(json!({
            "type": "message_start",
            "message": {"id": "msg_cite", "type": "message", "role": "assistant", "content": [], "stop_reason": null, "stop_sequence": null}
        })))
        .unwrap();
    decoder
        .push(sse(json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": "", "citations": null}
        })))
        .unwrap();
    decoder
        .push(sse(json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "引用的答案"}
        })))
        .unwrap();
    let cited = decoder
        .push(sse(json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "citations_delta", "citation": {
                "type": "char_location", "cited_text": "原文", "document_index": 0,
                "document_title": "doc", "start_char_index": 0, "end_char_index": 2
            }}
        })))
        .unwrap();
    // 引用不是正文增量：不产生观测事件。
    assert!(cited.events.is_empty());
    decoder
        .push(sse(json!({"type": "content_block_stop", "index": 0})))
        .unwrap();
    decoder
        .push(sse(json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null}
        })))
        .unwrap();
    let done = decoder.push(sse(json!({"type": "message_stop"}))).unwrap();
    let response = done.response.expect("message_stop 应产出完整响应");
    let raw = response.output.provider_data["anthropic.messages.content"]
        .as_array()
        .unwrap();
    assert_eq!(raw[0]["text"], "引用的答案");
    assert_eq!(raw[0]["citations"].as_array().map(Vec::len), Some(1));
    assert_eq!(raw[0]["citations"][0]["cited_text"], "原文");

    // 引用落在推理块上：与 text_delta 一样，不相容就拒绝。
    let codec = AnthropicMessagesCodec::new(AnthropicOptions::default());
    let mut decoder = codec.stream_decoder();
    decoder
        .push(sse(json!({
            "type": "message_start",
            "message": {"id": "msg_cite2", "type": "message", "role": "assistant", "content": [], "stop_reason": null, "stop_sequence": null}
        })))
        .unwrap();
    decoder
        .push(sse(json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""}
        })))
        .unwrap();
    let error = decoder
        .push(sse(json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "citations_delta", "citation": {"type": "char_location"}}
        })))
        .expect_err("citations_delta 只属于 text 块");
    assert!(
        error.summary.contains("delta_type_incompatible_with_block"),
        "{error:?}"
    );
}

/// 推理增量落在正文块上，会发出 `TextStart → ReasoningDelta → TextEnd`。
#[cfg(feature = "anthropic")]
#[test]
fn anthropic_stream_rejects_delta_type_incompatible_with_its_block() {
    let codec = AnthropicMessagesCodec::new(AnthropicOptions::default());
    let mut decoder = codec.stream_decoder();
    decoder
        .push(sse(json!({
            "type": "message_start",
            "message": {"id": "msg_x", "type": "message", "role": "assistant", "content": [], "stop_reason": null, "stop_sequence": null}
        })))
        .unwrap();
    decoder
        .push(sse(json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        })))
        .unwrap();
    let error = decoder
        .push(sse(json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "不该出现在正文块上"}
        })))
        .expect_err("推理增量不能落在正文块上");
    assert!(
        error.summary.contains("delta_type_incompatible_with_block"),
        "{error:?}"
    );
}

/// 反向：正文增量落在推理块上同样不成立。
#[cfg(feature = "anthropic")]
#[test]
fn anthropic_stream_rejects_text_delta_on_a_reasoning_block() {
    let codec = AnthropicMessagesCodec::new(AnthropicOptions::default());
    let mut decoder = codec.stream_decoder();
    decoder
        .push(sse(json!({
            "type": "message_start",
            "message": {"id": "msg_y", "type": "message", "role": "assistant", "content": [], "stop_reason": null, "stop_sequence": null}
        })))
        .unwrap();
    decoder
        .push(sse(json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""}
        })))
        .unwrap();
    let error = decoder
        .push(sse(json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "不该出现在推理块上"}
        })))
        .expect_err("正文增量不能落在推理块上");
    assert!(
        error.summary.contains("delta_type_incompatible_with_block"),
        "{error:?}"
    );
}
