//! OpenAI Chat Completions wire 协议。

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use crate::adapters::http::{
    content_as_text, merge_request_fields, model_error, parse_arguments, SseEvent, StreamDecode,
    WireCodec, WireStreamDecoder,
};
use crate::ports::{ModelRequest, ModelResponse, ModelStreamEvent};
use crate::types::{
    AgentError, ContentPart, JsonObject, ModelOutput, ModelUsage, ToolCall, ToolCallSlot,
    TranscriptItem,
};

const RAW_MESSAGE_KEY: &str = "openai.chat.message";
const RESPONSE_ID_KEY: &str = "openai.chat.response_id";

/// Chat Completions 请求的协议级可选字段。
///
/// `additional_fields` 可承载兼容端点扩展，但不得覆盖 `model`、`messages`、`tools`、
/// `tool_choice`、`max_tokens` 或由模型方法管理的 `stream`。`role_mappings` 可将框架中的自定义 canonical role
/// 映射到本协议接受的 wire role。
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct RequestOptions {
    pub max_tokens: Option<u64>,
    pub tool_choice: Option<Value>,
    pub additional_fields: JsonObject,
    pub role_mappings: BTreeMap<String, String>,
}

impl RequestOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_tool_choice(mut self, tool_choice: impl Into<Value>) -> Self {
        self.tool_choice = Some(tool_choice.into());
        self
    }

    pub fn with_additional_field(
        mut self,
        key: impl Into<String>,
        value: impl Into<Value>,
    ) -> Self {
        self.additional_fields.insert(key.into(), value.into());
        self
    }

    /// 将一个 canonical role 映射成 Chat Completions 接受的 wire role。
    pub fn with_role_mapping(
        mut self,
        canonical_role: impl Into<String>,
        wire_role: impl Into<String>,
    ) -> Self {
        self.role_mappings
            .insert(canonical_role.into(), wire_role.into());
        self
    }
}

pub(crate) struct OpenAiChatCodec {
    options: RequestOptions,
}

impl OpenAiChatCodec {
    pub(crate) fn new(options: RequestOptions) -> Self {
        Self { options }
    }
}

impl WireCodec for OpenAiChatCodec {
    fn encode_request(&self, request: &ModelRequest, model: &str) -> Result<Value, AgentError> {
        if self.options.max_tokens == Some(0) {
            return Err(model_error("openai_chat_invalid_max_tokens"));
        }
        if request.function_tools.is_empty() && self.options.tool_choice.is_some() {
            return Err(model_error("openai_chat_tool_choice_without_tools"));
        }

        let mut body = Map::new();
        body.insert("model".to_owned(), Value::String(model.to_owned()));
        body.insert(
            "messages".to_owned(),
            Value::Array(encode_messages(
                &request.transcript,
                &self.options.role_mappings,
            )?),
        );

        if !request.function_tools.is_empty() {
            let tools = request
                .function_tools
                .iter()
                .map(|tool| {
                    let mut function = json!({
                        "name": tool.name.as_str(),
                        "parameters": tool.input_schema,
                    });
                    if let Some(description) = &tool.description {
                        function["description"] = Value::String(description.clone());
                    }
                    json!({"type": "function", "function": function})
                })
                .collect();
            body.insert("tools".to_owned(), Value::Array(tools));
            if let Some(tool_choice) = &self.options.tool_choice {
                body.insert("tool_choice".to_owned(), tool_choice.clone());
            }
        }
        if let Some(max_tokens) = self.options.max_tokens {
            body.insert("max_tokens".to_owned(), Value::from(max_tokens));
        }
        merge_request_fields(
            &mut body,
            &self.options.additional_fields,
            &["model", "messages", "tools", "tool_choice", "max_tokens"],
            "openai_chat",
        )?;
        Ok(Value::Object(body))
    }

    fn decode_response(&self, value: Value) -> Result<ModelResponse, AgentError> {
        decode_chat_response(value)
    }

    fn stream_decoder(&self) -> Box<dyn WireStreamDecoder> {
        Box::new(ChatStreamDecoder::default())
    }
}

fn decode_chat_response(value: Value) -> Result<ModelResponse, AgentError> {
    let choice = value
        .pointer("/choices/0")
        .ok_or_else(|| model_error("openai_chat_response_missing_choice"))?;
    let message = choice
        .get("message")
        .ok_or_else(|| model_error("openai_chat_response_missing_message"))?;
    let (content, tool_calls) = decode_chat_message(message)?;

    let finish_reason = choice.get("finish_reason").cloned();
    let reason = finish_reason
        .as_ref()
        .and_then(Value::as_str)
        .ok_or_else(|| model_error("openai_chat_missing_finish_reason"))?;
    match reason {
        "stop" | "tool_calls" => {}
        "length" | "content_filter" => {
            return Err(model_error(format!(
                "openai_chat_unsuccessful_finish_reason:{reason}"
            )));
        }
        _ => return Err(model_error("openai_chat_unsupported_finish_reason")),
    }
    // 工具数组与终止原因必须双向一致，避免把缺失数组或意外附带的数组当成完整批次。
    match (tool_calls.is_empty(), reason) {
        (false, "tool_calls") | (true, "stop") => {}
        (false, _) => {
            return Err(model_error(
                "openai_chat_tool_calls_without_tool_calls_finish",
            ));
        }
        (true, _) => {
            return Err(model_error(
                "openai_chat_tool_calls_finish_without_tool_calls",
            ));
        }
    }

    let mut provider_data = JsonObject::new();
    // 原始消息保留兼容端点添加的未知字段，并在续轮时完整回放。
    provider_data.insert(RAW_MESSAGE_KEY.to_owned(), message.clone());
    if let Some(response_id) = value.get("id") {
        provider_data.insert(RESPONSE_ID_KEY.to_owned(), response_id.clone());
    }

    let usage = value.get("usage").map(|usage| ModelUsage {
        input_tokens: usage.get("prompt_tokens").and_then(Value::as_u64),
        output_tokens: usage.get("completion_tokens").and_then(Value::as_u64),
        total_tokens: usage.get("total_tokens").and_then(Value::as_u64),
        cached_input_tokens: usage
            .pointer("/prompt_tokens_details/cached_tokens")
            .and_then(Value::as_u64),
        cache_write_input_tokens: None,
        reasoning_output_tokens: usage
            .pointer("/completion_tokens_details/reasoning_tokens")
            .and_then(Value::as_u64),
    });

    Ok(ModelResponse {
        output: ModelOutput {
            content,
            tool_calls,
            provider_data,
        },
        finish_reason,
        usage,
    })
}

fn decode_chat_message(message: &Value) -> Result<(Vec<ContentPart>, Vec<ToolCall>), AgentError> {
    let message = message
        .as_object()
        .ok_or_else(|| model_error("openai_chat_message_not_object"))?;
    if message.get("role").and_then(Value::as_str) != Some("assistant") {
        return Err(model_error("openai_chat_message_role_not_assistant"));
    }
    if message
        .get("function_call")
        .is_some_and(|value| !value.is_null())
    {
        // 旧版 function_call 没有进入 canonical tool_calls，直接回放会绕过调用闭包校验。
        return Err(model_error("openai_chat_legacy_function_call_unsupported"));
    }

    let mut content = Vec::new();
    match message.get("content") {
        None | Some(Value::Null) => {}
        Some(Value::String(text)) if text.is_empty() => {}
        Some(Value::String(text)) => content.push(ContentPart::text(text)),
        Some(_) => return Err(model_error("openai_chat_message_content_not_string")),
    }
    // Chat 将安全拒答放在独立 refusal 字段。框架当前没有专用拒答 variant，因此把其
    // 解释文本作为模型可见正文暴露；原字段仍保留在 provider_data 中以便精确回放。
    match message.get("refusal") {
        None | Some(Value::Null) => {}
        Some(Value::String(refusal)) if refusal.is_empty() => {}
        Some(Value::String(refusal)) => content.push(ContentPart::text(refusal)),
        Some(_) => return Err(model_error("openai_chat_message_refusal_not_string")),
    }

    let calls = match message.get("tool_calls") {
        None | Some(Value::Null) => &[][..],
        Some(Value::Array(calls)) => calls.as_slice(),
        Some(_) => return Err(model_error("openai_chat_tool_calls_not_array")),
    };
    let tool_calls = calls
        .iter()
        .map(|call| {
            if call.get("type").and_then(Value::as_str) != Some("function") {
                return Err(model_error("openai_chat_unsupported_tool_type"));
            }
            let id = required_string(call, "id", "openai_chat_tool_call_missing_id")?;
            let function = call
                .get("function")
                .ok_or_else(|| model_error("openai_chat_tool_call_missing_function"))?;
            let name = required_string(function, "name", "openai_chat_tool_call_missing_name")?;
            let arguments = required_string(
                function,
                "arguments",
                "openai_chat_tool_call_missing_arguments",
            )?;
            Ok(ToolCall::new(id, name, parse_arguments(arguments)))
        })
        .collect::<Result<Vec<_>, AgentError>>()?;
    Ok((content, tool_calls))
}

fn validated_raw_message<'a>(
    output: &ModelOutput,
    raw: &'a Value,
) -> Result<&'a Value, AgentError> {
    let (content, tool_calls) = decode_chat_message(raw)?;
    if content != output.content || !same_tool_calls(&tool_calls, &output.tool_calls) {
        return Err(model_error("openai_chat_raw_message_canonical_mismatch"));
    }
    Ok(raw)
}

fn same_tool_calls(left: &[ToolCall], right: &[ToolCall]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.id == right.id && left.name == right.name && left.arguments == right.arguments
        })
}

#[derive(Default)]
struct JsonObjectBoundary {
    started: bool,
    completed: bool,
    expected_closers: Vec<char>,
    in_string: bool,
    escaped: bool,
}

impl JsonObjectBoundary {
    /// 吸收一段 JSON 文本；返回值只在顶层对象刚刚闭合时为 true。
    fn push(&mut self, fragment: &str) -> Result<bool, AgentError> {
        let was_completed = self.completed;
        for character in fragment.chars() {
            if self.completed {
                if !is_json_whitespace(character) {
                    return Err(model_error(
                        "openai_chat_stream_tool_arguments_after_complete",
                    ));
                }
                continue;
            }

            if !self.started {
                if is_json_whitespace(character) {
                    continue;
                }
                if character != '{' {
                    return Err(model_error("openai_chat_stream_tool_arguments_not_object"));
                }
                self.started = true;
                self.expected_closers.push('}');
                continue;
            }

            if self.in_string {
                if self.escaped {
                    self.escaped = false;
                } else {
                    match character {
                        '\\' => self.escaped = true,
                        '"' => self.in_string = false,
                        _ => {}
                    }
                }
                continue;
            }

            match character {
                '"' => self.in_string = true,
                '{' => self.expected_closers.push('}'),
                '[' => self.expected_closers.push(']'),
                '}' | ']' => {
                    if self.expected_closers.pop() != Some(character) {
                        return Err(model_error(
                            "openai_chat_stream_tool_arguments_mismatched_delimiter",
                        ));
                    }
                    if self.expected_closers.is_empty() {
                        self.completed = true;
                    }
                }
                _ => {}
            }
        }
        Ok(!was_completed && self.completed)
    }
}

fn is_json_whitespace(character: char) -> bool {
    matches!(character, ' ' | '\t' | '\r' | '\n')
}

#[derive(Default)]
struct ChatToolDraft {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
    argument_boundary: JsonObjectBoundary,
    parsed_arguments: Option<Value>,
    ready: Option<ToolCall>,
}

/// Chat 协议只有一段正文和一段推理，用固定的两个 part_index 表示。
const TEXT_PART: u32 = 0;
const REASONING_PART: u32 = 1;

#[derive(Default)]
struct ChatStreamDecoder {
    response_id: Option<Value>,
    role: Option<String>,
    content: String,
    /// Chat 协议没有内容段的开始/结束信号，只能由适配器合成：首个非空增量视为开始，
    /// `[DONE]` 时把还开着的段封口。
    text_started: bool,
    reasoning_started: bool,
    tools: BTreeMap<u32, ChatToolDraft>,
    message_extensions: Map<String, Value>,
    finish_reason: Option<Value>,
    usage: Option<Value>,
    terminal_chunk_seen: bool,
}

impl WireStreamDecoder for ChatStreamDecoder {
    fn push(&mut self, event: SseEvent) -> Result<StreamDecode, AgentError> {
        if self.terminal_chunk_seen {
            return Err(model_error("openai_chat_event_after_done"));
        }
        if event.data.trim() == "[DONE]" {
            return self.complete();
        }

        let value: Value = serde_json::from_str(&event.data)
            .map_err(|error| model_error(format!("openai_chat_stream_json_failed:{error}")))?;
        if value.get("error").is_some()
            || value.get("type").and_then(Value::as_str) == Some("error")
        {
            return Err(model_error("openai_chat_stream_error"));
        }
        set_consistent_value(
            &mut self.response_id,
            value.get("id"),
            "openai_chat_stream_response_id_changed",
        )?;
        if let Some(usage) = value.get("usage").filter(|usage| !usage.is_null()) {
            set_consistent_value(
                &mut self.usage,
                Some(usage),
                "openai_chat_stream_usage_changed",
            )?;
        }

        let mut events = Vec::new();
        let choices = value
            .get("choices")
            .and_then(Value::as_array)
            .ok_or_else(|| model_error("openai_chat_stream_missing_choices"))?;
        for choice in choices {
            let index = choice.get("index").and_then(Value::as_u64).unwrap_or(0);
            // one-shot 路径也只选择 choices[0]；其他候选不属于本次规范响应。
            if index != 0 {
                continue;
            }
            // Kimi 等兼容端点把流式 usage 放在终止 choice 内，而 OpenAI 放在顶层。
            // 两种位置统一进入同一规范字段；若同时出现则必须完全一致。
            if let Some(usage) = choice.get("usage").filter(|usage| !usage.is_null()) {
                set_consistent_value(
                    &mut self.usage,
                    Some(usage),
                    "openai_chat_stream_usage_changed",
                )?;
            }
            if let Some(delta) = choice.get("delta") {
                if let Some(content) = delta.get("content").filter(|value| !value.is_null()) {
                    let content = content.as_str().ok_or_else(|| {
                        model_error("openai_chat_stream_content_delta_not_string")
                    })?;
                    if !content.is_empty() {
                        self.content.push_str(content);
                        if !self.text_started {
                            self.text_started = true;
                            events.push(ModelStreamEvent::TextStart {
                                part_index: TEXT_PART,
                            });
                        }
                        events.push(ModelStreamEvent::TextDelta {
                            part_index: TEXT_PART,
                            delta: content.to_owned(),
                        });
                    }
                }
                // 多个兼容端点（DeepSeek、Kimi 等）用 reasoning_content 承载推理。
                if let Some(reasoning) = delta
                    .get("reasoning_content")
                    .filter(|value| !value.is_null())
                {
                    let reasoning = reasoning.as_str().ok_or_else(|| {
                        model_error("openai_chat_stream_reasoning_delta_not_string")
                    })?;
                    if !reasoning.is_empty() {
                        // 正文累积在 message_extensions 里（回放保真），这里只负责观测。
                        if !self.reasoning_started {
                            self.reasoning_started = true;
                            events.push(ModelStreamEvent::ReasoningStart {
                                part_index: REASONING_PART,
                            });
                        }
                        events.push(ModelStreamEvent::ReasoningDelta {
                            part_index: REASONING_PART,
                            delta: reasoning.to_owned(),
                        });
                    }
                }
                if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                    for call in calls {
                        if let Some((slot, call)) = self.push_tool_delta(call)? {
                            events.push(ModelStreamEvent::ToolCallReady { slot, call });
                        }
                    }
                }
                self.push_message_extensions(delta)?;
            }
            if let Some(reason) = choice.get("finish_reason").filter(|value| !value.is_null()) {
                set_consistent_value(
                    &mut self.finish_reason,
                    Some(reason),
                    "openai_chat_stream_finish_reason_changed",
                )?;
            }
        }
        Ok(StreamDecode::events(events))
    }

    fn finish_eof(&mut self) -> Result<(), AgentError> {
        Err(model_error("openai_chat_stream_missing_done"))
    }
}

impl ChatStreamDecoder {
    fn push_message_extensions(&mut self, delta: &Value) -> Result<(), AgentError> {
        let delta = delta
            .as_object()
            .ok_or_else(|| model_error("openai_chat_stream_delta_not_object"))?;
        for (field, value) in delta {
            match field.as_str() {
                "content" | "tool_calls" => {}
                "role" if value.is_null() => {}
                "role" => {
                    let role = value
                        .as_str()
                        .ok_or_else(|| model_error("openai_chat_stream_role_not_string"))?;
                    if role != "assistant" {
                        return Err(model_error("openai_chat_stream_role_not_assistant"));
                    }
                    set_consistent_string(
                        &mut self.role,
                        Some(role),
                        "openai_chat_stream_role_changed",
                    )?;
                }
                _ if value.is_null() => {}
                _ => merge_message_extension(&mut self.message_extensions, field, value)?,
            }
        }
        Ok(())
    }

    fn push_tool_delta(
        &mut self,
        call: &Value,
    ) -> Result<Option<(ToolCallSlot, ToolCall)>, AgentError> {
        let index = call
            .get("index")
            .and_then(Value::as_u64)
            .ok_or_else(|| model_error("openai_chat_stream_tool_delta_missing_index"))?;
        let index = u32::try_from(index)
            .map_err(|_| model_error("openai_chat_stream_tool_index_overflow"))?;
        let draft = self.tools.entry(index).or_default();
        set_consistent_string(
            &mut draft.id,
            call.get("id").and_then(Value::as_str),
            "openai_chat_stream_tool_id_changed",
        )?;
        if let Some(kind) = call.get("type").and_then(Value::as_str) {
            if kind != "function" {
                return Err(model_error(format!(
                    "openai_chat_stream_unsupported_tool_type:{kind}"
                )));
            }
        }
        if let Some(function) = call.get("function").filter(|value| !value.is_null()) {
            let function = function
                .as_object()
                .ok_or_else(|| model_error("openai_chat_stream_tool_function_not_object"))?;
            set_consistent_string(
                &mut draft.name,
                function.get("name").and_then(Value::as_str),
                "openai_chat_stream_tool_name_changed",
            )?;
            if let Some(arguments) = function.get("arguments").filter(|value| !value.is_null()) {
                let arguments = arguments
                    .as_str()
                    .ok_or_else(|| model_error("openai_chat_stream_tool_arguments_not_string"))?;
                draft.arguments.push_str(arguments);
                if draft.argument_boundary.push(arguments)? {
                    let parsed = serde_json::from_str::<Value>(&draft.arguments).map_err(|_| {
                        model_error("openai_chat_stream_tool_arguments_invalid_json")
                    })?;
                    if !parsed.is_object() {
                        return Err(model_error("openai_chat_stream_tool_arguments_not_object"));
                    }
                    draft.parsed_arguments = Some(parsed);
                }
            }
        }

        if draft.ready.is_none() {
            if let (Some(id), Some(name), Some(arguments)) = (
                draft.id.as_deref().filter(|value| !value.is_empty()),
                draft.name.as_deref().filter(|value| !value.is_empty()),
                draft.parsed_arguments.as_ref(),
            ) {
                let call = ToolCall::new(id, name, arguments.clone());
                draft.ready = Some(call.clone());
                return Ok(Some((ToolCallSlot::new(index), call)));
            }
        }
        Ok(None)
    }

    fn complete(&mut self) -> Result<StreamDecode, AgentError> {
        self.terminal_chunk_seen = true;
        let reason = self
            .finish_reason
            .as_ref()
            .and_then(Value::as_str)
            .ok_or_else(|| model_error("openai_chat_stream_missing_finish_reason"))?;
        if matches!(reason, "length" | "content_filter") {
            return Err(model_error(format!(
                "openai_chat_stream_unsuccessful_finish_reason:{reason}"
            )));
        }
        if !matches!(reason, "stop" | "tool_calls") {
            return Err(model_error("openai_chat_stream_unsupported_finish_reason"));
        }
        match (self.tools.is_empty(), reason) {
            (false, "tool_calls") | (true, "stop") => {}
            (false, _) => {
                return Err(model_error(
                    "openai_chat_stream_tool_calls_without_tool_calls_finish",
                ));
            }
            (true, _) => {
                return Err(model_error(
                    "openai_chat_stream_tool_calls_finish_without_tool_calls",
                ));
            }
        }

        let mut wire_calls = Vec::with_capacity(self.tools.len());
        for expected in 0..self.tools.len() {
            let index = u32::try_from(expected)
                .map_err(|_| model_error("openai_chat_stream_tool_count_overflow"))?;
            let draft = self
                .tools
                .get(&index)
                .ok_or_else(|| model_error("openai_chat_stream_non_contiguous_tool_indices"))?;
            let id = draft
                .id
                .as_deref()
                .filter(|value| !value.is_empty())
                .ok_or_else(|| model_error("openai_chat_stream_tool_call_missing_id"))?;
            let name = draft
                .name
                .as_deref()
                .filter(|value| !value.is_empty())
                .ok_or_else(|| model_error("openai_chat_stream_tool_call_missing_name"))?;
            draft
                .parsed_arguments
                .as_ref()
                .ok_or_else(|| model_error("openai_chat_stream_tool_call_incomplete_arguments"))?;
            wire_calls.push(json!({
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": draft.arguments},
            }));
        }

        let mut message = std::mem::take(&mut self.message_extensions);
        message.insert(
            "role".to_owned(),
            Value::String(self.role.clone().unwrap_or_else(|| "assistant".to_owned())),
        );
        message.insert(
            "content".to_owned(),
            if self.content.is_empty() {
                Value::Null
            } else {
                Value::String(self.content.clone())
            },
        );
        if !wire_calls.is_empty() {
            message.insert("tool_calls".to_owned(), Value::Array(wire_calls));
        }
        let message = Value::Object(message);
        let mut response = json!({
            "choices": [{"index": 0, "finish_reason": self.finish_reason.clone(), "message": message}],
        });
        if let Some(id) = &self.response_id {
            response["id"] = id.clone();
        }
        if let Some(usage) = &self.usage {
            response["usage"] = usage.clone();
        }
        let response = decode_chat_response(response)?;
        let call_count = u32::try_from(response.output.tool_calls.len())
            .map_err(|_| model_error("openai_chat_stream_tool_count_overflow"))?;
        let mut events = Vec::new();
        // Chat 没有段结束信号，只能在流终止时把还开着的段封口。
        if self.reasoning_started {
            events.push(ModelStreamEvent::ReasoningEnd {
                part_index: REASONING_PART,
            });
        }
        if self.text_started {
            events.push(ModelStreamEvent::TextEnd {
                part_index: TEXT_PART,
            });
        }
        for (index, call) in response.output.tool_calls.iter().cloned().enumerate() {
            let index = u32::try_from(index)
                .map_err(|_| model_error("openai_chat_stream_tool_count_overflow"))?;
            let draft = self
                .tools
                .get(&index)
                .ok_or_else(|| model_error("openai_chat_stream_final_call_missing_draft"))?;
            match &draft.ready {
                Some(ready) if ready == &call => {}
                Some(_) => {
                    return Err(model_error("openai_chat_stream_ready_call_mismatch"));
                }
                None => events.push(ModelStreamEvent::ToolCallReady {
                    slot: ToolCallSlot::new(index),
                    call,
                }),
            }
        }
        events.push(ModelStreamEvent::ToolCallsSealed { call_count });
        Ok(StreamDecode::completed(events, response))
    }
}

fn set_consistent_string(
    current: &mut Option<String>,
    incoming: Option<&str>,
    error: &'static str,
) -> Result<(), AgentError> {
    let Some(incoming) = incoming else {
        return Ok(());
    };
    match current {
        Some(current) if current != incoming => Err(model_error(error)),
        Some(_) => Ok(()),
        None => {
            *current = Some(incoming.to_owned());
            Ok(())
        }
    }
}

fn set_consistent_value(
    current: &mut Option<Value>,
    incoming: Option<&Value>,
    error: &'static str,
) -> Result<(), AgentError> {
    let Some(incoming) = incoming else {
        return Ok(());
    };
    match current {
        Some(current) if current != incoming => Err(model_error(error)),
        Some(_) => Ok(()),
        None => {
            *current = Some(incoming.clone());
            Ok(())
        }
    }
}

fn merge_message_extension(
    extensions: &mut Map<String, Value>,
    field: &str,
    incoming: &Value,
) -> Result<(), AgentError> {
    match (extensions.get_mut(field), incoming) {
        (None, value) => {
            extensions.insert(field.to_owned(), value.clone());
            Ok(())
        }
        (Some(Value::String(current)), Value::String(delta)) => {
            current.push_str(delta);
            Ok(())
        }
        (Some(current), incoming) if current == incoming => Ok(()),
        _ => Err(model_error(
            "openai_chat_stream_message_extension_changed_type_or_value",
        )),
    }
}

fn encode_messages(
    transcript: &[TranscriptItem],
    role_mappings: &BTreeMap<String, String>,
) -> Result<Vec<Value>, AgentError> {
    let mut messages = Vec::new();
    for item in transcript {
        match item {
            TranscriptItem::Input(message) => {
                let role = wire_role(&message.role, role_mappings)?;
                messages.push(json!({
                    "role": role,
                    "content": content_as_text(&message.content),
                }));
            }
            TranscriptItem::ModelOutput(output) => {
                if let Some(message) = output.provider_data.get(RAW_MESSAGE_KEY) {
                    // 服务端签发的调用结构和未知扩展字段必须与原响应一起回放。
                    messages.push(validated_raw_message(output, message)?.clone());
                    continue;
                }
                let mut message = json!({
                    "role": "assistant",
                    "content": if output.content.is_empty() {
                        Value::Null
                    } else {
                        Value::String(content_as_text(&output.content))
                    },
                });
                if !output.tool_calls.is_empty() {
                    message["tool_calls"] = Value::Array(
                        output
                            .tool_calls
                            .iter()
                            .map(|call| {
                                json!({
                                    "id": call.id.as_str(),
                                    "type": "function",
                                    "function": {
                                        "name": call.name.as_str(),
                                        "arguments": match &call.arguments {
                                            Value::String(raw) => raw.clone(),
                                            value => value.to_string(),
                                        },
                                    }
                                })
                            })
                            .collect(),
                    );
                }
                messages.push(message);
            }
            TranscriptItem::ToolResults(batch) => {
                messages.extend(batch.results.iter().map(|result| {
                    json!({
                        "role": "tool",
                        "tool_call_id": result.call_id.as_str(),
                        "content": content_as_text(&result.content),
                    })
                }));
            }
        }
    }
    Ok(messages)
}

fn wire_role<'a>(
    role: &'a str,
    role_mappings: &'a BTreeMap<String, String>,
) -> Result<&'a str, AgentError> {
    let lowered = role_mappings.get(role).map(String::as_str).unwrap_or(role);
    match lowered {
        "developer" | "system" | "user" | "assistant" => Ok(lowered),
        _ => Err(model_error(format!(
            "openai_chat_unsupported_input_role:{lowered}"
        ))),
    }
}

fn required_string<'a>(
    value: &'a Value,
    field: &str,
    error: &'static str,
) -> Result<&'a str, AgentError> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| model_error(error))
}
