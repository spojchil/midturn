//! 协议 codec 共用的 one-shot 与 SSE HTTP 传输。

use std::io::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, AUTHORIZATION};
#[cfg(any(feature = "openai", feature = "anthropic"))]
use serde_json::Map;
use serde_json::Value;

use crate::ports::{
    Model, ModelRequest, ModelResponse, ModelStreamEvent, ModelStreamSink, PortFuture,
};
use crate::types::{AgentError, AgentErrorKind};
#[cfg(any(feature = "openai", feature = "anthropic"))]
use crate::types::{ContentPart, JsonObject};

#[cfg(feature = "anthropic")]
use crate::adapters::anthropic::messages::{
    AnthropicMessagesCodec, RequestOptions as AnthropicOptions,
};
#[cfg(feature = "openai")]
use crate::adapters::openai::{
    chat::{OpenAiChatCodec, RequestOptions as OpenAiChatOptions},
    responses::{OpenAiResponsesCodec, RequestOptions as OpenAiResponsesOptions},
};

use super::config::{HttpAuth, HttpModelConfig, WireLogPolicy};
use super::sse::{SseEvent, SseParser};

/// HTTP 模型使用的 wire 协议及其请求选项。
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Protocol {
    #[cfg(feature = "openai")]
    OpenAiChat(OpenAiChatOptions),
    #[cfg(feature = "openai")]
    OpenAiResponses(OpenAiResponsesOptions),
    #[cfg(feature = "anthropic")]
    AnthropicMessages(AnthropicOptions),
}

impl Protocol {
    #[cfg(feature = "openai")]
    pub fn openai_chat() -> Self {
        Self::OpenAiChat(OpenAiChatOptions::default())
    }

    #[cfg(feature = "openai")]
    pub fn openai_chat_with(options: OpenAiChatOptions) -> Self {
        Self::OpenAiChat(options)
    }

    #[cfg(feature = "openai")]
    pub fn openai_responses() -> Self {
        Self::OpenAiResponses(OpenAiResponsesOptions::default())
    }

    #[cfg(feature = "openai")]
    pub fn openai_responses_with(options: OpenAiResponsesOptions) -> Self {
        Self::OpenAiResponses(options)
    }

    #[cfg(feature = "anthropic")]
    pub fn anthropic_messages() -> Self {
        Self::AnthropicMessages(AnthropicOptions::default())
    }

    #[cfg(feature = "anthropic")]
    pub fn anthropic_messages_with(options: AnthropicOptions) -> Self {
        Self::AnthropicMessages(options)
    }

    pub fn label(&self) -> &'static str {
        #[cfg(any(feature = "openai", feature = "anthropic"))]
        {
            match self {
                #[cfg(feature = "openai")]
                Self::OpenAiChat(_) => "openai-chat",
                #[cfg(feature = "openai")]
                Self::OpenAiResponses(_) => "openai-responses",
                #[cfg(feature = "anthropic")]
                Self::AnthropicMessages(_) => "anthropic-messages",
            }
        }
        #[cfg(not(any(feature = "openai", feature = "anthropic")))]
        unreachable!("no wire protocol feature is enabled")
    }

    pub(crate) fn default_auth(&self, api_key: String) -> HttpAuth {
        #[cfg(any(feature = "openai", feature = "anthropic"))]
        {
            match self {
                #[cfg(feature = "openai")]
                Self::OpenAiChat(_) | Self::OpenAiResponses(_) => HttpAuth::Bearer(api_key),
                #[cfg(feature = "anthropic")]
                Self::AnthropicMessages(_) => HttpAuth::Headers(vec![
                    ("x-api-key".to_owned(), api_key),
                    ("anthropic-version".to_owned(), "2023-06-01".to_owned()),
                ]),
            }
        }
        #[cfg(not(any(feature = "openai", feature = "anthropic")))]
        {
            let _ = api_key;
            unreachable!("no wire protocol feature is enabled")
        }
    }

    fn codec(&self) -> Arc<dyn WireCodec> {
        #[cfg(any(feature = "openai", feature = "anthropic"))]
        {
            match self {
                #[cfg(feature = "openai")]
                Self::OpenAiChat(options) => Arc::new(OpenAiChatCodec::new(options.clone())),
                #[cfg(feature = "openai")]
                Self::OpenAiResponses(options) => {
                    Arc::new(OpenAiResponsesCodec::new(options.clone()))
                }
                #[cfg(feature = "anthropic")]
                Self::AnthropicMessages(options) => {
                    Arc::new(AnthropicMessagesCodec::new(options.clone()))
                }
            }
        }
        #[cfg(not(any(feature = "openai", feature = "anthropic")))]
        unreachable!("no wire protocol feature is enabled")
    }
}

/// codec 只处理 wire 映射；HTTP、日志和状态码处理统一由 [`HttpModel`] 完成。
pub(crate) trait WireCodec: Send + Sync {
    fn encode_request(&self, request: &ModelRequest, model: &str) -> Result<Value, AgentError>;

    fn decode_response(&self, response: Value) -> Result<ModelResponse, AgentError>;

    fn stream_decoder(&self) -> Box<dyn WireStreamDecoder>;
}

/// 一个 SSE 事件经协议解码后可以同时产生多个进度事件和一个完整响应。
#[derive(Debug)]
pub(crate) struct StreamDecode {
    pub(crate) events: Vec<ModelStreamEvent>,
    pub(crate) response: Option<ModelResponse>,
}

impl StreamDecode {
    pub(crate) fn events(events: Vec<ModelStreamEvent>) -> Self {
        Self {
            events,
            response: None,
        }
    }

    pub(crate) fn completed(events: Vec<ModelStreamEvent>, response: ModelResponse) -> Self {
        Self {
            events,
            response: Some(response),
        }
    }
}

/// 有状态的协议流解码器。EOF 本身不是任何内置协议的成功终态。
pub(crate) trait WireStreamDecoder: Send {
    fn push(&mut self, event: SseEvent) -> Result<StreamDecode, AgentError>;

    fn finish_eof(&mut self) -> Result<(), AgentError>;
}

/// 使用完整 endpoint 的 HTTP 模型适配器。
///
/// [`Model::complete`] 读取一个完整 JSON 响应；[`Model::complete_stream`] 请求 SSE，并在
/// 协议终态到达前持续聚合正文和工具调用事件。
pub struct HttpModel {
    client: reqwest::Client,
    endpoint: reqwest::Url,
    endpoint_label: String,
    model: String,
    headers: HeaderMap,
    protocol_label: &'static str,
    wire_log: WireLogPolicy,
    max_response_bytes: usize,
    request_sequence: AtomicU64,
    codec: Arc<dyn WireCodec>,
}

impl HttpModel {
    pub fn new(config: HttpModelConfig) -> Result<Self, AgentError> {
        let endpoint = validate_endpoint(&config.endpoint, config.allow_insecure_http)?;
        if config.model.trim().is_empty() {
            return Err(model_error("transport_empty_model"));
        }
        if config.timeout.is_zero() {
            return Err(model_error("transport_zero_timeout"));
        }
        if config.max_response_bytes == 0 {
            return Err(model_error("transport_zero_response_limit"));
        }

        let endpoint_label = endpoint_without_query(&endpoint);
        let headers = build_headers(config.auth)?;
        let protocol_label = config.protocol.label();
        let codec = config.protocol.codec();
        let mut client_builder = reqwest::Client::builder()
            .timeout(config.timeout)
            // 自定义 API-key header 不在 reqwest 的跨源剥离名单中。完整 endpoint 应直接
            // 指向最终地址，因此默认拒绝重定向，避免凭据被 30x 转发到另一 origin。
            .redirect(reqwest::redirect::Policy::none());
        if is_loopback_host(&endpoint) {
            // loopback 明文请求属于显式允许的本机边界，不能再被环境代理转发到外部。
            client_builder = client_builder.no_proxy();
        }
        let client = client_builder.build().map_err(|error| {
            model_error(format!("transport_client_failed: {}", error.without_url()))
        })?;

        Ok(Self {
            client,
            endpoint,
            endpoint_label,
            model: config.model,
            headers,
            protocol_label,
            wire_log: config.wire_log,
            max_response_bytes: config.max_response_bytes,
            request_sequence: AtomicU64::new(0),
            codec,
        })
    }
}

impl Model for HttpModel {
    fn complete<'a>(
        &'a self,
        request: ModelRequest,
    ) -> PortFuture<'a, Result<ModelResponse, AgentError>> {
        Box::pin(async move {
            // 在 await 前完成规范请求到 wire 请求的投影，避免 codec 借用跨越网络等待。
            let body = self.codec.encode_request(&request, &self.model)?;
            let request_sequence = self.request_sequence.fetch_add(1, Ordering::Relaxed) + 1;
            self.log_request(request_sequence, &body);
            let mut response = self
                .client
                .post(self.endpoint.clone())
                .headers(self.headers.clone())
                .json(&body)
                .send()
                .await
                .map_err(|error| {
                    model_error(format!("transport_request_failed: {}", error.without_url()))
                })?;

            let status = response.status();
            let body = read_response_body(&mut response, self.max_response_bytes).await?;
            self.log_response(request_sequence, status.as_u16(), &body);
            if !status.is_success() {
                let code = provider_error_code(&body)
                    .map(|code| format!(":{code}"))
                    .unwrap_or_default();
                return Err(model_error(format!(
                    "transport_http_{}{code}",
                    status.as_u16()
                )));
            }

            let value = serde_json::from_str(&body)
                .map_err(|error| model_error(format!("transport_json_failed: {error}")))?;
            self.codec.decode_response(value)
        })
    }

    fn complete_stream<'a>(
        &'a self,
        request: ModelRequest,
        sink: &'a mut dyn ModelStreamSink,
    ) -> PortFuture<'a, Result<ModelResponse, AgentError>> {
        Box::pin(async move {
            let mut body = self.codec.encode_request(&request, &self.model)?;
            enable_streaming(&mut body)?;
            let request_sequence = self.request_sequence.fetch_add(1, Ordering::Relaxed) + 1;
            self.log_request(request_sequence, &body);
            let mut response = self
                .client
                .post(self.endpoint.clone())
                .headers(self.headers.clone())
                .header(ACCEPT, "text/event-stream")
                .json(&body)
                .send()
                .await
                .map_err(|error| {
                    model_error(format!("transport_request_failed: {}", error.without_url()))
                })?;

            let status = response.status();
            if !status.is_success() {
                let body = read_response_body(&mut response, self.max_response_bytes).await?;
                self.log_response(request_sequence, status.as_u16(), &body);
                let code = provider_error_code(&body)
                    .map(|code| format!(":{code}"))
                    .unwrap_or_default();
                return Err(model_error(format!(
                    "transport_http_{}{code}",
                    status.as_u16()
                )));
            }
            if response
                .content_length()
                .is_some_and(|length| length > self.max_response_bytes as u64)
            {
                return Err(model_error("transport_response_too_large"));
            }

            let mut decoder = self.codec.stream_decoder();
            let mut parser = SseParser::new(self.max_response_bytes);
            let mut received_bytes = 0_usize;
            let mut event_sequence = 0_u64;

            while let Some(chunk) = response.chunk().await.map_err(|error| {
                model_error(format!("transport_body_failed: {}", error.without_url()))
            })? {
                received_bytes = received_bytes
                    .checked_add(chunk.len())
                    .ok_or_else(|| model_error("transport_response_too_large"))?;
                if received_bytes > self.max_response_bytes {
                    return Err(model_error("transport_response_too_large"));
                }
                let mut completed = None;
                for event in parser.push(&chunk)? {
                    event_sequence += 1;
                    self.log_stream_event(request_sequence, event_sequence, &event);
                    if let Some(response) = emit_decoded(decoder.as_mut(), event, sink).await? {
                        completed = Some(response);
                    }
                }
                // 必须先消费本次 parser 已经产出的全部事件。否则与协议终态位于同一
                // HTTP chunk 的尾随事件会被静默丢弃，decoder 的 after-terminal 校验永远
                // 没有机会执行。
                if let Some(response) = completed {
                    return Ok(response);
                }
            }

            let mut completed = None;
            for event in parser.finish()? {
                event_sequence += 1;
                self.log_stream_event(request_sequence, event_sequence, &event);
                if let Some(response) = emit_decoded(decoder.as_mut(), event, sink).await? {
                    completed = Some(response);
                }
            }
            if let Some(response) = completed {
                return Ok(response);
            }
            decoder.finish_eof()?;
            Err(model_error("transport_stream_ended_without_terminal_event"))
        })
    }
}

async fn emit_decoded(
    decoder: &mut dyn WireStreamDecoder,
    event: SseEvent,
    sink: &mut dyn ModelStreamSink,
) -> Result<Option<ModelResponse>, AgentError> {
    let decoded = decoder.push(event)?;
    for event in decoded.events {
        sink.emit(event).await?;
    }
    Ok(decoded.response)
}

async fn read_response_body(
    response: &mut reqwest::Response,
    max_response_bytes: usize,
) -> Result<String, AgentError> {
    if response
        .content_length()
        .is_some_and(|length| length > max_response_bytes as u64)
    {
        return Err(model_error("transport_response_too_large"));
    }

    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| model_error(format!("transport_body_failed: {}", error.without_url())))?
    {
        let Some(new_length) = body.len().checked_add(chunk.len()) else {
            return Err(model_error("transport_response_too_large"));
        };
        if new_length > max_response_bytes {
            return Err(model_error("transport_response_too_large"));
        }
        body.extend_from_slice(&chunk);
    }
    String::from_utf8(body).map_err(|_| model_error("transport_body_not_utf8"))
}

impl HttpModel {
    fn log_request(&self, sequence: u64, body: &Value) {
        if self.wire_log != WireLogPolicy::Full {
            return;
        }
        let body = serde_json::to_string_pretty(body).unwrap_or_else(|_| body.to_string());
        let _ = writeln!(
            std::io::stderr().lock(),
            "[transport/{}][wire][request:{sequence}] {}\n{body}",
            self.protocol_label,
            self.endpoint_label,
        );
    }

    fn log_response(&self, sequence: u64, status: u16, body: &str) {
        if self.wire_log != WireLogPolicy::Full {
            return;
        }
        let formatted = serde_json::from_str::<Value>(body)
            .ok()
            .and_then(|value| serde_json::to_string_pretty(&value).ok())
            .unwrap_or_else(|| escape_terminal_text(body));
        let _ = writeln!(
            std::io::stderr().lock(),
            "[transport/{}][wire][response:{sequence}][http:{status}]\n{formatted}",
            self.protocol_label
        );
    }

    fn log_stream_event(&self, sequence: u64, event_sequence: u64, event: &SseEvent) {
        if self.wire_log != WireLogPolicy::Full {
            return;
        }
        let data = serde_json::from_str::<Value>(&event.data)
            .ok()
            .and_then(|value| serde_json::to_string_pretty(&value).ok())
            .unwrap_or_else(|| escape_terminal_text(&event.data));
        let _ = writeln!(
            std::io::stderr().lock(),
            "[transport/{}][wire][response:{sequence}][sse:{event_sequence}][event:{}]\n{data}",
            self.protocol_label,
            escape_terminal_text(&event.event),
        );
    }
}

fn escape_terminal_text(value: &str) -> String {
    value.chars().flat_map(char::escape_debug).collect()
}

fn validate_endpoint(
    endpoint: &str,
    allow_insecure_http: bool,
) -> Result<reqwest::Url, AgentError> {
    let parsed =
        reqwest::Url::parse(endpoint).map_err(|_| model_error("transport_invalid_endpoint"))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err(model_error("transport_invalid_endpoint"));
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(model_error("transport_endpoint_has_userinfo"));
    }
    if parsed.fragment().is_some() {
        return Err(model_error("transport_endpoint_has_fragment"));
    }
    if parsed.scheme() == "http" && !allow_insecure_http && !is_loopback_host(&parsed) {
        return Err(model_error("transport_insecure_http_endpoint"));
    }
    Ok(parsed)
}

fn is_loopback_host(endpoint: &reqwest::Url) -> bool {
    let Some(host) = endpoint.host_str() else {
        return false;
    };
    let address_host = host
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(host);
    host.eq_ignore_ascii_case("localhost")
        || address_host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

fn endpoint_without_query(endpoint: &reqwest::Url) -> String {
    let mut safe = endpoint.clone();
    safe.set_query(None);
    safe.set_fragment(None);
    safe.to_string()
}

fn build_headers(auth: HttpAuth) -> Result<HeaderMap, AgentError> {
    let mut headers = HeaderMap::new();
    match auth {
        HttpAuth::Bearer(token) => {
            if token.is_empty() {
                return Err(model_error("transport_empty_auth_value"));
            }
            let mut value = header_value(&format!("Bearer {token}"))?;
            value.set_sensitive(true);
            headers.insert(AUTHORIZATION, value);
        }
        HttpAuth::ApiKeyHeader { name, value } => {
            append_header(&mut headers, &name, &value)?;
        }
        HttpAuth::Headers(values) => {
            for (name, value) in values {
                append_header(&mut headers, &name, &value)?;
            }
        }
        HttpAuth::None => {}
    }
    Ok(headers)
}

fn append_header(headers: &mut HeaderMap, name: &str, value: &str) -> Result<(), AgentError> {
    if value.is_empty() {
        return Err(model_error("transport_empty_auth_value"));
    }
    let name = HeaderName::from_bytes(name.as_bytes())
        .map_err(|_| model_error("transport_invalid_auth_header_name"))?;
    let mut value = header_value(value)?;
    value.set_sensitive(true);
    headers.append(name, value);
    Ok(())
}

fn header_value(value: &str) -> Result<HeaderValue, AgentError> {
    HeaderValue::from_str(value).map_err(|_| model_error("transport_invalid_auth_header_value"))
}

/// 将协议扩展字段合并进请求，同时保护核心结构。one-shot 编码先固定为非流式；
/// [`HttpModel::complete_stream`] 会在发送前把该字段切换为 `true`。
#[cfg(any(feature = "openai", feature = "anthropic"))]
pub(crate) fn merge_request_fields(
    body: &mut Map<String, Value>,
    additional_fields: &JsonObject,
    protected_fields: &[&str],
    error_prefix: &str,
) -> Result<(), AgentError> {
    for (key, value) in additional_fields {
        if key == "stream" {
            return Err(model_error("transport_stream_option_is_managed"));
        }
        if protected_fields.contains(&key.as_str()) {
            return Err(model_error(format!(
                "{error_prefix}_reserved_request_field:{key}"
            )));
        }
        body.insert(key.clone(), value.clone());
    }
    body.insert("stream".to_owned(), Value::Bool(false));
    Ok(())
}

fn enable_streaming(body: &mut Value) -> Result<(), AgentError> {
    let object = body
        .as_object_mut()
        .ok_or_else(|| model_error("transport_request_body_not_object"))?;
    object.insert("stream".to_owned(), Value::Bool(true));
    Ok(())
}

pub(crate) fn model_error(summary: impl Into<String>) -> AgentError {
    AgentError::new(AgentErrorKind::Model, summary)
}

#[cfg(any(feature = "openai", feature = "anthropic"))]
pub(crate) fn content_as_text(parts: &[ContentPart]) -> String {
    parts
        .iter()
        .map(|part| match part {
            ContentPart::Text { text } => text.clone(),
            ContentPart::Json { value } => value.to_string(),
            ContentPart::Opaque { data, .. } => data.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(feature = "openai")]
pub(crate) fn parse_arguments(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_owned()))
}

fn provider_error_code(body: &str) -> Option<String> {
    let value = serde_json::from_str::<Value>(body).ok()?;
    let code = value
        .pointer("/error/code")
        .or_else(|| value.pointer("/error/type"))
        .or_else(|| value.get("code"))?
        .as_str()?;
    let safe = code
        .chars()
        .take(80)
        .filter(|character| character.is_ascii_alphanumeric() || "._-".contains(*character))
        .collect::<String>();
    (!safe.is_empty()).then_some(safe)
}

#[cfg(test)]
mod tests {
    #[cfg(any(feature = "openai", feature = "anthropic"))]
    use std::io::{Read, Write};
    #[cfg(any(feature = "openai", feature = "anthropic"))]
    use std::net::{TcpListener, TcpStream};
    #[cfg(any(feature = "openai", feature = "anthropic"))]
    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
    #[cfg(any(feature = "openai", feature = "anthropic"))]
    use std::sync::Arc;
    #[cfg(any(feature = "openai", feature = "anthropic"))]
    use std::thread;
    #[cfg(any(feature = "openai", feature = "anthropic"))]
    use std::time::Duration;

    use super::*;

    #[cfg(feature = "openai")]
    #[derive(Default)]
    struct RecordingSink {
        events: Vec<ModelStreamEvent>,
    }

    #[cfg(feature = "openai")]
    impl ModelStreamSink for RecordingSink {
        fn emit<'a>(
            &'a mut self,
            event: ModelStreamEvent,
        ) -> PortFuture<'a, Result<(), AgentError>> {
            Box::pin(async move {
                self.events.push(event);
                Ok(())
            })
        }
    }

    #[cfg(any(feature = "openai", feature = "anthropic"))]
    fn test_protocol() -> Protocol {
        #[cfg(feature = "openai")]
        return Protocol::openai_chat();
        #[cfg(all(not(feature = "openai"), feature = "anthropic"))]
        return Protocol::anthropic_messages();
    }

    #[cfg(any(feature = "openai", feature = "anthropic"))]
    fn read_http_request(stream: &mut TcpStream) -> Vec<u8> {
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut request = Vec::new();
        let mut buffer = [0_u8; 4096];
        let header_len = loop {
            let read = stream.read(&mut buffer).unwrap();
            assert!(read > 0, "测试请求在 header 完整前断开");
            request.extend_from_slice(&buffer[..read]);
            assert!(request.len() <= 1024 * 1024, "测试请求异常过大");
            if let Some(position) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                break position + 4;
            }
        };

        let headers = String::from_utf8_lossy(&request[..header_len]);
        let content_len = headers
            .lines()
            .filter_map(|line| line.split_once(':'))
            .find_map(|(name, value)| {
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().unwrap())
            })
            .unwrap_or(0);
        let expected_len = header_len.checked_add(content_len).unwrap();
        while request.len() < expected_len {
            let read = stream.read(&mut buffer).unwrap();
            assert!(read > 0, "测试请求在 body 完整前断开");
            request.extend_from_slice(&buffer[..read]);
            assert!(request.len() <= 1024 * 1024, "测试请求异常过大");
        }
        request
    }

    #[test]
    fn endpoint_must_be_complete_http_url() {
        assert_eq!(
            validate_endpoint("api.example.test/v1/chat", false)
                .unwrap_err()
                .summary,
            "transport_invalid_endpoint"
        );
        assert_eq!(
            validate_endpoint("file:///tmp/response", false)
                .unwrap_err()
                .summary,
            "transport_invalid_endpoint"
        );
        validate_endpoint("https://api.example.test/v1/chat", false).unwrap();
    }

    #[test]
    fn endpoint_rejects_embedded_username_or_password() {
        for endpoint in [
            "https://user@api.example.test/v1/chat",
            "https://user:password@api.example.test/v1/chat",
        ] {
            assert_eq!(
                validate_endpoint(endpoint, false).unwrap_err().summary,
                "transport_endpoint_has_userinfo"
            );
        }
    }

    #[test]
    fn remote_plain_http_requires_explicit_opt_in() {
        assert_eq!(
            validate_endpoint("http://api.example.test/v1/chat", false)
                .unwrap_err()
                .summary,
            "transport_insecure_http_endpoint"
        );
        validate_endpoint("http://api.example.test/v1/chat", true).unwrap();
        validate_endpoint("http://127.0.0.1:8080/v1/chat", false).unwrap();
        validate_endpoint("http://[::1]:8080/v1/chat", false).unwrap();
        validate_endpoint("http://localhost:8080/v1/chat", false).unwrap();
    }

    #[test]
    fn non_json_wire_text_escapes_terminal_controls() {
        let escaped = escape_terminal_text("中文 bad\u{1b}[31m\r\n");
        assert!(!escaped.chars().any(char::is_control));
        assert!(escaped.contains("中文"));
        assert!(escaped.contains("\\u{1b}"));
        assert!(escaped.contains("\\r\\n"));
    }

    #[test]
    #[cfg(any(feature = "openai", feature = "anthropic"))]
    fn request_fields_protect_structure_and_keep_stream_mode_managed() {
        let mut body = Map::new();
        let mut additional = JsonObject::new();
        additional.insert("temperature".to_owned(), Value::from(0.5));
        merge_request_fields(&mut body, &additional, &["model"], "test").unwrap();
        assert_eq!(body.get("temperature"), Some(&Value::from(0.5)));
        assert_eq!(body.get("stream"), Some(&Value::Bool(false)));

        additional.insert("stream".to_owned(), Value::Bool(true));
        assert_eq!(
            merge_request_fields(&mut body, &additional, &["model"], "test")
                .unwrap_err()
                .summary,
            "transport_stream_option_is_managed"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    #[cfg(any(feature = "openai", feature = "anthropic"))]
    async fn redirect_is_not_followed_with_sensitive_custom_headers() {
        let destination = TcpListener::bind("127.0.0.1:0").unwrap();
        destination.set_nonblocking(true).unwrap();
        let destination_address = destination.local_addr().unwrap();
        let destination_hit = Arc::new(AtomicBool::new(false));
        let stop_destination = Arc::new(AtomicBool::new(false));
        let destination_hit_in_thread = Arc::clone(&destination_hit);
        let stop_destination_in_thread = Arc::clone(&stop_destination);
        let destination_thread = thread::spawn(move || {
            while !stop_destination_in_thread.load(AtomicOrdering::SeqCst) {
                match destination.accept() {
                    Ok((mut stream, _)) => {
                        destination_hit_in_thread.store(true, AtomicOrdering::SeqCst);
                        let _request = read_http_request(&mut stream);
                        let _ = stream.write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
                        );
                        break;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("目标测试服务器失败: {error}"),
                }
            }
        });

        let redirect = TcpListener::bind("127.0.0.1:0").unwrap();
        let redirect_address = redirect.local_addr().unwrap();
        let redirect_thread = thread::spawn(move || {
            let (mut stream, _) = redirect.accept().unwrap();
            let _request = read_http_request(&mut stream);
            write!(
                stream,
                "HTTP/1.1 302 Found\r\nLocation: http://{destination_address}/stolen\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )
            .unwrap();
        });

        let model = HttpModel::new(
            HttpModelConfig::new(
                format!("http://{redirect_address}/start"),
                "unused-default-key",
                "test-model",
                test_protocol(),
            )
            .with_auth(HttpAuth::api_key_header("x-api-key", "must-not-leak")),
        )
        .unwrap();
        let error = model
            .complete(ModelRequest::test(Vec::new(), Vec::new()))
            .await
            .unwrap_err();

        stop_destination.store(true, AtomicOrdering::SeqCst);
        redirect_thread.join().unwrap();
        destination_thread.join().unwrap();
        assert_eq!(error.summary, "transport_http_302");
        assert!(
            !destination_hit.load(AtomicOrdering::SeqCst),
            "敏感自定义 header 不得随重定向发往另一 origin"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    #[cfg(any(feature = "openai", feature = "anthropic"))]
    async fn transport_errors_do_not_expose_endpoint_query() {
        // 保持随机端口的所有权，接收请求后直接断开。若先释放端口，并行测试可能恰好重新绑定
        // 同一端口，既消费别人的测试请求，也让本用例意外收到别人的 HTTP 响应。
        let unavailable = TcpListener::bind("127.0.0.1:0").unwrap();
        let unavailable_address = unavailable.local_addr().unwrap();
        let unavailable_thread = thread::spawn(move || {
            let (mut stream, _) = unavailable.accept().unwrap();
            let _request = read_http_request(&mut stream);
            // 不发送 HTTP 响应，让 transport 得到一个确定且不含 endpoint 的连接错误。
        });
        let secret = "query-secret-must-not-leak";
        let model = HttpModel::new(HttpModelConfig::new(
            format!("http://{unavailable_address}/v1/messages?api_key={secret}"),
            "header-key",
            "test-model",
            test_protocol(),
        ))
        .unwrap();

        let error = model
            .complete(ModelRequest::test(Vec::new(), Vec::new()))
            .await
            .unwrap_err();
        unavailable_thread.join().unwrap();
        assert_eq!(error.kind, AgentErrorKind::Model);
        assert!(error.summary.starts_with("transport_request_failed:"));
        assert!(!error.summary.contains(secret));
        assert!(!error.summary.contains("api_key"));
    }

    #[tokio::test(flavor = "current_thread")]
    #[cfg(any(feature = "openai", feature = "anthropic"))]
    async fn response_body_is_bounded_before_json_decoding() {
        let server = TcpListener::bind("127.0.0.1:0").unwrap();
        let server_address = server.local_addr().unwrap();
        let server_thread = thread::spawn(move || {
            let (mut stream, _) = server.accept().unwrap();
            let _request = read_http_request(&mut stream);
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 10\r\nConnection: close\r\n\r\n0123456789",
                )
                .unwrap();
        });
        let model = HttpModel::new(
            HttpModelConfig::new(
                format!("http://{server_address}/response"),
                "header-key",
                "test-model",
                test_protocol(),
            )
            .with_max_response_bytes(4),
        )
        .unwrap();

        let error = model
            .complete(ModelRequest::test(Vec::new(), Vec::new()))
            .await
            .unwrap_err();
        server_thread.join().unwrap();
        assert_eq!(error.summary, "transport_response_too_large");
    }

    #[tokio::test(flavor = "current_thread")]
    #[cfg(feature = "openai")]
    async fn streaming_transport_sets_stream_true_and_delivers_chat_events() {
        let server = TcpListener::bind("127.0.0.1:0").unwrap();
        let server_address = server.local_addr().unwrap();
        let server_thread = thread::spawn(move || {
            let (mut stream, _) = server.accept().unwrap();
            let request = read_http_request(&mut stream);
            let request = String::from_utf8_lossy(&request);
            assert!(request.contains("\"stream\":true"));
            assert!(request
                .to_ascii_lowercase()
                .contains("accept: text/event-stream"));

            let body = concat!(
                "data: {\"id\":\"chat_http\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"你好\"},\"finish_reason\":null}]}\n\n",
                "data: {\"id\":\"chat_http\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_http\",\"type\":\"function\",\"function\":{\"name\":\"add\",\"arguments\":\"{\\\"x\\\":1}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
                "data: [DONE]\n\n",
            );
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .unwrap();
            // 故意在中文 UTF-8 附近拆成很小的网络写入。
            for chunk in body.as_bytes().chunks(7) {
                stream.write_all(chunk).unwrap();
                stream.flush().unwrap();
            }
        });

        let model = HttpModel::new(HttpModelConfig::new(
            format!("http://{server_address}/v1/chat/completions"),
            "header-key",
            "test-model",
            Protocol::openai_chat(),
        ))
        .unwrap();
        let mut sink = RecordingSink::default();
        let response = model
            .complete_stream(ModelRequest::test(Vec::new(), Vec::new()), &mut sink)
            .await
            .unwrap();
        server_thread.join().unwrap();

        assert_eq!(response.output.text_content(), "你好");
        assert_eq!(response.output.tool_calls[0].id.as_str(), "call_http");
        assert!(matches!(
            &sink.events[0],
            ModelStreamEvent::TextStart { part_index: 0 }
        ));
        assert!(matches!(
            &sink.events[1],
            ModelStreamEvent::TextDelta { part_index: 0, delta } if delta == "你好"
        ));
        assert!(matches!(
            &sink.events[2],
            ModelStreamEvent::ToolCallReady { slot, call }
                if slot.get() == 0 && call.id.as_str() == "call_http"
        ));
        assert!(matches!(
            &sink.events[3],
            ModelStreamEvent::TextEnd { part_index: 0 }
        ));
        assert_eq!(
            sink.events[4],
            ModelStreamEvent::ToolCallsSealed { call_count: 1 }
        );
    }

    #[tokio::test(flavor = "current_thread")]
    #[cfg(feature = "openai")]
    async fn truncated_responses_stream_keeps_ready_events_but_never_seals() {
        let server = TcpListener::bind("127.0.0.1:0").unwrap();
        let server_address = server.local_addr().unwrap();
        let server_thread = thread::spawn(move || {
            let (mut stream, _) = server.accept().unwrap();
            let _request = read_http_request(&mut stream);
            stream
                .write_all(
                    concat!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
                        "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"type\":\"function_call\",\"id\":\"fc_0\",\"call_id\":\"call_0\",\"name\":\"add\",\"arguments\":\"\"}}\n\n",
                        "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"type\":\"function_call\",\"status\":\"completed\",\"id\":\"fc_0\",\"call_id\":\"call_0\",\"name\":\"add\",\"arguments\":\"{\\\"x\\\":1}\"}}\n\n",
                        "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":2,\"item_id\":\"fc_1\",\"delta\":\"{\\\"x\\\":\"}\n\n",
                    )
                    .as_bytes(),
                )
                .unwrap();
        });

        let model = HttpModel::new(HttpModelConfig::new(
            format!("http://{server_address}/v1/responses"),
            "header-key",
            "test-model",
            Protocol::openai_responses(),
        ))
        .unwrap();
        let mut sink = RecordingSink::default();
        let error = model
            .complete_stream(ModelRequest::test(Vec::new(), Vec::new()), &mut sink)
            .await
            .unwrap_err();
        server_thread.join().unwrap();

        assert_eq!(error.summary, "openai_responses_stream_missing_terminal");
        assert!(matches!(
            &sink.events[..],
            [ModelStreamEvent::ToolCallReady { slot, call }]
                if slot.get() == 0 && call.id.as_str() == "call_0"
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    #[cfg(feature = "openai")]
    async fn rejects_an_event_trailing_a_terminal_in_the_same_http_chunk() {
        let server = TcpListener::bind("127.0.0.1:0").unwrap();
        let server_address = server.local_addr().unwrap();
        let server_thread = thread::spawn(move || {
            let (mut stream, _) = server.accept().unwrap();
            let _request = read_http_request(&mut stream);
            let body = concat!(
                "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
                "data: [DONE]\n\n",
                "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"forged tail\"},\"finish_reason\":null}]}\n\n",
            );
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
        });

        let model = HttpModel::new(HttpModelConfig::new(
            format!("http://{server_address}/v1/chat/completions"),
            "header-key",
            "test-model",
            Protocol::openai_chat(),
        ))
        .unwrap();
        let mut sink = RecordingSink::default();
        let error = model
            .complete_stream(ModelRequest::test(Vec::new(), Vec::new()), &mut sink)
            .await
            .unwrap_err();
        server_thread.join().unwrap();
        assert_eq!(error.summary, "openai_chat_event_after_done");
    }
}
