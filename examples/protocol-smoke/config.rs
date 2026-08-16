//! 示例应用的命令行、环境变量与强类型配置。

use std::env;
use std::io;
use std::time::Duration;

use midturn::adapters::http::WireLogPolicy;
use midturn::{JsonObject, LevelFilter};
use serde_json::Value;

const DEFAULT_TIMEOUT_SECS: u64 = 90;
const USAGE: &str =
    "cargo run --example protocol_smoke --features all-adapters -- [chat|responses|anthropic|all]";

/// 冒烟示例可选择的协议；与携带请求选项的公共 [`midturn::adapters::http::Protocol`]
/// 分开保存，便于作为 endpoint 配置的稳定键。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SmokeProtocol {
    OpenAiChat,
    OpenAiResponses,
    AnthropicMessages,
}

/// OpenAI-compatible Chat 端点使用的输出 token 字段。
///
/// 许多兼容端点仍接收 `max_tokens`；部分推理模型则只接收
/// `max_completion_tokens`。值由 harness 统一设为小上限，不从 JSON 扩展中读取。
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum ChatTokenField {
    #[default]
    MaxTokens,
    MaxCompletionTokens,
}

impl SmokeProtocol {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::OpenAiChat => "openai-chat",
            Self::OpenAiResponses => "openai-responses",
            Self::AnthropicMessages => "anthropic-messages",
        }
    }
}

/// 示例应用汇总后的强类型配置；核心 crate 与协议适配器不读取这些外部来源。
pub(crate) struct SmokeConfig {
    pub(crate) api_key: String,
    pub(crate) model: String,
    pub(crate) level: LevelFilter,
    pub(crate) timeout: Duration,
    /// 两次模型请求之间的最小间隔。有些服务商按分钟限速（Kimi 实测连续第 4 个请求
    /// 就 429），冒烟一趟要发 4 次；这是应用层的节流，内核不管这个。
    pub(crate) min_request_gap: Duration,
    pub(crate) wire_log: WireLogPolicy,
    pub(crate) chat_fields: JsonObject,
    pub(crate) chat_token_field: ChatTokenField,
    pub(crate) protocols: Vec<SmokeProtocol>,
    endpoints: Vec<(SmokeProtocol, String)>,
}

impl SmokeConfig {
    /// 按“命令行选择协议、环境变量提供运行配置”的规则读取当前进程。
    pub(crate) fn from_process() -> Result<Self, io::Error> {
        let selection = parse_selection(env::args().skip(1))?;
        let protocols = select_protocols(&selection)?;
        let level = parse_level(env::var("AGENT_LOG").as_deref().unwrap_or("info"))?;
        let wire_log = parse_wire_log(env::var("MODEL_WIRE_LOG").as_deref().unwrap_or("off"))?;
        let api_key = read_required_env("MODEL_API_KEY")?;
        let model = read_required_env("MODEL_NAME")?;
        let endpoints = parse_selected_endpoints(&protocols, |protocol| {
            read_required_env(endpoint_variable(protocol))
        })?;
        let (chat_fields, chat_token_field) = if protocols.contains(&SmokeProtocol::OpenAiChat) {
            parse_chat_configuration(
                read_optional_env("MODEL_CHAT_FIELDS_JSON")?.as_deref(),
                read_optional_env("MODEL_CHAT_TOKEN_FIELD")?.as_deref(),
            )?
        } else {
            (JsonObject::new(), ChatTokenField::default())
        };
        let timeout_secs = match env::var("MODEL_TIMEOUT_SECS") {
            Ok(value) => parse_timeout(&value)?,
            Err(env::VarError::NotPresent) => DEFAULT_TIMEOUT_SECS,
            Err(env::VarError::NotUnicode(_)) => {
                return Err(invalid_input("MODEL_TIMEOUT_SECS 必须是有效的 UTF-8 整数"));
            }
        };

        let min_request_gap = match env::var("MODEL_MIN_GAP_MS") {
            Ok(value) => Duration::from_millis(parse_millis(&value)?),
            Err(env::VarError::NotPresent) => Duration::ZERO,
            Err(env::VarError::NotUnicode(_)) => {
                return Err(invalid_input("MODEL_MIN_GAP_MS 必须是有效的 UTF-8 整数"));
            }
        };

        Ok(Self {
            api_key,
            model,
            level,
            timeout: Duration::from_secs(timeout_secs),
            min_request_gap,
            wire_log,
            chat_fields,
            chat_token_field,
            protocols,
            endpoints,
        })
    }

    /// 返回指定协议对应的完整 HTTP endpoint。
    pub(crate) fn endpoint(&self, protocol: SmokeProtocol) -> &str {
        self.endpoints
            .iter()
            .find_map(|(candidate, endpoint)| (*candidate == protocol).then_some(endpoint.as_str()))
            .expect("SmokeConfig 只会保存已完整校验的选中协议")
    }
}

fn parse_selection(arguments: impl IntoIterator<Item = String>) -> Result<String, io::Error> {
    let mut arguments = arguments.into_iter();
    let selection = arguments.next().unwrap_or_else(|| "chat".to_owned());
    if arguments.next().is_some() {
        return Err(invalid_input(format!("用法：{USAGE}")));
    }
    Ok(selection)
}

fn select_protocols(value: &str) -> Result<Vec<SmokeProtocol>, io::Error> {
    match value.to_ascii_lowercase().as_str() {
        "chat" => Ok(vec![SmokeProtocol::OpenAiChat]),
        "responses" => Ok(vec![SmokeProtocol::OpenAiResponses]),
        "anthropic" => Ok(vec![SmokeProtocol::AnthropicMessages]),
        "all" => Ok(vec![
            SmokeProtocol::OpenAiChat,
            SmokeProtocol::OpenAiResponses,
            SmokeProtocol::AnthropicMessages,
        ]),
        value => Err(invalid_input(format!(
            "未知协议：{value}；用法：chat|responses|anthropic|all"
        ))),
    }
}

fn endpoint_variable(protocol: SmokeProtocol) -> &'static str {
    match protocol {
        SmokeProtocol::OpenAiChat => "MODEL_CHAT_ENDPOINT",
        SmokeProtocol::OpenAiResponses => "MODEL_RESPONSES_ENDPOINT",
        SmokeProtocol::AnthropicMessages => "MODEL_ANTHROPIC_ENDPOINT",
    }
}

/// 只读取并解析本次实际选择的协议，未选择的协议不要求配置 endpoint。
fn parse_selected_endpoints(
    protocols: &[SmokeProtocol],
    mut read: impl FnMut(SmokeProtocol) -> Result<String, io::Error>,
) -> Result<Vec<(SmokeProtocol, String)>, io::Error> {
    protocols
        .iter()
        .copied()
        .map(|protocol| {
            let variable = endpoint_variable(protocol);
            let endpoint = read(protocol)?;
            parse_endpoint(variable, &endpoint).map(|endpoint| (protocol, endpoint))
        })
        .collect()
}

fn read_required_env(name: &str) -> Result<String, io::Error> {
    match env::var(name) {
        Ok(value) if !value.trim().is_empty() => Ok(value),
        Ok(_) | Err(env::VarError::NotPresent) => Err(invalid_input(format!("请设置 {name}"))),
        Err(env::VarError::NotUnicode(_)) => {
            Err(invalid_input(format!("{name} 必须是有效的 UTF-8 字符串")))
        }
    }
}

fn read_optional_env(name: &str) -> Result<Option<String>, io::Error> {
    match env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(env::VarError::NotUnicode(_)) => {
            Err(invalid_input(format!("{name} 必须是有效的 UTF-8 字符串")))
        }
    }
}

fn parse_chat_configuration(
    fields_json: Option<&str>,
    token_field: Option<&str>,
) -> Result<(JsonObject, ChatTokenField), io::Error> {
    let fields = match fields_json {
        None => JsonObject::new(),
        Some(source) => serde_json::from_str::<Value>(source)
            .map_err(|_| invalid_input("MODEL_CHAT_FIELDS_JSON 必须是有效的 JSON object"))?
            .as_object()
            .cloned()
            .ok_or_else(|| invalid_input("MODEL_CHAT_FIELDS_JSON 必须是 JSON object"))?,
    };
    if fields.contains_key("max_tokens") || fields.contains_key("max_completion_tokens") {
        return Err(invalid_input(
            "MODEL_CHAT_FIELDS_JSON 不得设置 token 上限；请使用 MODEL_CHAT_TOKEN_FIELD",
        ));
    }

    let token_field = match token_field.unwrap_or("max_tokens") {
        "max_tokens" => ChatTokenField::MaxTokens,
        "max_completion_tokens" => ChatTokenField::MaxCompletionTokens,
        _ => {
            return Err(invalid_input(
                "MODEL_CHAT_TOKEN_FIELD 必须是 max_tokens 或 max_completion_tokens",
            ));
        }
    };
    Ok((fields, token_field))
}

fn parse_level(value: &str) -> Result<LevelFilter, io::Error> {
    match value.to_ascii_lowercase().as_str() {
        "off" => Ok(LevelFilter::Off),
        "error" => Ok(LevelFilter::Error),
        "warn" => Ok(LevelFilter::Warn),
        "info" => Ok(LevelFilter::Info),
        "debug" => Ok(LevelFilter::Debug),
        "trace" => Ok(LevelFilter::Trace),
        value => Err(invalid_input(format!(
            "未知 AGENT_LOG 级别：{value}；应为 off/error/warn/info/debug/trace"
        ))),
    }
}

fn parse_wire_log(value: &str) -> Result<WireLogPolicy, io::Error> {
    match value.to_ascii_lowercase().as_str() {
        "off" => Ok(WireLogPolicy::Off),
        _ => Err(invalid_input(
            "protocol_smoke 会使用真实凭据，MODEL_WIRE_LOG 必须为 off",
        )),
    }
}

fn parse_millis(value: &str) -> Result<u64, io::Error> {
    value
        .trim()
        .parse::<u64>()
        .map_err(|_| invalid_input("MODEL_MIN_GAP_MS 必须是非负整数（毫秒）"))
}

fn parse_timeout(value: &str) -> Result<u64, io::Error> {
    value
        .parse::<u64>()
        .ok()
        .filter(|seconds| *seconds > 0)
        .ok_or_else(|| invalid_input("MODEL_TIMEOUT_SECS 必须是大于 0 的整数"))
}

fn parse_endpoint(variable: &str, value: &str) -> Result<String, io::Error> {
    let parsed = reqwest::Url::parse(value)
        .map_err(|_| invalid_input(format!("{variable} 必须是有效的完整 HTTP(S) URL")))?;
    if !matches!(parsed.scheme(), "http" | "https")
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.fragment().is_some()
    {
        return Err(invalid_input(format!(
            "{variable} 只允许不含凭据和片段的完整 HTTP(S) URL"
        )));
    }
    Ok(parsed.to_string())
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_selection_and_log_configuration_are_explicit() {
        assert_eq!(
            select_protocols("all").unwrap(),
            vec![
                SmokeProtocol::OpenAiChat,
                SmokeProtocol::OpenAiResponses,
                SmokeProtocol::AnthropicMessages
            ]
        );
        assert!(select_protocols("unknown").is_err());
        assert_eq!(parse_level("trace").unwrap(), LevelFilter::Trace);
        assert!(parse_level("verbose").is_err());
        assert_eq!(parse_wire_log("off").unwrap(), WireLogPolicy::Off);
        assert!(parse_wire_log("full").is_err());
        assert!(parse_wire_log("body").is_err());
    }

    #[test]
    fn cli_defaults_to_chat_and_rejects_extra_arguments() {
        assert_eq!(parse_selection(Vec::new()).unwrap(), "chat");
        assert_eq!(
            parse_selection(vec!["responses".to_owned()]).unwrap(),
            "responses"
        );
        assert!(parse_selection(vec!["chat".to_owned(), "extra".to_owned()]).is_err());
    }

    #[test]
    fn timeout_must_be_positive() {
        assert_eq!(parse_timeout("90").unwrap(), 90);
        assert!(parse_timeout("0").is_err());
        assert!(parse_timeout("invalid").is_err());
    }

    #[test]
    fn chat_fields_must_be_an_object_and_cannot_override_token_limits() {
        let (fields, token_field) = parse_chat_configuration(
            Some(r#"{"thinking":{"type":"disabled"},"temperature":0}"#),
            None,
        )
        .unwrap();
        assert_eq!(fields["temperature"], 0);
        assert_eq!(fields["thinking"]["type"], "disabled");
        assert_eq!(token_field, ChatTokenField::MaxTokens);

        assert!(parse_chat_configuration(Some("[]"), None).is_err());
        assert!(parse_chat_configuration(Some("not-json"), None).is_err());
        assert!(parse_chat_configuration(Some(r#"{"max_tokens":4096}"#), None).is_err());
        assert!(parse_chat_configuration(
            Some(r#"{"max_completion_tokens":4096}"#),
            Some("max_completion_tokens")
        )
        .is_err());
    }

    #[test]
    fn chat_token_field_accepts_only_the_two_supported_wire_names() {
        assert_eq!(
            parse_chat_configuration(None, Some("max_completion_tokens"))
                .unwrap()
                .1,
            ChatTokenField::MaxCompletionTokens
        );
        assert!(parse_chat_configuration(None, Some("max_output_tokens")).is_err());
    }

    #[test]
    fn selected_endpoint_parser_does_not_require_unselected_protocols() {
        let endpoints = parse_selected_endpoints(&[SmokeProtocol::OpenAiResponses], |protocol| {
            assert_eq!(protocol, SmokeProtocol::OpenAiResponses);
            Ok("https://gateway.example/v1/responses?api-version=latest".to_owned())
        })
        .unwrap();

        assert_eq!(endpoints.len(), 1);
        assert_eq!(endpoints[0].0, SmokeProtocol::OpenAiResponses);
        assert_eq!(
            endpoints[0].1,
            "https://gateway.example/v1/responses?api-version=latest"
        );
    }

    #[test]
    fn every_selected_protocol_requires_a_valid_full_endpoint() {
        let protocols = [SmokeProtocol::OpenAiChat, SmokeProtocol::AnthropicMessages];
        let error = parse_selected_endpoints(&protocols, |protocol| match protocol {
            SmokeProtocol::OpenAiChat => {
                Ok("https://gateway.example/v1/chat/completions".to_owned())
            }
            SmokeProtocol::AnthropicMessages => Ok("not-a-url".to_owned()),
            SmokeProtocol::OpenAiResponses => unreachable!(),
        })
        .unwrap_err();

        assert!(error.to_string().contains("MODEL_ANTHROPIC_ENDPOINT"));
        assert!(parse_endpoint(
            "MODEL_CHAT_ENDPOINT",
            "https://name:secret@example.com/v1/chat/completions"
        )
        .is_err());
        assert!(parse_endpoint("MODEL_CHAT_ENDPOINT", "file:///tmp/socket").is_err());
    }
}
