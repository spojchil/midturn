use std::time::Duration;

use super::transport::Protocol;

/// 默认允许读取的最大响应正文长度；SSE 模式按整条流的累计字节数计算（8 MiB）。
pub const DEFAULT_MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

/// HTTP 请求使用的鉴权头。
///
/// `Headers` 会完整替代协议根据 API key 生成的便利默认值，适合使用兼容协议但采用自定义
/// 鉴权或额外版本头的端点。
#[derive(Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum HttpAuth {
    Bearer(String),
    ApiKeyHeader { name: String, value: String },
    Headers(Vec<(String, String)>),
    None,
}

impl HttpAuth {
    pub fn bearer(token: impl Into<String>) -> Self {
        Self::Bearer(token.into())
    }

    pub fn api_key_header(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self::ApiKeyHeader {
            name: name.into(),
            value: value.into(),
        }
    }

    pub fn headers(headers: impl IntoIterator<Item = (String, String)>) -> Self {
        Self::Headers(headers.into_iter().collect())
    }
}

/// 是否记录完整 wire JSON body 或逐条 SSE 事件。完整日志可能包含敏感对话内容，默认关闭。
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum WireLogPolicy {
    #[default]
    Off,
    Full,
}

/// 一个完整 HTTP 模型端点的配置。
///
/// `endpoint` 必须是包含 scheme、host 和协议路径的完整 HTTP(S) URL；本模块不会再拼接
/// 服务商路径。远端地址默认要求 HTTPS，明文 HTTP 仅自动允许 loopback。`api_key` 仅用于
/// 生成协议便利鉴权，可通过 [`Self::with_auth`] 完全覆盖。
#[derive(Clone)]
pub struct HttpModelConfig {
    pub(crate) endpoint: String,
    pub(crate) model: String,
    pub(crate) protocol: Protocol,
    pub(crate) auth: HttpAuth,
    pub(crate) timeout: Duration,
    pub(crate) wire_log: WireLogPolicy,
    pub(crate) allow_insecure_http: bool,
    pub(crate) max_response_bytes: usize,
}

impl HttpModelConfig {
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

    pub fn new(
        endpoint: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        protocol: Protocol,
    ) -> Self {
        let auth = protocol.default_auth(api_key.into());
        Self::new_with_auth(endpoint, model, protocol, auth)
    }

    /// 使用显式认证配置构造端点，不要求提供占位 API key。
    pub fn new_with_auth(
        endpoint: impl Into<String>,
        model: impl Into<String>,
        protocol: Protocol,
        auth: HttpAuth,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            model: model.into(),
            protocol,
            auth,
            timeout: Self::DEFAULT_TIMEOUT,
            wire_log: WireLogPolicy::Off,
            allow_insecure_http: false,
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
        }
    }

    /// 完整替换协议根据 API key 生成的便利鉴权头。
    pub fn with_auth(mut self, auth: HttpAuth) -> Self {
        self.auth = auth;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_wire_log(mut self, wire_log: WireLogPolicy) -> Self {
        self.wire_log = wire_log;
        self
    }

    /// 允许向非 loopback 的明文 HTTP endpoint 发送请求。默认关闭，因为请求通常携带
    /// API key、prompt 和工具数据。
    pub fn allow_insecure_http(mut self, allow: bool) -> Self {
        self.allow_insecure_http = allow;
        self
    }

    /// 设置单次响应正文上限。零值会在构造 [`super::HttpModel`] 时被拒绝。
    pub fn with_max_response_bytes(mut self, max_response_bytes: usize) -> Self {
        self.max_response_bytes = max_response_bytes;
        self
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn protocol(&self) -> &Protocol {
        &self.protocol
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    pub fn wire_log(&self) -> WireLogPolicy {
        self.wire_log
    }

    pub fn insecure_http_allowed(&self) -> bool {
        self.allow_insecure_http
    }

    pub fn max_response_bytes(&self) -> usize {
        self.max_response_bytes
    }
}
