//! 协议适配器共用的 HTTP 配置与模型实现。

mod config;
mod sse;
mod transport;

pub use config::{HttpAuth, HttpModelConfig, WireLogPolicy, DEFAULT_MAX_RESPONSE_BYTES};
pub use transport::{HttpModel, Protocol};

#[cfg(any(feature = "openai", feature = "anthropic"))]
pub(crate) use sse::SseEvent;
#[cfg(feature = "openai")]
pub(crate) use transport::parse_arguments;
#[cfg(any(feature = "openai", feature = "anthropic"))]
pub(crate) use transport::{
    content_as_text, merge_request_fields, model_error, StreamDecode, WireCodec, WireStreamDecoder,
};
