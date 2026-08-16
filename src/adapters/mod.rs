//! 可选的服务商 wire 协议适配器。
//!
//! HTTP 传输、鉴权和日志策略由 [`http`] 共享；具体请求与响应格式按协议分别实现。

pub mod http;

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(all(test, any(feature = "openai", feature = "anthropic")))]
mod tests;
