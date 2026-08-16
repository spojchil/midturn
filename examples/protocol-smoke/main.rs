//! 用同一套场景验证 OpenAI Chat、OpenAI Responses 与 Anthropic Messages 兼容入口。
//!
//! `MODEL_API_KEY` 仅在真正运行示例时读取，测试与编译不会访问它。模型名、完整
//! endpoint 与日志开关由 `config` 模块统一解析，wire 协议由 crate 的公共 `adapters`
//! 模块实现。
//!
//! OpenAI-compatible Chat 可用 `MODEL_CHAT_FIELDS_JSON` 传入一个 JSON object，
//! 并用 `MODEL_CHAT_TOKEN_FIELD=max_tokens|max_completion_tokens` 选择兼容端点
//! 接收的字段名。示例自己控制 token 上限，且禁止 wire body 日志。

mod config;
mod harness;

use std::error::Error;

use config::SmokeConfig;
use harness::run_smoke;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let config = SmokeConfig::from_process()?;

    for protocol in config.protocols.iter().copied() {
        run_smoke(protocol, &config).await?;
    }
    Ok(())
}
