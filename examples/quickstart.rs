//! 最小可跑的例子：一个工具、一个 OpenAI-compatible 端点、一句话进去、一个终态出来。
//!
//! ```text
//! AGENT_API_KEY=sk-... cargo run --example quickstart --features openai
//! ```
//!
//! 可选：`AGENT_ENDPOINT`（默认 DeepSeek 的 chat/completions）、`AGENT_MODEL`。
//! **它会真的调用服务商并产生费用。**

use std::sync::Arc;

use midturn::adapters::http::{HttpModel, HttpModelConfig, Protocol};
use midturn::{
    AgentError, AgentSession, Compaction, Enqueued, InputMessage, MailboxInput, PortFuture,
    PromptSource, SessionConfig, ToolCallBatch, ToolDefinition, ToolResult, ToolResultBatch,
    ToolRuntime, TranscriptItem, TurnOutcome,
};
use serde_json::json;

/// 提示源：只需要给出受保护的前缀（系统提示）。不想给就连这个都可以不实现。
struct Prompt;

impl PromptSource for Prompt {
    fn base_context(&self) -> Result<Vec<TranscriptItem>, AgentError> {
        Ok(vec![InputMessage::text(
            "system",
            "你是一个会用工具的助手。需要当前时间就调用 now，不要猜。",
        )
        .into()])
    }
}

/// 工具运行时：内核只知道工具的声明和流经的数据，怎么跑全在这里。
struct Tools;

impl ToolRuntime for Tools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        let mut now = ToolDefinition::new("now", json!({"type": "object", "properties": {}}));
        now.description = Some("返回当前的 UTC 时间戳（秒）。".to_owned());
        vec![now]
    }

    /// 模型说完之后，整批调用一次交过来；按调用 ID 把结果配回去。
    fn dispatch<'a>(
        &'a self,
        batch: ToolCallBatch,
    ) -> PortFuture<'a, Result<ToolResultBatch, AgentError>> {
        Box::pin(async move {
            let results = batch
                .calls
                .into_iter()
                .map(|call| match call.name.as_str() {
                    "now" => {
                        let secs = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_secs())
                            .unwrap_or(0);
                        ToolResult::success_json(call.id, json!({"unix_seconds": secs}))
                    }
                    other => ToolResult::failure(call.id, format!("unknown tool: {other}")),
                })
                .collect();
            Ok(ToolResultBatch { results })
        })
    }
}

/// 压缩策略：这个例子不压。真实应用在这里概括历史（耐久事实必须原样保留）。
struct KeepEverything;

impl Compaction for KeepEverything {
    fn compact<'a>(
        &'a self,
        conversation: &'a [TranscriptItem],
    ) -> PortFuture<'a, Result<Vec<TranscriptItem>, AgentError>> {
        Box::pin(async move { Ok(conversation.to_vec()) })
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("AGENT_API_KEY")
        .map_err(|_| "请设置 AGENT_API_KEY；这个示例会真的调用服务商并产生费用")?;
    let endpoint = std::env::var("AGENT_ENDPOINT")
        .unwrap_or_else(|_| "https://api.deepseek.com/chat/completions".to_owned());
    let model_name =
        std::env::var("AGENT_MODEL").unwrap_or_else(|_| "deepseek-v4-flash".to_owned());

    // 模型：一个 OpenAI-compatible Chat 端点。Responses / Anthropic 只是换 `Protocol`。
    let model = HttpModel::new(HttpModelConfig::new(
        endpoint,
        api_key,
        model_name,
        Protocol::openai_chat(),
    ))?;

    let session = Arc::new(AgentSession::new(
        Arc::new(Prompt),
        Arc::new(Tools),
        Arc::new(KeepEverything),
        Arc::new(model),
        SessionConfig::default(),
    ));

    // 投递一句话。会话空闲 → 这句话把它叫醒 → 拿到这一轮的句柄。
    let Enqueued::Started(run) = session
        .enqueue(MailboxInput::next_model_request(vec![InputMessage::text(
            "user",
            "现在几点了？用工具查，然后用一句话回答。",
        )
        .into()]))
        .await?
    else {
        unreachable!("空闲会话收到触发内容一定开跑");
    };

    // 运行期间还能继续投递：这些内容会在下一个边界进入模型（这里不需要）。
    match run.join().await {
        TurnOutcome::Completed { output, usage, .. } => {
            println!("{}", output.text_content());
            println!("usage: {usage:?}");
        }
        TurnOutcome::Stopped { .. } => println!("被取消了"),
        TurnOutcome::Failed { stage, error, .. } => println!("失败于 {stage:?}: {error:?}"),
        other => println!("{other:?}"),
    }
    Ok(())
}
