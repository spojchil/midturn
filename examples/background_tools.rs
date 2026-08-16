//! 后台任务：工具立刻返回一个句柄，真正的结果稍后从信箱回来——「不等了」不需要内核多做任何事。
//!
//! ```text
//! AGENT_API_KEY=sk-... cargo run --example background_tools --features openai
//! ```
//!
//! 可选：`AGENT_ENDPOINT`、`AGENT_MODEL`。**它会真的调用服务商并产生费用。**
//!
//! 形状：
//!
//! 1. 工具 `start_job` 的语义是「启动一个后台任务」，它返回 `{"job": 1, "state": "running"}`
//!    就是一个**真实的**结果——任务确实启动了。这一轮不被卡住。
//! 2. 运行时自己持有任务表；任务跑完，运行时把结果作为一条 `MailboxInput` 投回会话：
//!    要模型立刻处理就 `next_model_request`（空闲时会把会话叫醒），只想搭便车就 `passive`。
//! 3. `list_jobs` / `cancel_job` 是运行时自己的普通工具：任务表在它手里，列表不会漂移。
//!
//! 内核始终只看到「一次成功的工具调用」和「一条新输入」，不知道也不需要知道后台任务的存在。

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use midturn::adapters::http::{HttpModel, HttpModelConfig, Protocol};
use midturn::{
    AgentError, AgentSession, Compaction, Enqueued, InputMessage, MailboxInput, PortFuture,
    PromptSource, SessionConfig, ToolCallBatch, ToolDefinition, ToolResult, ToolResultBatch,
    ToolRuntime, TranscriptItem, TurnOutcome,
};
use serde_json::json;

struct Prompt;

impl PromptSource for Prompt {
    fn base_context(&self) -> Result<Vec<TranscriptItem>, AgentError> {
        Ok(vec![InputMessage::text(
            "system",
            "你是一个会用工具的助手。start_job 会立刻返回一个任务句柄，任务在后台跑；\
             结果出来后会以一条 tool 角色的消息送到你面前。启动任务之后不要猜结果，\
             直接告诉用户任务已经开始并等待。",
        )
        .into()])
    }
}

#[derive(Clone, Copy, Debug)]
enum JobState {
    Running,
    Done,
    Cancelled,
}

/// 运行时自己的任务表。内核不知道它。
struct Jobs {
    next_id: AtomicU64,
    table: Mutex<BTreeMap<u64, (String, JobState)>>,
    /// 用来把结果投回会话。用 `Weak`：会话没了就没人要结果了。
    session: Mutex<Weak<AgentSession>>,
}

impl Jobs {
    fn definitions() -> Vec<ToolDefinition> {
        let mut start = ToolDefinition::new(
            "start_job",
            json!({"type": "object", "properties": {"task": {"type": "string"}, "seconds": {"type": "integer"}}, "required": ["task"]}),
        );
        start.description = Some(
            "启动一个后台任务并立刻返回句柄；结果稍后会作为新消息送达。seconds 是任务耗时（默认 2）。"
                .to_owned(),
        );
        let mut list =
            ToolDefinition::new("list_jobs", json!({"type": "object", "properties": {}}));
        list.description = Some("列出所有后台任务及其状态。".to_owned());
        let mut cancel = ToolDefinition::new(
            "cancel_job",
            json!({"type": "object", "properties": {"job": {"type": "integer"}}, "required": ["job"]}),
        );
        cancel.description = Some("取消一个还在跑的后台任务。".to_owned());
        vec![start, list, cancel]
    }

    fn start(self: &Arc<Self>, task: String, seconds: u64) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst) + 1;
        self.table
            .lock()
            .unwrap()
            .insert(id, (task.clone(), JobState::Running));
        let jobs = Arc::clone(self);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(seconds)).await;
            {
                let mut table = jobs.table.lock().unwrap();
                match table.get_mut(&id) {
                    Some((_, state @ JobState::Running)) => *state = JobState::Done,
                    _ => return, // 已被取消
                }
            }
            // 结果出来了：作为一条新输入投回会话。会话空闲就会被叫醒，在跑就在下一个边界进入。
            let Some(session) = jobs.session.lock().unwrap().upgrade() else {
                return;
            };
            let _ = session
                .enqueue(MailboxInput::next_model_request(vec![InputMessage::new(
                    "user".to_owned(),
                    vec![midturn::ContentPart::json(json!({
                        "kind": "job_finished",
                        "job": id,
                        "task": task,
                        "result": format!("任务「{task}」完成，用了 {seconds} 秒"),
                    }))],
                )
                .into()]))
                .await;
        });
        id
    }
}

/// `ToolRuntime` 实现：一个握着任务表的薄壳。
struct JobRuntime(Arc<Jobs>);

impl ToolRuntime for JobRuntime {
    fn definitions(&self) -> Vec<ToolDefinition> {
        Jobs::definitions()
    }

    fn dispatch<'a>(
        &'a self,
        batch: ToolCallBatch,
    ) -> PortFuture<'a, Result<ToolResultBatch, AgentError>> {
        Box::pin(async move {
            let results = batch
                .calls
                .into_iter()
                .map(|call| match call.name.as_str() {
                    "start_job" => {
                        let task = call.arguments["task"].as_str().unwrap_or("unnamed").to_owned();
                        let seconds = call.arguments["seconds"].as_u64().unwrap_or(2).min(30);
                        let id = self.0.start(task, seconds);
                        // 真实的结果：任务已启动。不是占位符，不是谎。
                        ToolResult::success_json(call.id, json!({"job": id, "state": "running"}))
                    }
                    "list_jobs" => {
                        let table = self.0.table.lock().unwrap();
                        let jobs: Vec<_> = table
                            .iter()
                            .map(|(id, (task, state))| json!({"job": id, "task": task, "state": format!("{state:?}")}))
                            .collect();
                        ToolResult::success_json(call.id, json!({"jobs": jobs}))
                    }
                    "cancel_job" => {
                        let id = call.arguments["job"].as_u64().unwrap_or(0);
                        let mut table = self.0.table.lock().unwrap();
                        let outcome = match table.get_mut(&id) {
                            Some((_, state @ JobState::Running)) => {
                                *state = JobState::Cancelled;
                                "cancelled"
                            }
                            Some(_) => "already_finished",
                            None => "unknown_job",
                        };
                        ToolResult::success_json(call.id, json!({"job": id, "outcome": outcome}))
                    }
                    other => ToolResult::failure(call.id, format!("unknown tool: {other}")),
                })
                .collect();
            Ok(ToolResultBatch { results })
        })
    }
}

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
    let model = HttpModel::new(HttpModelConfig::new(
        endpoint,
        api_key,
        model_name,
        Protocol::openai_chat(),
    ))?;

    let jobs = Arc::new(Jobs {
        next_id: AtomicU64::new(0),
        table: Mutex::new(BTreeMap::new()),
        session: Mutex::new(Weak::new()),
    });
    let session = Arc::new(AgentSession::new(
        Arc::new(Prompt),
        Arc::new(JobRuntime(Arc::clone(&jobs))),
        Arc::new(KeepEverything),
        Arc::new(model),
        SessionConfig::default(),
    ));
    *jobs.session.lock().unwrap() = Arc::downgrade(&session);

    // 第一轮：模型启动任务，拿到句柄，这一轮就结束了——没有人在等那个任务。
    let Enqueued::Started(run) = session
        .enqueue(MailboxInput::next_model_request(vec![InputMessage::text(
            "user",
            "帮我启动一个叫「编译」的后台任务，大约 3 秒。启动后告诉我一声就行。",
        )
        .into()]))
        .await?
    else {
        unreachable!()
    };
    match run.join().await {
        TurnOutcome::Completed { output, .. } => println!("[第一轮] {}", output.text_content()),
        other => println!("[第一轮] {other:?}"),
    }

    // 任务在后台跑完，结果投进信箱，会话空闲 → 自动开跑第二轮：模型看到结果再说一句。
    println!("[等待后台任务……]");
    tokio::time::sleep(Duration::from_secs(4)).await;
    session.wait_until_idle().await;
    let conversation = session.conversation().await;
    match conversation.last() {
        Some(TranscriptItem::ModelOutput(output)) => println!("[第二轮] {}", output.text_content()),
        other => println!("[第二轮] 没有等到模型的回应：{other:?}"),
    }
    Ok(())
}
