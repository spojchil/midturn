//! 测量「出向提前量」：工具调用交到运行时手上，比整份响应聚合完成早多久。
//!
//! 这是内核核心论点的量化。单个调用的提前量很小（实测 DeepSeek Chat 约 58ms），
//! 说明不了什么。真正该拿出来的数字是**一批里第一个调用与最后一个调用的间隔**——
//! 聚合后执行的实现必须等到最后一个才能开始，所以那个间隔就是白白空等的时间，
//! 而且随批内调用数增长。
//!
//! 跑法：
//!
//! ```text
//! LEAD_API_KEY=sk-...  LEAD_ENDPOINT=https://api.deepseek.com/chat/completions \
//! LEAD_MODEL=deepseek-v4-flash  LEAD_PROTOCOL=chat  LEAD_CALLS=6 \
//!   cargo run --release --example lead-time --features openai,anthropic
//! ```
//!
//! `LEAD_PROTOCOL` 取 `chat` | `responses` | `anthropic`；`LEAD_ROUNDS` 控制重复次数
//! （默认 3，取中位数更稳）。**这个示例会真的调用服务商并产生费用。**

use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::time::{Duration, Instant};

use midturn::adapters::http::{HttpModel, HttpModelConfig, Protocol};
use midturn::{
    AgentError, AgentSession, Compaction, Enqueued, InputMessage, MailboxInput,
    ModelStreamObservation, ObservedModelStreamEvent, PortFuture, SessionConfig, StreamObserver,
    ToolCallBatch, ToolDefinition, ToolResult, ToolResultBatch, ToolRuntime, TranscriptItem,
};
use serde_json::json;

/// 一次响应里，各个关键时刻相对请求发出时刻的毫秒数。
#[derive(Default)]
struct Marks {
    started_at: Option<Instant>,
    /// 每个 `ToolCallReady` 到达的时刻，按到达顺序。
    ready: Vec<f64>,
    sealed: Option<f64>,
    committed: Option<f64>,
}

#[derive(Default)]
struct Timeline(StdMutex<Marks>);

impl Timeline {
    fn begin(&self) {
        let mut marks = self.0.lock().unwrap();
        *marks = Marks {
            started_at: Some(Instant::now()),
            ..Marks::default()
        };
    }

    fn elapsed(marks: &Marks) -> f64 {
        marks
            .started_at
            .map(|start| start.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0)
    }

    fn take(&self) -> Marks {
        std::mem::take(&mut *self.0.lock().unwrap())
    }
}

impl StreamObserver for Timeline {
    fn observe(&self, event: &ObservedModelStreamEvent) {
        let mut marks = self.0.lock().unwrap();
        // 工具结果回去之后还有第二次请求。只采第一份响应的时间线，否则它的聚合时刻
        // 会把第一份的覆盖掉，量出来的提前量是错的。
        if marks.committed.is_some() {
            return;
        }
        let at = Self::elapsed(&marks);
        match &event.payload {
            ModelStreamObservation::Delta(midturn::ModelStreamEvent::ToolCallReady { .. }) => {
                marks.ready.push(at);
            }
            ModelStreamObservation::Delta(midturn::ModelStreamEvent::ToolCallsSealed {
                ..
            }) => {
                marks.sealed = Some(at);
            }
            // 聚合完成 —— 聚合后执行的实现在这一刻才拿到第一个调用。
            ModelStreamObservation::AttemptCommitted => {
                marks.committed = Some(at);
            }
            _ => {}
        }
    }
}

struct Tools;

impl midturn::PromptSource for Tools {}

impl ToolRuntime for Tools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition::new(
            "record",
            json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "note": {"type": "string"}
                },
                "required": ["city", "note"],
                "additionalProperties": false
            }),
        )]
    }

    /// 工具本身必须瞬时返回：这里测的是交付时机，不是执行耗时。
    fn dispatch<'a>(
        &'a self,
        batch: ToolCallBatch,
    ) -> PortFuture<'a, Result<ToolResultBatch, AgentError>> {
        Box::pin(async move {
            Ok(ToolResultBatch {
                results: batch
                    .calls
                    .into_iter()
                    .map(|call| ToolResult::success_json(call.id, json!({"ok": true})))
                    .collect(),
            })
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

fn env(name: &str) -> Result<String, Box<dyn std::error::Error>> {
    std::env::var(name).map_err(|_| format!("请设置 {name}").into())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = env("LEAD_API_KEY")?;
    let endpoint = env("LEAD_ENDPOINT")?;
    let model = env("LEAD_MODEL")?;
    let protocol = std::env::var("LEAD_PROTOCOL").unwrap_or_else(|_| "chat".to_owned());
    let call_count: usize = std::env::var("LEAD_CALLS")
        .unwrap_or_else(|_| "6".to_owned())
        .parse()?;
    let rounds: usize = std::env::var("LEAD_ROUNDS")
        .unwrap_or_else(|_| "3".to_owned())
        .parse()?;

    if !matches!(protocol.as_str(), "chat" | "responses" | "anthropic") {
        return Err(format!("未知协议 {protocol}；取 chat|responses|anthropic").into());
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async move {
        measure(api_key, endpoint, model, protocol, call_count, rounds).await
    })
}

fn wire_protocol(label: &str) -> Protocol {
    match label {
        "responses" => Protocol::openai_responses(),
        "anthropic" => Protocol::anthropic_messages(),
        _ => Protocol::openai_chat(),
    }
}

async fn measure(
    api_key: String,
    endpoint: String,
    model: String,
    protocol_label: String,
    call_count: usize,
    rounds: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let timeline = Arc::new(Timeline::default());

    println!("协议 {protocol_label} / 模型 {model} / 目标调用数 {call_count} / 轮次 {rounds}\n");

    let mut first_leads = Vec::new();
    let mut spreads = Vec::new();
    let mut observed_counts = Vec::new();

    for round in 1..=rounds {
        let session = Arc::new(
            AgentSession::new(
                Arc::new(Tools),
                Arc::new(Tools),
                Arc::new(KeepEverything),
                Arc::new(HttpModel::new(
                    HttpModelConfig::new(
                        endpoint.clone(),
                        api_key.clone(),
                        model.clone(),
                        wire_protocol(&protocol_label),
                    )
                    .with_timeout(Duration::from_secs(180)),
                )?),
                SessionConfig::default(),
            )
            .with_stream_observer(timeline.clone()),
        );

        let prompt = format!(
            "请在同一条回复里连续调用 record 工具 {call_count} 次，\
             city 依次用北京、上海、广州、深圳、杭州、成都（不够就继续编），\
             note 每次写一句不同的、至少二十个字的说明。现在不要输出正文。"
        );
        timeline.begin();
        let Enqueued::Started(handle) = session
            .enqueue(MailboxInput::next_model_request(vec![
                TranscriptItem::from(InputMessage::text("user", prompt)),
            ]))
            .await?
        else {
            return Err("空闲会话应当因这次投递开跑".into());
        };
        // 只关心第一次响应的时间线；跑完这一轮即可。
        let _ = handle.join().await;
        session.set_auto_start(false).await;
        session.wait_until_idle().await;

        let marks = timeline.take();
        let Some(committed) = marks.committed else {
            println!("  第 {round} 轮：没有观察到聚合完成，跳过");
            continue;
        };
        if marks.ready.is_empty() {
            println!("  第 {round} 轮：这一轮没有工具调用，跳过");
            continue;
        }
        let first = marks.ready[0];
        let last = *marks.ready.last().unwrap();
        let first_lead = committed - first;
        let spread = last - first;
        first_leads.push(first_lead);
        spreads.push(spread);
        observed_counts.push(marks.ready.len());

        println!(
            "  第 {round} 轮：{} 个调用｜首个 {first:.0}ms｜末个 {last:.0}ms｜封口 {}｜聚合 {committed:.0}ms",
            marks.ready.len(),
            marks
                .sealed
                .map(|value| format!("{value:.0}ms"))
                .unwrap_or_else(|| "—".to_owned()),
        );
        println!("      首个调用的提前量 {first_lead:.0}ms｜首末间隔 {spread:.0}ms");
    }

    if first_leads.is_empty() {
        println!("\n没有采到有效数据。");
        return Ok(());
    }

    println!("\n== {protocol_label} 汇总 ==");
    println!("  实际调用数：{observed_counts:?}");
    println!(
        "  首个调用提前量中位数：{:.0}ms",
        median(&mut first_leads.clone())
    );
    println!("  首末间隔中位数：{:.0}ms", median(&mut spreads.clone()));
    println!(
        "\n  解读：聚合后才执行的实现，最早也只能在聚合完成那一刻开始跑第一个调用；\n  \
         本内核在首个调用到达时就能交出去，因此白白空等的时间就是上面的提前量。\n  \
         首末间隔说明这个差距随批内调用数增长——它才是该拿出来的数字。"
    );
    Ok(())
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).expect("时间戳不会是 NaN"));
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}
