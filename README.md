# midturn

一个提供方中立的 Rust agent 内核（库）。你给它四样东西——提示源、工具运行时、压缩策略、
模型适配器——它负责一轮里的一切：什么时候叫模型、工具调用什么时候交给你、新输入什么时候
进入模型、崩溃之后从哪里接着来。

它的中心论点只有一句：**模型的一轮不是原子的。** 工具调用一旦参数完整就可以交给运行时
（不用等模型说完），新输入在明确的边界进入模型（不用等这一轮结束）。内核不知道工具怎么跑：
并行、互斥、审批、限流全在你手里。

> `0.0.x`，公开 API 仍会调整，`publish = false`。

## 五分钟上手

`Cargo.toml`：

```toml
[dependencies]
# 本地路径（自己用 / 边改边用），或 git 依赖（不需要 crates.io）：
midturn = { path = "../midturn", features = ["openai"] }            # 或 anthropic / all-adapters
# midturn = { git = "https://github.com/spojchil/midturn", features = ["openai"] }
tokio = { version = "1", features = ["macros", "rt"] }
serde_json = "1"
```

下面是 `examples/quickstart.rs` 的精简版——一个工具、一个 OpenAI-compatible 端点、一句话进去、
一个终态出来。跑它：`AGENT_API_KEY=sk-... cargo run --example quickstart --features openai`。

```rust,ignore
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
    let api_key = std::env::var("AGENT_API_KEY")?;
    // 模型：一个 OpenAI-compatible Chat 端点。Responses / Anthropic 只是换 `Protocol`。
    let model = HttpModel::new(HttpModelConfig::new(
        "https://api.deepseek.com/chat/completions",
        api_key,
        "deepseek-v4-flash",
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

    match run.join().await {
        TurnOutcome::Completed { output, .. } => println!("{}", output.text_content()),
        TurnOutcome::Stopped { .. } => println!("被取消了"),
        TurnOutcome::Failed { stage, error, .. } => println!("失败于 {stage:?}: {error:?}"),
        other => println!("{other:?}"),
    }
    Ok(())
}
```

跑一次大概是这样：

```text
现在是 2026 年 8 月 15 日 01:17:29（UTC）。
```

四个端口里 `PromptSource` 有默认实现（空前缀），其余三个必须给。`Model` 只要求 `complete`
（一次性）；流式协议的适配器覆盖 `complete_stream`——`midturn::adapters::http::HttpModel`
三种协议都做了。

## 之后你会想做的四件事

**1. 运行期间继续说话。** `enqueue` 任何时候都收。会话在跑，内容会在它的下一个边界进入模型；
会话空闲，内容把它叫醒。三种交付语义：

| `MailboxInput::…` | 什么时候进模型 | 空闲时 |
|---|---|---|
| `next_model_request` | 下一次模型请求前 | 立刻开跑 |
| `when_idle` | 这一轮本该结束时（模型说完了才带上，再请求一次） | 立刻开跑 |
| `passive` | 搭下一次请求的便车，从不自己引发请求 | 收下，不开跑 |

改主意用 `reschedule(from, to)`；`cancel_run()` 取消当前运行；`set_auto_start(false)` 让内容只
排队不开跑。

**2. 让工具在模型还在说话时就开跑。** 实现 `ToolRuntime::begin_incremental`：返回一个
`IncrementalToolBatch`，模型每流出一个完整的调用，内核就 `submit` 给你；数组封口时
`calls_sealed`；模型成功说完时 `commit`；断流 / 取消时 `abort`——你要说清楚每个已交付的调用
是**确定没开始**还是**不知道**。结果一律通过 `ToolCallReporter::settled` 上报，上报返回 `Ok`
就是「内核已经把它落盘了」。`examples/chat.rs` 是一份提前执行的参考实现，`examples/lead-time.rs`
测过它能提前多少（批内每多一个调用约 +350ms）。

**3. 让会话活过重启。** 用 `AgentSession::open(session_id, store, …)` 接一个 `CheckpointStore`
（`InMemoryCheckpointStore` 测试用；`sqlite` feature 给你 `SqliteCheckpointStore`）。每一次状态
转移都以一次 CAS 收尾；进程死在任何地方，重开后：

- 没有半截运行 → 信箱有内容就自己开跑（不想它自己动，`SessionConfig { auto_start: false, .. }`）；
- 有半截运行 → `enqueue` 得到 `Held(ResumeRequired)`，你显式 `try_resume()`；
- 有结果不明的副作用 → `enqueue` 得到 `Held(ReconciliationRequired)`；读
  `pending_reconciliation()`，向外部权威（幂等记录、工具系统、人）查清楚，`reconcile_effect()`；
  最后一次对账清掉阻塞就自动接着跑。

内核**从不**把结果未知的调用当成可以安全重投的；它能保证的是「已经发生的事实一定有一个
下一轮看得见的出口」——已结算的结果与未知的槽都会以中断回执写进对话。exactly-once 它给不了，
外部幂等键还是你的事。

**4. 看见发生了什么。** 三个观察端：`with_observer`（不带正文的事件：开跑、边界排空、模型请求、
工具批次、压缩、终态）、`with_content_observer`（完整对话、模型输出、工具结果——不装就不构造）、
`with_stream_observer`（模型增量）。三条流共用一个序号，按序号归并就是完整的时间线。都得在
第一次异步调用之前装。

## 内部形状（读代码前知道这些就够）

- **单写者。** 会话的全部状态在一个驱动循环任务的局部变量里；改它的只有纯函数
  `step(&mut Machine, Command) -> Vec<Effect>`。公共方法、模型任务、工具运行时、观察端都只是
  往一条命令通道里发消息、等回复。
- **循环不 await 用户代码。** 模型请求、`begin_incremental`、每一次 `submit / seal / commit / abort`、
  `dispatch`、`compact` 全部 spawn 出去，完成后以命令回流；观察端回调在另一个投递任务上。
  唯一的例外是 `CheckpointStore`：它是每次转移的串行化点，状态没落盘之前不该有人看到它。
- **`Persist` 是 effect，排在第一。** 依赖持久化的回复（`Reply(Started)`、`Runtime::Submit`、
  上报的 `Ok`）都排在它之后；落盘失败就隔离这个实例，重开以 store 为准。
- **「未返回的工具列表」不是内存里的一份表**，是账本里当前尝试下仍 `InFlight` 的记录，每次现算。
  已结算的事实与结果未知的槽，在运行怎么收口之前都会以中断回执写进对话。
- **崩溃点表**在 `src/session/tests/crash/`：进程死在任意一次 `Persist` 之后，重开看到什么，
  每一行一条用例。

## 端口一览

| 端口 | 你实现什么 | 内核承诺什么 |
|---|---|---|
| `PromptSource` | `base_context()`：受保护前缀 | 每次开跑重读；不写进对话、不参与压缩 |
| `Model` | `complete` / `complete_stream` | 一次尝试一个 `ModelRequest`；流事件按到达顺序进入状态机 |
| `ToolRuntime` | `definitions` / `dispatch` / 可选 `begin_incremental` | 整批只交一次；增量交付前先落盘；结果按调用 ID 配对、按槽序进对话 |
| `Compaction` | `compact(conversation)` | 只在安全边界调；耐久事实必须逐项原样保留（用 `durable_fact_kind` 认），否则整份结果丢弃 |
| `CheckpointStore` | `load` / `compare_and_swap` | 是每次转移的串行化点；不得回调会话；`Err` / panic 即隔离，重开为准 |

## Cargo features

| Feature | 作用 |
| --- | --- |
| `default` | 内核、会话、事件、内存持久化；不带网络或数据库依赖 |
| `openai` | OpenAI Chat Completions 与 Responses adapter（自动带 `http`） |
| `anthropic` | Anthropic Messages adapter |
| `all-adapters` | 两者都要 |
| `sqlite` | bundled SQLite checkpoint backend |

## 示例

| 例子 | 干什么 | 怎么跑 |
|---|---|---|
| `quickstart` | 上面那段 | `AGENT_API_KEY=… cargo run --example quickstart --features openai` |
| `chat` | 交互式对话；提前执行的增量工具运行时参考实现；运行期间可插话 | `AGENT_API_KEY=… cargo run --example chat --features openai` |
| `protocol_smoke` | 三种协议同一套场景冒烟（`chat` / `responses` / `anthropic` / `all`） | 见文件头；`MODEL_WIRE_LOG` 必须为 `off` |
| `lead-time` | 实测工具调用交到运行时手上比整份响应早多少 | 见文件头 |

都会真的调用服务商、产生费用。密钥只从进程环境注入，不要写进仓库、命令参数或日志。

## 目录

- `src/run.rs`：无 I/O 回合状态机 `Turn`（对话闭合规则、请求边界、工具轮配对）。
- `src/session/mod.rs`：公开 `AgentSession`——一个 `Sender<Command>` 的薄壳，公共方法都是
  「发一条命令、等回复」。
- `src/session/engine.rs`：驱动循环，状态的唯一持有者；只 await 命令通道，所有端口调用都
  spawn 出去再以命令回流；`Persist` 失败即隔离。
- `src/session/machine/`：纯函数 `step(&mut Machine, Command) -> Vec<Effect>`——
  `mailbox.rs`（投递、改期、触发）、`run.rs`（开跑、推进、压缩、收尾）、`model.rs`（模型流事件）、
  `tools/`（进入工具阶段与收口、逐槽交付与结算、中止分类与超时、整批 dispatch）、
  `recovery.rs`（重开、续跑、对账）、`snapshot.rs`（checkpoint 投影）。
- `src/session/tests/`：端到端用例按主题分目录；崩溃点表在 `crash/`。
- `src/persistence.rs` + `persistence/{store,sqlite}.rs`：checkpoint、运行快照、副作用账本、store。
- `src/adapters/`：可选的 provider wire adapter。

## 开发

需要 Rust 1.86+（`rust-toolchain.toml` 用 stable，带 `rustfmt` 与 `clippy`）。提交前：

```sh
cargo test
cargo test --features all-adapters,sqlite
cargo clippy --all-targets --features all-adapters,sqlite -- -D warnings
cargo fmt --check
```
