//! `AgentSession` 的端到端用例：只走公共 API（外加 `Inspect` 窥视），按主题分文件。
//! 共享的测试替身与夹具都在这里。

mod compaction;
mod crash;
mod effects;
mod mailbox;
mod observability;
mod supervision;
mod tools;

mod support;
use support::models::*;
use support::runtimes::*;
use support::stores::*;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex as StdMutex;

use serde_json::{json, Value};
use tokio::sync::{mpsc, Mutex, Semaphore};

use super::*;
use crate::events::{
    AgentEvent, ContentEvent, EventMetadata, FilteredObserver, LevelFilter, ModelStreamObservation,
    NoopStreamObserver, ObservedModelStreamEvent,
};
use crate::mailbox::{Delivery, MailboxRejectedReason};
use crate::persistence::{EffectOutcome as DurableEffectOutcome, EffectState};
use crate::ports::{
    IncrementalToolBatch, ModelRequest, ModelResponse, ModelStreamEvent, ModelStreamSink,
    PortFuture, ToolCallReporter,
};
use crate::types::{
    AbortClassification, AgentErrorKind, ContentPart, DurableFactKind, IncrementalToolCall,
    InputMessage, ToolBatchAbortReason, ToolBatchStart, ToolCall, ToolCallBatch, ToolCallId,
    ToolCallSlot, ToolDefinition, ToolResult, ToolResultBatch,
};

/// 投递一条引导输入，并要求它把空闲会话叫醒；返回可等待的句柄。
async fn start(
    session: &Arc<AgentSession>,
    items: Vec<TranscriptItem>,
) -> Result<RunHandle, MailboxRejected> {
    match session
        .enqueue(MailboxInput::next_model_request(items))
        .await?
    {
        Enqueued::Started(handle) => Ok(handle),
        other => panic!("空闲会话收到引导输入后应当开跑，实际是 {other:?}"),
    }
}

/// 起一轮并等到终态。
async fn run_once(
    session: &Arc<AgentSession>,
    items: Vec<TranscriptItem>,
) -> Result<TurnOutcome, MailboxRejected> {
    Ok(start(session, items).await?.join().await)
}

/// 投递，不关心是否恰好开跑。
async fn post(session: &Arc<AgentSession>, input: MailboxInput) -> Result<(), MailboxRejected> {
    session.enqueue(input).await.map(|_| ())
}

fn input(role: &str, text: &str) -> TranscriptItem {
    InputMessage::text(role, text).into()
}

fn dangling_tool_call(id: &str) -> TranscriptItem {
    TranscriptItem::ModelOutput(ModelOutput::calls(vec![ToolCall::new(
        id,
        "read",
        json!({"path": "injected"}),
    )]))
}

fn complete_tool_round(id: &str) -> Vec<TranscriptItem> {
    vec![
        dangling_tool_call(id),
        TranscriptItem::ToolResults(ToolResultBatch {
            results: vec![ToolResult::success_json(
                ToolCallId::new(id),
                json!({"ok": true}),
            )],
        }),
    ]
}

fn visible_texts(transcript: &[TranscriptItem]) -> Vec<String> {
    transcript
        .iter()
        .flat_map(|item| match item {
            TranscriptItem::Input(message) => message.content.iter(),
            TranscriptItem::ModelOutput(output) => output.content.iter(),
            TranscriptItem::ToolResults(_) => [].iter(),
        })
        .filter_map(|part| match part {
            ContentPart::Text { text } => Some(text.clone()),
            ContentPart::Json { .. } | ContentPart::Opaque { .. } => None,
        })
        .collect()
}

fn tool_call_ids(transcript: &[TranscriptItem]) -> Vec<String> {
    transcript
        .iter()
        .filter_map(|item| match item {
            TranscriptItem::ModelOutput(output) => Some(&output.tool_calls),
            TranscriptItem::Input(_) | TranscriptItem::ToolResults(_) => None,
        })
        .flatten()
        .map(|call| call.id.as_str().to_owned())
        .collect()
}

fn is_recovery_receipt(item: &TranscriptItem) -> bool {
    matches!(
        item,
        TranscriptItem::Input(message)
            if message.content.iter().any(|part| matches!(
                part,
                ContentPart::Json { value }
                    if value.get("kind").and_then(Value::as_str)
                        == Some(DurableFactKind::InterruptedToolBatch.as_wire())
            ))
    )
}

fn has_json_fact_kind(item: &TranscriptItem, expected: &str) -> bool {
    matches!(
        item,
        TranscriptItem::Input(message)
            if message.content.iter().any(|part| matches!(
                part,
                ContentPart::Json { value }
                    if value.get("kind").and_then(Value::as_str) == Some(expected)
            ))
    )
}

async fn reconcile_all_as_cancelled(session: &AgentSession) -> usize {
    let pending = session.pending_reconciliation().await;
    assert!(!pending.is_empty(), "测试场景应留下必须人工对账的 effect");
    for action in &pending {
        let EffectRecoveryAction::Reconcile { id, .. } = action else {
            panic!("pending_reconciliation 只能返回 Reconcile 动作")
        };
        session
            .reconcile_effect(id, DurableEffectOutcome::CancelledBeforeStart)
            .await
            .unwrap();
    }
    pending.len()
}

struct Fixture {
    session: Arc<AgentSession>,
    requests: mpsc::UnboundedReceiver<ModelRequest>,
    responses: mpsc::UnboundedSender<Result<ModelResponse, AgentError>>,
    batches: mpsc::UnboundedReceiver<ToolCallBatch>,
    gate: Arc<Semaphore>,
    runtime: Arc<GatedBatchRuntime>,
}

struct StreamingFixture {
    session: Arc<AgentSession>,
    requests: mpsc::UnboundedReceiver<ModelRequest>,
    commands: mpsc::UnboundedSender<StreamCommand>,
    updates: mpsc::UnboundedReceiver<IncrementalUpdate>,
    runtime: Arc<EagerIncrementalRuntime>,
}

fn streaming_fixture(
    settled_slots: u32,
    fail_commit: bool,
    config: SessionConfig,
    compaction: Arc<dyn Compaction>,
    stream_observer: Arc<dyn StreamObserver>,
) -> StreamingFixture {
    let (request_tx, request_rx) = mpsc::unbounded_channel();
    let (command_tx, command_rx) = mpsc::unbounded_channel();
    let (update_tx, update_rx) = mpsc::unbounded_channel();
    let runtime = Arc::new(EagerIncrementalRuntime {
        fail_commit,
        ..EagerIncrementalRuntime::new(update_tx, settled_slots)
    });
    let session = Arc::new(
        AgentSession::new(
            Arc::new(StaticPrompt),
            runtime.clone(),
            compaction,
            Arc::new(CommandStreamModel {
                requests: request_tx,
                commands: Mutex::new(command_rx),
            }),
            config,
        )
        .with_stream_observer(stream_observer),
    );
    StreamingFixture {
        session,
        requests: request_rx,
        commands: command_tx,
        updates: update_rx,
        runtime,
    }
}

fn fixture() -> Fixture {
    fixture_with_config(SessionConfig::default())
}

fn fixture_with_config(config: SessionConfig) -> Fixture {
    let (request_tx, request_rx) = mpsc::unbounded_channel();
    let (response_tx, response_rx) = mpsc::unbounded_channel();
    let (batch_tx, batch_rx) = mpsc::unbounded_channel();
    let gate = Arc::new(Semaphore::new(0));
    let runtime = Arc::new(GatedBatchRuntime {
        batches: batch_tx,
        gate: gate.clone(),
        dispatches: AtomicUsize::new(0),
    });
    let session = Arc::new(AgentSession::new(
        Arc::new(StaticPrompt),
        runtime.clone(),
        Arc::new(NoCompaction),
        Arc::new(ChannelModel {
            requests: request_tx,
            responses: Mutex::new(response_rx),
        }),
        config,
    ));
    Fixture {
        session,
        requests: request_rx,
        responses: response_tx,
        batches: batch_rx,
        gate,
        runtime,
    }
}

fn model_response(output: ModelOutput) -> ModelResponse {
    ModelResponse {
        output,
        ..ModelResponse::default()
    }
}
