//! 崩溃点表：进程死在某一次 `Persist` 之后、下一次之前，重开看到什么。每一行一条用例。
//!
//! 「崩溃」不靠真的杀进程：`RecordingStore` 记下每一次成功提交的 checkpoint，用例在
//! 观察到某个外部信号（运行时收到 submit、上报被确认……）后取当时最新的那份，灌进一个
//! 全新的 store 重开会话（checkpoint 的一次分叉）。原会话随后被丢弃——外壳 drop 会让它的
//! 循环退出并撤销在途任务；就算慢一步，它写的也是另一个 store，与重开后的世界无关。

mod finalization;
mod model_attempt;
mod tool_batch;

use super::*;
use crate::persistence::RunResumePoint;

/// 记下每一次成功提交的 checkpoint 的 store。
#[derive(Default)]
pub(super) struct RecordingStore {
    pub(super) inner: InMemoryCheckpointStore,
    pub(super) committed: StdMutex<Vec<SessionCheckpoint>>,
}

impl RecordingStore {
    pub(super) fn latest(&self) -> SessionCheckpoint {
        self.committed
            .lock()
            .unwrap()
            .last()
            .cloned()
            .expect("至少提交过一次")
    }
}

impl CheckpointStore for RecordingStore {
    fn load<'a>(
        &'a self,
        session_id: &'a SessionId,
    ) -> crate::persistence::PersistenceFuture<
        'a,
        Result<Option<crate::persistence::StoredCheckpoint>, PersistenceError>,
    > {
        self.inner.load(session_id)
    }

    fn compare_and_swap<'a>(
        &'a self,
        expected: Option<CheckpointRevision>,
        checkpoint: SessionCheckpoint,
    ) -> crate::persistence::PersistenceFuture<
        'a,
        Result<crate::persistence::StoredCheckpoint, PersistenceError>,
    > {
        Box::pin(async move {
            let stored = self
                .inner
                .compare_and_swap(expected, checkpoint.clone())
                .await?;
            self.committed.lock().unwrap().push(checkpoint);
            Ok(stored)
        })
    }
}

/// 崩溃前的世界：流式模型 + 提前执行的增量运行时，接在记录 store 上。
pub(super) struct Before {
    pub(super) store: Arc<RecordingStore>,
    pub(super) session: Arc<AgentSession>,
    pub(super) requests: mpsc::UnboundedReceiver<ModelRequest>,
    pub(super) commands: mpsc::UnboundedSender<StreamCommand>,
    pub(super) updates: mpsc::UnboundedReceiver<IncrementalUpdate>,
}

pub(super) async fn before(name: &str, report_slot_on_submit: Option<u32>) -> Before {
    before_with(name, |updates| EagerIncrementalRuntime {
        report_slot_on_submit,
        ..EagerIncrementalRuntime::new(updates, 0)
    })
    .await
}

pub(super) async fn before_with(
    name: &str,
    runtime: impl FnOnce(mpsc::UnboundedSender<IncrementalUpdate>) -> EagerIncrementalRuntime,
) -> Before {
    let session_id = SessionId::new(format!("crash/{name}")).unwrap();
    let store = Arc::new(RecordingStore::default());
    let (request_tx, requests) = mpsc::unbounded_channel();
    let (commands, command_rx) = mpsc::unbounded_channel();
    let (update_tx, updates) = mpsc::unbounded_channel();
    let runtime = Arc::new(runtime(update_tx));
    let session = Arc::new(
        AgentSession::open(
            session_id.clone(),
            store.clone(),
            Arc::new(StaticPrompt),
            runtime,
            Arc::new(NoCompaction),
            Arc::new(CommandStreamModel {
                requests: request_tx,
                commands: Mutex::new(command_rx),
            }),
            SessionConfig::default(),
        )
        .await
        .unwrap(),
    );
    Before {
        store,
        session,
        requests,
        commands,
        updates,
    }
}

/// 崩溃后的世界：只有一份 checkpoint，一个普通的整批运行时，一个非流式模型。
pub(super) struct After {
    pub(super) session: Arc<AgentSession>,
    pub(super) requests: mpsc::UnboundedReceiver<ModelRequest>,
    pub(super) responses: mpsc::UnboundedSender<Result<ModelResponse, AgentError>>,
}

pub(super) async fn reopen(session_id: &SessionId, checkpoint: SessionCheckpoint) -> After {
    reopen_with(
        session_id,
        checkpoint,
        Arc::new(NoCompaction),
        SessionConfig::default(),
    )
    .await
}

pub(super) async fn reopen_with(
    session_id: &SessionId,
    checkpoint: SessionCheckpoint,
    compaction: Arc<dyn Compaction>,
    config: SessionConfig,
) -> After {
    let store = Arc::new(InMemoryCheckpointStore::new());
    store.compare_and_swap(None, checkpoint).await.unwrap();
    let (request_tx, requests) = mpsc::unbounded_channel();
    let (responses, response_rx) = mpsc::unbounded_channel();
    let session = Arc::new(
        AgentSession::open(
            session_id.clone(),
            store,
            Arc::new(StaticPrompt),
            Arc::new(FailingBatchRuntime::default()),
            compaction,
            Arc::new(ChannelModel {
                requests: request_tx,
                responses: Mutex::new(response_rx),
            }),
            config,
        )
        .await
        .unwrap(),
    );
    After {
        session,
        requests,
        responses,
    }
}

pub(super) fn read_call(id: &str) -> ToolCall {
    ToolCall::new(id, "read", json!({"path": id}))
}

/// 断言重开后的第一次模型请求，并让它以 `text` 收尾。
pub(super) async fn resume_and_finish(after: &mut After, text: &str) -> ModelRequest {
    let handle = after
        .session
        .try_resume()
        .await
        .unwrap()
        .expect("盘上有半截运行");
    let request = after.requests.recv().await.unwrap();
    after
        .responses
        .send(Ok(model_response(ModelOutput::text(text))))
        .unwrap();
    assert!(matches!(
        handle.join().await,
        TurnOutcome::Completed { output, .. } if output.text_content() == text
    ));
    request
}
