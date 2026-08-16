//! 检查点存储端口及内存参考实现。

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;

use tokio::sync::Mutex;

use super::{CheckpointRevision, PersistenceError, SessionCheckpoint, SessionId, StoredCheckpoint};

/// 持久化端口返回的 Future，无需依赖 `async-trait`。
pub type PersistenceFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// 采用乐观并发控制的持久化检查点存储。
///
/// `expected = None` 表示“仅在不存在时创建”。更新既有检查点时，必须提供 `load`
/// 或上一次成功调用 `compare_and_swap` 所返回的准确修订号。
///
/// # 它跑在会话的驱动循环上
///
/// 这是**唯一**一个会话驱动循环会同步 `await` 的用户端口：每一次改变持久状态的转移
/// 都以一次 `compare_and_swap` 收尾，它就是整个会话的串行化点。因此——
///
/// - 它慢，会话就慢：写盘期间没有任何命令被处理（模型流事件、工具上报都在排队）。
///   这是有意的：状态没落盘之前不该有人看到它。
/// - **它不得回调同一个会话**（`enqueue` / `settled` / `Inspect`……）：那些调用要等循环
///   处理命令，而循环正等它返回——死锁。
/// - 返回 `Err` 就是隔离：`Backend` 类错误按「可能已提交」处理，其余按「未提交」；两种都
///   要求丢弃这个会话实例、从 store 重开。
/// - **panic 也不许带走循环**：内核兜住它，按「可能已提交」隔离（`PersistenceCommitUnknown`）。
pub trait CheckpointStore: Send + Sync {
    fn load<'a>(
        &'a self,
        session_id: &'a SessionId,
    ) -> PersistenceFuture<'a, Result<Option<StoredCheckpoint>, PersistenceError>>;

    fn compare_and_swap<'a>(
        &'a self,
        expected: Option<CheckpointRevision>,
        checkpoint: SessionCheckpoint,
    ) -> PersistenceFuture<'a, Result<StoredCheckpoint, PersistenceError>>;
}

/// 用于测试、嵌入式场景和参考语义的进程内存储。
///
/// 它准确模拟 CAS，但无法跨进程重启持久保存。
#[derive(Default)]
pub struct InMemoryCheckpointStore {
    checkpoints: Mutex<BTreeMap<SessionId, StoredCheckpoint>>,
}

impl InMemoryCheckpointStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl CheckpointStore for InMemoryCheckpointStore {
    fn load<'a>(
        &'a self,
        session_id: &'a SessionId,
    ) -> PersistenceFuture<'a, Result<Option<StoredCheckpoint>, PersistenceError>> {
        Box::pin(async move {
            let checkpoint = self.checkpoints.lock().await.get(session_id).cloned();
            if let Some(stored) = &checkpoint {
                stored.checkpoint.validate()?;
            }
            Ok(checkpoint)
        })
    }

    fn compare_and_swap<'a>(
        &'a self,
        expected: Option<CheckpointRevision>,
        checkpoint: SessionCheckpoint,
    ) -> PersistenceFuture<'a, Result<StoredCheckpoint, PersistenceError>> {
        Box::pin(async move {
            checkpoint.validate()?;
            let session_id = checkpoint.session_id.clone();
            let mut checkpoints = self.checkpoints.lock().await;
            let actual = checkpoints.get(&session_id).map(|stored| stored.revision);
            if actual != expected {
                return Err(PersistenceError::Conflict {
                    session_id,
                    expected,
                    actual,
                });
            }
            let next_revision = actual
                .map(CheckpointRevision::get)
                .unwrap_or(0)
                .checked_add(1)
                .map(CheckpointRevision::new)
                .ok_or_else(|| PersistenceError::RevisionExhausted {
                    session_id: session_id.clone(),
                })?;
            let stored = StoredCheckpoint {
                revision: next_revision,
                checkpoint,
            };
            checkpoints.insert(session_id, stored.clone());
            Ok(stored)
        })
    }
}
