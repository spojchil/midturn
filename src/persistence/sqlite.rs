//! 可选的 SQLite 检查点存储实现。

use std::path::Path;
use std::sync::{Arc, Mutex as StdMutex, MutexGuard};
use std::time::Duration;

use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};

use super::{
    CheckpointRevision, CheckpointStore, PersistenceError, PersistenceFuture, SessionCheckpoint,
    SessionId, StoredCheckpoint,
};

const CREATE_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS agent_checkpoints (
    session_id     TEXT PRIMARY KEY NOT NULL,
    revision       INTEGER NOT NULL CHECK (revision > 0),
    checkpoint_json TEXT NOT NULL
);
"#;

/// 持久化 SQLite 检查点存储。
///
/// 每个实例都持有一个由短期同步互斥锁保护的 SQLite 连接。包括获取锁在内的所有数据库操作
/// 都在 Tokio 阻塞线程池上运行。不同存储实例与不同进程通过 SQLite 的 `IMMEDIATE` 事务协调，
/// 因而“加载/比较/写入”序列具有与 [`super::InMemoryCheckpointStore`] 相同的 CAS 语义。
#[derive(Clone)]
pub struct SqliteCheckpointStore {
    connection: Arc<StdMutex<Connection>>,
}

impl SqliteCheckpointStore {
    /// 打开或创建文件数据库，并初始化其模式。
    pub async fn open(path: impl AsRef<Path>) -> Result<Self, PersistenceError> {
        let path = path.as_ref().to_path_buf();
        let connection = run_blocking(move || {
            let connection = Connection::open(path).map_err(backend)?;
            initialize(connection)
        })
        .await?;
        Ok(Self {
            connection: Arc::new(StdMutex::new(connection)),
        })
    }

    /// 打开私有内存数据库，主要用于测试与临时 Agent。
    pub async fn open_in_memory() -> Result<Self, PersistenceError> {
        let connection = run_blocking(|| {
            let connection = Connection::open_in_memory().map_err(backend)?;
            initialize(connection)
        })
        .await?;
        Ok(Self {
            connection: Arc::new(StdMutex::new(connection)),
        })
    }
}

impl CheckpointStore for SqliteCheckpointStore {
    fn load<'a>(
        &'a self,
        session_id: &'a SessionId,
    ) -> PersistenceFuture<'a, Result<Option<StoredCheckpoint>, PersistenceError>> {
        let connection = Arc::clone(&self.connection);
        let session_id = session_id.clone();
        Box::pin(async move {
            run_blocking(move || {
                let connection = lock_connection(&connection)?;
                let row = connection
                    .query_row(
                        "SELECT revision, checkpoint_json \
                         FROM agent_checkpoints WHERE session_id = ?1",
                        params![session_id.as_str()],
                        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
                    )
                    .optional()
                    .map_err(backend)?;
                let Some((revision, encoded)) = row else {
                    return Ok(None);
                };
                let revision = decode_revision(&session_id, revision)?;
                let checkpoint = decode_checkpoint(&session_id, &encoded)?;
                Ok(Some(StoredCheckpoint {
                    revision,
                    checkpoint,
                }))
            })
            .await
        })
    }

    fn compare_and_swap<'a>(
        &'a self,
        expected: Option<CheckpointRevision>,
        checkpoint: SessionCheckpoint,
    ) -> PersistenceFuture<'a, Result<StoredCheckpoint, PersistenceError>> {
        let connection = Arc::clone(&self.connection);
        Box::pin(async move {
            run_blocking(move || {
                checkpoint.validate()?;
                let session_id = checkpoint.session_id.clone();
                let encoded = serde_json::to_string(&checkpoint).map_err(backend)?;
                let mut connection = lock_connection(&connection)?;
                let transaction = connection
                    .transaction_with_behavior(TransactionBehavior::Immediate)
                    .map_err(backend)?;
                let actual_raw = transaction
                    .query_row(
                        "SELECT revision FROM agent_checkpoints WHERE session_id = ?1",
                        params![session_id.as_str()],
                        |row| row.get::<_, i64>(0),
                    )
                    .optional()
                    .map_err(backend)?;
                let actual = actual_raw
                    .map(|value| decode_revision(&session_id, value))
                    .transpose()?;
                if actual != expected {
                    return Err(PersistenceError::Conflict {
                        session_id,
                        expected,
                        actual,
                    });
                }

                let next = actual
                    .map(CheckpointRevision::get)
                    .unwrap_or(0)
                    .checked_add(1)
                    .filter(|revision| *revision <= i64::MAX as u64)
                    .ok_or_else(|| PersistenceError::RevisionExhausted {
                        session_id: session_id.clone(),
                    })?;
                let next_sql = next as i64;

                match actual {
                    None => {
                        transaction
                            .execute(
                                "INSERT INTO agent_checkpoints \
                                 (session_id, revision, checkpoint_json) VALUES (?1, ?2, ?3)",
                                params![session_id.as_str(), next_sql, encoded],
                            )
                            .map_err(backend)?;
                    }
                    Some(previous) => {
                        let changed = transaction
                            .execute(
                                "UPDATE agent_checkpoints \
                                 SET revision = ?2, checkpoint_json = ?3 \
                                 WHERE session_id = ?1 AND revision = ?4",
                                params![
                                    session_id.as_str(),
                                    next_sql,
                                    encoded,
                                    previous.get() as i64
                                ],
                            )
                            .map_err(backend)?;
                        if changed != 1 {
                            return Err(PersistenceError::Conflict {
                                session_id,
                                expected,
                                actual,
                            });
                        }
                    }
                }
                transaction.commit().map_err(backend)?;

                Ok(StoredCheckpoint {
                    revision: CheckpointRevision::new(next),
                    checkpoint,
                })
            })
            .await
        })
    }
}

fn initialize(connection: Connection) -> Result<Connection, PersistenceError> {
    connection
        .busy_timeout(Duration::from_secs(5))
        .map_err(backend)?;
    connection
        .pragma_update(None, "synchronous", "FULL")
        .map_err(backend)?;
    connection.execute_batch(CREATE_SCHEMA).map_err(backend)?;
    Ok(connection)
}

fn lock_connection(
    connection: &StdMutex<Connection>,
) -> Result<MutexGuard<'_, Connection>, PersistenceError> {
    connection.lock().map_err(|_| PersistenceError::Backend {
        summary: "sqlite_connection_lock_poisoned".to_owned(),
    })
}

fn decode_revision(
    session_id: &SessionId,
    revision: i64,
) -> Result<CheckpointRevision, PersistenceError> {
    let revision = u64::try_from(revision).map_err(|_| PersistenceError::Backend {
        summary: format!("invalid SQLite checkpoint revision for {session_id}"),
    })?;
    if revision == 0 {
        return Err(PersistenceError::Backend {
            summary: format!("invalid SQLite checkpoint revision for {session_id}"),
        });
    }
    Ok(CheckpointRevision::new(revision))
}

fn decode_checkpoint(
    session_id: &SessionId,
    encoded: &str,
) -> Result<SessionCheckpoint, PersistenceError> {
    let checkpoint: SessionCheckpoint = serde_json::from_str(encoded).map_err(backend)?;
    if &checkpoint.session_id != session_id {
        return Err(PersistenceError::Backend {
            summary: "sqlite_checkpoint_session_id_mismatch".to_owned(),
        });
    }
    checkpoint.validate()?;
    Ok(checkpoint)
}

async fn run_blocking<T, F>(operation: F) -> Result<T, PersistenceError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, PersistenceError> + Send + 'static,
{
    tokio::task::spawn_blocking(operation)
        .await
        .map_err(|error| PersistenceError::Backend {
            summary: format!("sqlite blocking task failed: {error}"),
        })?
}

fn backend(error: impl std::fmt::Display) -> PersistenceError {
    PersistenceError::Backend {
        summary: format!("sqlite: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mailbox::MailboxInput;
    use crate::types::InputMessage;

    fn session_id() -> SessionId {
        SessionId::new("sqlite-session").unwrap()
    }

    fn checkpoint(run_sequence: u64) -> SessionCheckpoint {
        let mut checkpoint = SessionCheckpoint::empty(session_id());
        checkpoint.run_sequence = run_sequence;
        std::sync::Arc::make_mut(&mut checkpoint.conversation)
            .push(InputMessage::text("user", "持久化测试").into());
        checkpoint
            .mailbox
            .push(MailboxInput::when_idle(vec![InputMessage::text(
                "operator",
                "稍后处理",
            )
            .into()]));
        checkpoint
    }

    #[tokio::test]
    async fn sqlite_round_trips_a_checkpoint() {
        let store = SqliteCheckpointStore::open_in_memory().await.unwrap();
        assert!(store.load(&session_id()).await.unwrap().is_none());

        let expected = checkpoint(9);
        let saved = store
            .compare_and_swap(None, expected.clone())
            .await
            .unwrap();
        assert_eq!(saved.revision, CheckpointRevision::new(1));
        assert_eq!(saved.checkpoint, expected);

        let loaded = store.load(&session_id()).await.unwrap().unwrap();
        assert_eq!(loaded, saved);
    }

    #[tokio::test]
    async fn sqlite_cas_rejects_create_and_update_races() {
        let store = SqliteCheckpointStore::open_in_memory().await.unwrap();
        let first = store.compare_and_swap(None, checkpoint(1)).await.unwrap();

        let duplicate_create = store
            .compare_and_swap(None, checkpoint(2))
            .await
            .unwrap_err();
        assert!(matches!(
            duplicate_create,
            PersistenceError::Conflict {
                expected: None,
                actual: Some(CheckpointRevision(1)),
                ..
            }
        ));

        let second = store
            .compare_and_swap(Some(first.revision), checkpoint(2))
            .await
            .unwrap();
        assert_eq!(second.revision, CheckpointRevision::new(2));

        let stale = store
            .compare_and_swap(Some(first.revision), checkpoint(3))
            .await
            .unwrap_err();
        assert!(matches!(
            stale,
            PersistenceError::Conflict {
                expected: Some(CheckpointRevision(1)),
                actual: Some(CheckpointRevision(2)),
                ..
            }
        ));
        assert_eq!(
            store
                .load(&session_id())
                .await
                .unwrap()
                .unwrap()
                .checkpoint
                .run_sequence,
            2
        );
    }
}
