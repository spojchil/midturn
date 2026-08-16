//! 持久化与压缩替身：确认丢失的 store，各种压缩策略。

use super::super::*;

pub(crate) struct NoCompaction;

impl Compaction for NoCompaction {
    fn compact<'a>(
        &'a self,
        conversation: &'a [TranscriptItem],
    ) -> PortFuture<'a, Result<Vec<TranscriptItem>, AgentError>> {
        Box::pin(async move { Ok(conversation.to_vec()) })
    }
}

#[derive(Default)]
pub(crate) struct PanicOnceCompaction {
    pub(crate) panicked: AtomicBool,
}

impl Compaction for PanicOnceCompaction {
    fn compact<'a>(
        &'a self,
        conversation: &'a [TranscriptItem],
    ) -> PortFuture<'a, Result<Vec<TranscriptItem>, AgentError>> {
        Box::pin(async move {
            if !self.panicked.swap(true, Ordering::SeqCst) {
                panic!("final compaction bug")
            }
            Ok(conversation.to_vec())
        })
    }
}

#[derive(Default)]
pub(crate) struct AckLossCheckpointStore {
    pub(crate) inner: InMemoryCheckpointStore,
    pub(crate) lose_next_ack: AtomicBool,
    pub(crate) committed_writes: AtomicUsize,
}

impl AckLossCheckpointStore {
    pub(crate) fn arm(&self) {
        assert!(
            !self.lose_next_ack.swap(true, Ordering::SeqCst),
            "ack-loss injection must be consumed before it is armed again"
        );
    }

    pub(crate) fn committed_writes(&self) -> usize {
        self.committed_writes.load(Ordering::SeqCst)
    }
}

impl CheckpointStore for AckLossCheckpointStore {
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
            let stored = self.inner.compare_and_swap(expected, checkpoint).await?;
            self.committed_writes.fetch_add(1, Ordering::SeqCst);
            if self.lose_next_ack.swap(false, Ordering::SeqCst) {
                Err(PersistenceError::Backend {
                    summary: "checkpoint committed but acknowledgement was lost".to_owned(),
                })
            } else {
                Ok(stored)
            }
        })
    }
}

pub(crate) struct GatedCompaction {
    pub(crate) started: mpsc::UnboundedSender<()>,
    pub(crate) gate: Arc<Semaphore>,
}

impl Compaction for GatedCompaction {
    fn compact<'a>(
        &'a self,
        conversation: &'a [TranscriptItem],
    ) -> PortFuture<'a, Result<Vec<TranscriptItem>, AgentError>> {
        Box::pin(async move {
            self.started.send(()).unwrap();
            let _permit = self.gate.acquire().await.unwrap();
            Ok(conversation.to_vec())
        })
    }
}

pub(crate) struct GatedSummaryCompaction {
    pub(crate) started: mpsc::UnboundedSender<Vec<TranscriptItem>>,
    pub(crate) gate: Arc<Semaphore>,
}

impl Compaction for GatedSummaryCompaction {
    fn compact<'a>(
        &'a self,
        conversation: &'a [TranscriptItem],
    ) -> PortFuture<'a, Result<Vec<TranscriptItem>, AgentError>> {
        Box::pin(async move {
            self.started.send(conversation.to_vec()).unwrap();
            let _permit = self.gate.acquire().await.unwrap();
            Ok(vec![input("system", "summary")])
        })
    }
}

#[derive(Default)]
pub(crate) struct DanglingCompaction {
    pub(crate) compactions: AtomicUsize,
}

impl Compaction for DanglingCompaction {
    fn compact<'a>(
        &'a self,
        conversation: &'a [TranscriptItem],
    ) -> PortFuture<'a, Result<Vec<TranscriptItem>, AgentError>> {
        Box::pin(async move {
            self.compactions.fetch_add(1, Ordering::SeqCst);
            let mut corrupted = conversation.to_vec();
            corrupted.push(dangling_tool_call("compaction-orphan"));
            Ok(corrupted)
        })
    }
}

#[derive(Default)]
pub(crate) struct DroppingRecoveryCompaction;

impl Compaction for DroppingRecoveryCompaction {
    fn compact<'a>(
        &'a self,
        conversation: &'a [TranscriptItem],
    ) -> PortFuture<'a, Result<Vec<TranscriptItem>, AgentError>> {
        Box::pin(async move {
            Ok(conversation
                .iter()
                .filter(|item| !is_recovery_receipt(item))
                .cloned()
                .collect())
        })
    }
}
