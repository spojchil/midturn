use serde_json::{json, Value};

use super::*;
use crate::types::{ToolCallId, ToolName, ToolResult};

fn session_id() -> SessionId {
    SessionId::new("session/example").unwrap()
}

fn run_id() -> RunId {
    RunId::new("session/example/run/7")
}

fn intent(slot: u32, call_id: &str) -> EffectIntent {
    EffectIntent::new(
        session_id(),
        run_id(),
        EffectOrigin::StreamingAttempt {
            batch_attempt_id: ToolBatchAttemptId::new("attempt/3"),
        },
        ToolCallSlot::new(slot),
        ToolCall {
            id: ToolCallId::new(call_id),
            name: ToolName::new("charge"),
            arguments: json!({"cents": 500}),
            provider_data: Default::default(),
        },
    )
    .unwrap()
}

fn checkpoint_with_attempt(ledger: EffectLedger, ids: Vec<EffectId>) -> SessionCheckpoint {
    SessionCheckpoint {
        active_run: Some(RunSnapshot {
            run_id: run_id(),
            resume_from: RunResumePoint::ModelAttempt {
                request_index: 3,
                batch_attempt_id: ToolBatchAttemptId::new("attempt/3"),
                effect_ids: ids,
            },
            model_request_sequence: 3,
            tool_batch_sequence: 0,
            interrupted_tool_recoveries: 0,
            usage: None,
        }),
        run_sequence: 7,
        effects: ledger,
        ..SessionCheckpoint::empty(session_id())
    }
}

#[tokio::test]
async fn compare_and_swap_detects_stale_writers() {
    let store = InMemoryCheckpointStore::new();
    let original = SessionCheckpoint::empty(session_id());
    let first = store.compare_and_swap(None, original).await.unwrap();
    assert_eq!(first.revision, CheckpointRevision::new(1));

    let duplicate_create = store
        .compare_and_swap(None, first.checkpoint.clone())
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

    let mut updated = first.checkpoint.clone();
    updated.run_sequence = 1;
    let second = store
        .compare_and_swap(Some(first.revision), updated)
        .await
        .unwrap();
    assert_eq!(second.revision, CheckpointRevision::new(2));

    let stale = store
        .compare_and_swap(Some(first.revision), first.checkpoint)
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
        1
    );
}

#[tokio::test]
async fn recovery_never_redelivers_in_flight_or_unknown_effects() {
    let store = InMemoryCheckpointStore::new();
    let mut ledger = EffectLedger::default();
    let effect = intent(0, "call-1");
    let id = effect.id.clone();
    let expected_intent = effect.clone();
    ledger.record_intent(effect).unwrap();
    ledger.begin_delivery(&id).unwrap();

    let saved = store
        .compare_and_swap(None, checkpoint_with_attempt(ledger, vec![id.clone()]))
        .await
        .unwrap();
    let recovered = store.load(&session_id()).await.unwrap().unwrap();
    assert_eq!(
        recovered
            .checkpoint
            .effects
            .recovery_actions_for_run(&run_id()),
        vec![EffectRecoveryAction::Reconcile {
            id: id.clone(),
            intent: expected_intent,
            summary: "effect_was_in_flight_at_recovery:delivery_attempt=1".to_owned(),
        }]
    );

    let mut checkpoint = recovered.checkpoint;
    checkpoint
        .effects
        .reconcile(
            &id,
            EffectOutcome::OutcomeUnknown {
                summary: "processor_timeout_after_submission".to_owned(),
            },
        )
        .unwrap();
    let saved = store
        .compare_and_swap(Some(saved.revision), checkpoint)
        .await
        .unwrap();
    assert!(matches!(
        saved.checkpoint.effects.get(&id).unwrap().state,
        EffectState::Outcome {
            outcome: EffectOutcome::OutcomeUnknown { .. }
        }
    ));
    assert!(saved
        .checkpoint
        .effects
        .clone()
        .begin_delivery(&id)
        .is_err());
}

#[test]
fn reconciliation_can_settle_unknown_but_cannot_overwrite_terminal_facts() {
    let mut ledger = EffectLedger::default();
    let effect = intent(0, "call-1");
    let id = effect.id.clone();
    ledger.record_intent(effect).unwrap();
    ledger.begin_delivery(&id).unwrap();
    ledger
        .resolve(
            &id,
            EffectOutcome::OutcomeUnknown {
                summary: "network_lost".to_owned(),
            },
        )
        .unwrap();

    let result = ToolResult::success_json(ToolCallId::new("call-1"), json!({"ok": true}));
    ledger
        .reconcile(&id, EffectOutcome::Settled(result.clone()))
        .unwrap();
    assert_eq!(
        ledger.recovery_actions_for_run(&run_id()),
        vec![EffectRecoveryAction::ApplySettled {
            id: id.clone(),
            result,
        }]
    );
    assert!(ledger
        .reconcile(
            &id,
            EffectOutcome::OutcomeUnknown {
                summary: "late_conflict".to_owned(),
            },
        )
        .is_err());

    let cancelled = intent(1, "call-2");
    let cancelled_id = cancelled.id.clone();
    ledger.record_intent(cancelled).unwrap();
    ledger.cancel_prepared(&cancelled_id).unwrap();
    assert!(matches!(
        ledger.get(&cancelled_id).unwrap().state,
        EffectState::Outcome {
            outcome: EffectOutcome::CancelledBeforeStart
        }
    ));
}

#[test]
fn effect_outcomes_are_validated_before_mutating_the_ledger() {
    let mut ledger = EffectLedger::default();
    let effect = intent(0, "call-1");
    let id = effect.id.clone();
    ledger.record_intent(effect).unwrap();
    ledger.begin_delivery(&id).unwrap();

    let wrong_result =
        ToolResult::success_json(ToolCallId::new("another-call"), json!({"ok": true}));
    assert!(matches!(
        ledger.resolve(&id, EffectOutcome::Settled(wrong_result)),
        Err(EffectLedgerError::InvalidOutcome { .. })
    ));
    assert!(matches!(
        ledger.get(&id).unwrap().state,
        EffectState::InFlight { .. }
    ));

    assert!(matches!(
        ledger.resolve(
            &id,
            EffectOutcome::OutcomeUnknown {
                summary: "  ".to_owned(),
            },
        ),
        Err(EffectLedgerError::InvalidOutcome { .. })
    ));
    assert!(matches!(
        ledger.get(&id).unwrap().state,
        EffectState::InFlight { .. }
    ));
}

#[test]
fn checkpoint_schema_round_trips_and_rejects_unknown_versions() {
    let checkpoint = SessionCheckpoint::empty(session_id());
    let encoded = serde_json::to_value(&checkpoint).unwrap();
    let decoded: SessionCheckpoint = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, checkpoint);
    decoded.validate().unwrap();

    let mut unsupported = checkpoint;
    unsupported.schema_version = CHECKPOINT_SCHEMA_VERSION + 1;
    assert!(matches!(
        unsupported.validate(),
        Err(PersistenceError::UnsupportedSchema { .. })
    ));

    // 透明字符串 ID 可作为账本中的有效 JSON 对象键。
    let mut ledger = EffectLedger::default();
    ledger.record_intent(intent(0, "call-1")).unwrap();
    assert_ne!(serde_json::to_value(&ledger).unwrap(), Value::Null);
}

#[test]
fn checkpoint_validation_rejects_corrupt_resume_identity_and_progress() {
    let mut checkpoint = SessionCheckpoint {
        run_sequence: 1,
        active_run: Some(RunSnapshot {
            run_id: RunId::new("session/example/run/1"),
            resume_from: RunResumePoint::BeforeModelRequest,
            model_request_sequence: 0,
            tool_batch_sequence: 0,
            interrupted_tool_recoveries: 0,
            usage: None,
        }),
        ..SessionCheckpoint::empty(session_id())
    };
    checkpoint.validate().unwrap();

    checkpoint.active_run.as_mut().unwrap().run_id = RunId::new("another-session/run/1");
    assert!(matches!(
        checkpoint.validate(),
        Err(PersistenceError::InvalidCheckpoint { ref summary })
            if summary == "active_run_id_does_not_match_session_sequence"
    ));

    checkpoint.active_run.as_mut().unwrap().run_id = RunId::new("session/example/run/1");
    checkpoint.active_run.as_mut().unwrap().tool_batch_sequence = 1;
    assert!(matches!(
        checkpoint.validate(),
        Err(PersistenceError::InvalidCheckpoint { ref summary })
            if summary == "tool_sequence_exceeds_model_sequence"
    ));
}

#[test]
fn checkpoint_validation_rejects_mismatched_completion_and_orphan_prepared_effect() {
    let output = ModelOutput::text("final");
    let mut completion = SessionCheckpoint {
        run_sequence: 1,
        conversation: vec![TranscriptItem::ModelOutput(ModelOutput::text("different"))].into(),
        active_run: Some(RunSnapshot {
            run_id: RunId::new("session/example/run/1"),
            resume_from: RunResumePoint::BeforeCompletion {
                output: output.clone(),
            },
            model_request_sequence: 1,
            tool_batch_sequence: 0,
            interrupted_tool_recoveries: 0,
            usage: None,
        }),
        ..SessionCheckpoint::empty(session_id())
    };
    assert!(matches!(
        completion.validate(),
        Err(PersistenceError::InvalidCheckpoint { ref summary })
            if summary == "completion_output_not_at_conversation_tail"
    ));
    completion.conversation = vec![TranscriptItem::ModelOutput(output)].into();
    completion.validate().unwrap();

    let mut orphan = SessionCheckpoint {
        run_sequence: 7,
        active_run: Some(RunSnapshot {
            run_id: run_id(),
            resume_from: RunResumePoint::BeforeModelRequest,
            model_request_sequence: 1,
            tool_batch_sequence: 0,
            interrupted_tool_recoveries: 0,
            usage: None,
        }),
        ..SessionCheckpoint::empty(session_id())
    };
    orphan.effects.record_intent(intent(0, "call-1")).unwrap();
    assert!(matches!(
        orphan.validate(),
        Err(PersistenceError::InvalidCheckpoint { ref summary })
            if summary == "unreferenced_nonterminal_effect"
    ));
}
