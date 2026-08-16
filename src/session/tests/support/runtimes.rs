//! 工具运行时替身：提前执行的增量运行时、门控整批、各种故障注入。

use super::super::*;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum IncrementalUpdate {
    Began(String),
    Submitted(ToolCallSlot, String),
    CallsSealed(u32),
    Committed,
    Aborted(ToolBatchAbortReason),
}

pub(crate) struct EagerIncrementalRuntime {
    pub(crate) updates: mpsc::UnboundedSender<IncrementalUpdate>,
    /// slot 小于此值的调用在 abort 时已经有可信结果，其余调用确认尚未开始。
    pub(crate) settled_slots: u32,
    pub(crate) fail_commit: bool,
    pub(crate) legacy_dispatches: AtomicUsize,
    /// 非空时，`submit` 一收到该 slot 就立刻通过上报端结算它——模拟批内先完成的那一项。
    pub(crate) report_slot_on_submit: Option<u32>,
    pub(crate) reporter: StdMutex<Option<Arc<dyn ToolCallReporter>>>,
    /// 有值时 `commit` 在结算任何槽之前先等这个闸——用来把运行冻在「commit 在飞」。
    pub(crate) commit_gate: Option<Arc<Semaphore>>,
    /// 有值时 `abort` 在报告之前先等这个闸——用来把运行冻在 `Aborting`。
    pub(crate) abort_gate: Option<Arc<Semaphore>>,
    /// `abort` 时把每个已交付、未结算的槽都报成 OutcomeUnknown（默认按 settled_slots 分类）。
    pub(crate) unknown_on_abort: bool,
}

impl EagerIncrementalRuntime {
    pub(crate) fn new(
        updates: mpsc::UnboundedSender<IncrementalUpdate>,
        settled_slots: u32,
    ) -> Self {
        Self {
            updates,
            settled_slots,
            fail_commit: false,
            legacy_dispatches: AtomicUsize::new(0),
            report_slot_on_submit: None,
            reporter: StdMutex::new(None),
            commit_gate: None,
            abort_gate: None,
            unknown_on_abort: false,
        }
    }
}

impl ToolRuntime for EagerIncrementalRuntime {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition::new(
            "read",
            json!({"type": "object", "additionalProperties": true}),
        )]
    }

    fn dispatch<'a>(
        &'a self,
        batch: ToolCallBatch,
    ) -> PortFuture<'a, Result<ToolResultBatch, AgentError>> {
        Box::pin(async move {
            self.legacy_dispatches.fetch_add(1, Ordering::SeqCst);
            Ok(ToolResultBatch {
                results: batch
                    .calls
                    .into_iter()
                    .map(|call| ToolResult::success_json(call.id, json!({"legacy": true})))
                    .collect(),
            })
        })
    }

    fn begin_incremental<'a>(
        &'a self,
        batch: ToolBatchStart,
        reporter: Arc<dyn ToolCallReporter>,
    ) -> PortFuture<'a, Result<Option<Box<dyn IncrementalToolBatch + 'a>>, AgentError>> {
        Box::pin(async move {
            *self.reporter.lock().unwrap() = Some(reporter);
            self.updates
                .send(IncrementalUpdate::Began(
                    batch.batch_attempt_id.as_str().to_owned(),
                ))
                .unwrap();
            Ok(Some(Box::new(EagerIncrementalBatch {
                runtime: self,
                calls: Vec::new(),
                sealed: false,
            }) as Box<dyn IncrementalToolBatch + 'a>))
        })
    }
}

pub(crate) struct EagerIncrementalBatch<'a> {
    pub(crate) runtime: &'a EagerIncrementalRuntime,
    pub(crate) calls: Vec<IncrementalToolCall>,
    pub(crate) sealed: bool,
}

impl IncrementalToolBatch for EagerIncrementalBatch<'_> {
    fn submit<'a>(
        &'a mut self,
        call: IncrementalToolCall,
    ) -> PortFuture<'a, Result<(), AgentError>> {
        Box::pin(async move {
            // 模拟批内先完成的那一项：一交付就跑完并立刻上报结算。
            if self.runtime.report_slot_on_submit == Some(call.slot.get()) {
                let reporter = self.runtime.reporter.lock().unwrap().clone();
                let reporter = reporter.expect("begin_incremental 已经拿到上报端");
                reporter
                    .settled(
                        call.slot,
                        ToolResult::success_json(call.call.id.clone(), json!({"early": true})),
                    )
                    .await?;
            }
            // `Submitted` 在结算之后才发：用例收到它时，结算已经落盘（settled 返回 Ok 的含义）。
            self.runtime
                .updates
                .send(IncrementalUpdate::Submitted(
                    call.slot,
                    call.call.id.as_str().to_owned(),
                ))
                .unwrap();
            self.calls.push(call);
            Ok(())
        })
    }

    fn calls_sealed<'a>(&'a mut self, call_count: u32) -> PortFuture<'a, Result<(), AgentError>> {
        Box::pin(async move {
            self.runtime
                .updates
                .send(IncrementalUpdate::CallsSealed(call_count))
                .unwrap();
            self.sealed = true;
            Ok(())
        })
    }

    fn commit<'a>(self: Box<Self>) -> PortFuture<'a, Result<(), AgentError>>
    where
        Self: 'a,
    {
        Box::pin(async move {
            self.runtime
                .updates
                .send(IncrementalUpdate::Committed)
                .unwrap();
            if self.runtime.fail_commit {
                return Err(AgentError::new(
                    AgentErrorKind::ToolDispatch,
                    "incremental commit failed",
                ));
            }
            if let Some(gate) = &self.runtime.commit_gate {
                let _permit = gate.acquire().await;
            }
            assert!(self.sealed);
            let reporter = self.runtime.reporter.lock().unwrap().clone();
            let reporter = reporter.expect("begin_incremental 已经拿到上报端");
            // 倒序上报：结果进对话的顺序由内核按槽位排，与上报顺序无关。
            for item in self.calls.into_iter().rev() {
                if self.runtime.report_slot_on_submit == Some(item.slot.get()) {
                    continue; // 交付时已经上报过了。
                }
                reporter
                    .settled(
                        item.slot,
                        ToolResult::success_json(
                            item.call.id,
                            json!({"slot": item.slot.get(), "incremental": true}),
                        ),
                    )
                    .await?;
            }
            Ok(())
        })
    }

    fn abort<'a>(
        self: Box<Self>,
        reason: ToolBatchAbortReason,
    ) -> PortFuture<'a, Result<AbortClassification, AgentError>>
    where
        Self: 'a,
    {
        Box::pin(async move {
            self.runtime
                .updates
                .send(IncrementalUpdate::Aborted(reason))
                .unwrap();
            if let Some(gate) = &self.runtime.abort_gate {
                let _permit = gate.acquire().await;
            }
            let reporter = self.runtime.reporter.lock().unwrap().clone();
            let reporter = reporter.expect("begin_incremental 已经拿到上报端");
            let mut classification = AbortClassification::new();
            for item in self.calls {
                if self.runtime.report_slot_on_submit == Some(item.slot.get()) {
                    continue; // 交付时已经上报过了。
                }
                if self.runtime.unknown_on_abort {
                    classification =
                        classification.outcome_unknown(item.slot, "runtime does not know");
                    continue;
                }
                if item.slot.get() < self.runtime.settled_slots {
                    let mut result = ToolResult::success_json(
                        item.call.id.clone(),
                        json!({"executed": true, "slot": item.slot.get()}),
                    );
                    result.metadata.insert(
                        "private_runtime_trace".to_owned(),
                        json!("DO_NOT_SEND_TO_MODEL"),
                    );
                    // 已知结果先上报；报告里只剩不知道的。
                    let _ = reporter.settled(item.slot, result).await;
                } else {
                    classification = classification.cancelled_before_start(item.slot);
                }
            }
            Ok(classification)
        })
    }
}

pub(crate) struct GatedBatchRuntime {
    pub(crate) batches: mpsc::UnboundedSender<ToolCallBatch>,
    pub(crate) gate: Arc<Semaphore>,
    pub(crate) dispatches: AtomicUsize,
}

impl ToolRuntime for GatedBatchRuntime {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition::new(
            "read",
            json!({"type": "object", "additionalProperties": true}),
        )]
    }

    fn dispatch<'a>(
        &'a self,
        batch: ToolCallBatch,
    ) -> PortFuture<'a, Result<ToolResultBatch, AgentError>> {
        Box::pin(async move {
            self.dispatches.fetch_add(1, Ordering::SeqCst);
            self.batches.send(batch.clone()).unwrap();
            let _permit = self.gate.acquire().await.unwrap();
            Ok(ToolResultBatch {
                results: batch
                    .calls
                    .iter()
                    .rev()
                    .map(|call| {
                        ToolResult::success_json(
                            call.id.clone(),
                            json!({"name": call.name.as_str()}),
                        )
                    })
                    .collect(),
            })
        })
    }
}

#[derive(Default)]
pub(crate) struct FailingBatchRuntime {
    pub(crate) dispatches: AtomicUsize,
}

impl ToolRuntime for FailingBatchRuntime {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition::new(
            "read",
            json!({"type": "object", "additionalProperties": true}),
        )]
    }

    fn dispatch<'a>(
        &'a self,
        _batch: ToolCallBatch,
    ) -> PortFuture<'a, Result<ToolResultBatch, AgentError>> {
        Box::pin(async move {
            self.dispatches.fetch_add(1, Ordering::SeqCst);
            Err(AgentError::new(
                AgentErrorKind::ToolDispatch,
                "dispatch infrastructure failed",
            ))
        })
    }
}

pub(crate) struct PanickingBatchRuntime;

impl ToolRuntime for PanickingBatchRuntime {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition::new(
            "read",
            json!({"type": "object", "additionalProperties": true}),
        )]
    }

    fn dispatch<'a>(
        &'a self,
        _batch: ToolCallBatch,
    ) -> PortFuture<'a, Result<ToolResultBatch, AgentError>> {
        Box::pin(async move { panic!("tool runtime bug") })
    }
}

#[derive(Default)]
pub(crate) struct PanicOnceDefinitionsRuntime {
    pub(crate) panicked: AtomicBool,
}

impl ToolRuntime for PanicOnceDefinitionsRuntime {
    fn definitions(&self) -> Vec<ToolDefinition> {
        if !self.panicked.swap(true, Ordering::SeqCst) {
            panic!("tool definitions bug")
        }
        vec![ToolDefinition::new(
            "read",
            json!({"type": "object", "additionalProperties": true}),
        )]
    }

    // 本用例只验证 definitions 的 panic 隔离，从不走到分发。
    fn dispatch<'a>(
        &'a self,
        _batch: ToolCallBatch,
    ) -> PortFuture<'a, Result<ToolResultBatch, AgentError>> {
        Box::pin(async { unreachable!("此运行时不会被分发") })
    }
}

#[derive(Default)]
pub(crate) struct PartiallyInvalidBatchRuntime;

impl ToolRuntime for PartiallyInvalidBatchRuntime {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition::new(
            "read",
            json!({"type": "object", "additionalProperties": true}),
        )]
    }

    fn dispatch<'a>(
        &'a self,
        batch: ToolCallBatch,
    ) -> PortFuture<'a, Result<ToolResultBatch, AgentError>> {
        Box::pin(async move {
            let first = batch
                .calls
                .into_iter()
                .next()
                .expect("test batch must contain at least one call");
            Ok(ToolResultBatch {
                // 第一项结果有效，但缺少第二项会使整份报告无效。账本必须先原子校验
                // 整个批次，再结算其中任何一项。
                results: vec![ToolResult::success_json(first.id, json!({"ok": true}))],
            })
        })
    }
}

#[derive(Default)]
pub(crate) struct FailingSecondBatchRuntime {
    pub(crate) dispatches: AtomicUsize,
}

impl ToolRuntime for FailingSecondBatchRuntime {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition::new(
            "read",
            json!({"type": "object", "additionalProperties": true}),
        )]
    }

    fn dispatch<'a>(
        &'a self,
        batch: ToolCallBatch,
    ) -> PortFuture<'a, Result<ToolResultBatch, AgentError>> {
        Box::pin(async move {
            let index = self.dispatches.fetch_add(1, Ordering::SeqCst);
            if index == 1 {
                return Err(AgentError::new(
                    AgentErrorKind::ToolDispatch,
                    "second dispatch failed",
                ));
            }
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
