//! 轮外触发源：信箱。
//!
//! 「是否有运行在跑」= `run.is_some()`；「有内容等着」= 信箱非空。不设第二个标志位——
//! 多一个就多一份真相。check-then-act 的竞态靠单写者消除，不靠锁。

use tokio::sync::oneshot;

use crate::events::{AgentEvent, EventKind};
use crate::mailbox::{Delivery, Mailbox, MailboxInput, MailboxRejected, MailboxRejectedReason};
use crate::types::{
    durable_fact_kind, validate_closed_transcript, AgentError, AgentErrorKind, RunId,
};

use super::super::HoldReason;
use super::{reply_to, Effect, Machine, RescheduleReply, StartOutcome};

impl Machine {
    pub(super) fn on_enqueue(
        &mut self,
        input: MailboxInput,
        may_start: bool,
        expected_run: Option<RunId>,
        reply: oneshot::Sender<Result<StartOutcome, MailboxRejected>>,
        out: &mut Vec<Effect>,
    ) {
        // 耐久事实只能由内核自己写：外部输入若冒充（带保留 `kind` 的 JSON 段），压缩就
        // 得永远原样保留它——普通内容借此变成不可压缩、跨轮积累的钉子。
        let forges_durable_fact = input
            .items
            .iter()
            .any(|item| durable_fact_kind(item).is_some());
        if forges_durable_fact || validate_closed_transcript(&input.items).is_err() {
            reply_to(
                out,
                reply,
                Err(rejected(MailboxRejectedReason::InvalidInput, input)),
            );
            return;
        }
        if let Some(expected) = &expected_run {
            let matches = self.run.as_ref().is_some_and(|run| &run.id == expected);
            if !matches {
                reply_to(
                    out,
                    reply,
                    Err(rejected(MailboxRejectedReason::WrongRun, input)),
                );
                return;
            }
        }
        let core = &mut self.core;
        let input_items = input.items.len();
        let input_bytes = serde_json::to_vec(&input)
            .map(|encoded| encoded.len())
            .unwrap_or(usize::MAX);
        if core.mailbox.item_count().saturating_add(input_items) > core.config.max_mailbox_items
            || core.mailbox.estimated_bytes().saturating_add(input_bytes)
                > core.config.max_mailbox_bytes
        {
            reply_to(
                out,
                reply,
                Err(rejected(MailboxRejectedReason::Capacity, input)),
            );
            return;
        }

        let delivery = input.delivery;
        let event_run_id = self.run.as_ref().map(|run| run.id.clone());
        core.mailbox.push(input);
        core.touch();
        core.emit(out, EventKind::MailboxEnqueued, |sequence| {
            AgentEvent::MailboxEnqueued {
                sequence,
                run_id: event_run_id,
                delivery,
                item_count: input_items,
            }
        });

        let outcome = if !may_start {
            // enqueue_for：这句话只对当前这一轮有意义，永不开新运行。
            StartOutcome::Pending
        } else if delivery == Delivery::Passive {
            // 被动内容不构成叫醒会话的理由；有运行在跑时它也只是搭下一次请求的便车，
            // 不是「有人会来取」的那种 Pending。
            StartOutcome::Held(HoldReason::Passive)
        } else {
            self.try_start(out)
        };
        reply_to(out, reply, Ok(outcome));
    }

    pub(super) fn on_reschedule(
        &mut self,
        from: Delivery,
        to: Delivery,
        reply: oneshot::Sender<Result<RescheduleReply, AgentError>>,
        out: &mut Vec<Effect>,
    ) {
        let core = &mut self.core;
        // 回滚要还原三个队列的原貌：反向搬一次会把目标队列的原住民一起搬走。
        let previous = core.mailbox.snapshot();
        let envelopes = core.mailbox.reschedule(from, to);
        if envelopes == 0 {
            // 什么都没动就不必写盘，也不该顺带把会话叫醒。
            reply_to(
                out,
                reply,
                Ok(RescheduleReply {
                    envelopes: 0,
                    outcome: StartOutcome::Pending,
                }),
            );
            return;
        }
        // 改标签会改序列化字节数：原本恰好低于上限的队列可以被改期推过线。
        if core.mailbox.estimated_bytes() > core.config.max_mailbox_bytes {
            core.mailbox = Mailbox::from_inputs(previous);
            reply_to(
                out,
                reply,
                Err(AgentError::new(
                    AgentErrorKind::InvalidState,
                    "reschedule_exceeds_mailbox_byte_budget",
                )),
            );
            return;
        }
        core.touch();
        let event_run_id = self.run.as_ref().map(|run| run.id.clone());
        // Emit 排在 Persist 之后：只在落盘确认后才有人看到这条。
        core.emit(out, EventKind::MailboxRescheduled, |sequence| {
            AgentEvent::MailboxRescheduled {
                sequence,
                run_id: event_run_id,
                from,
                to,
                envelopes,
            }
        });
        let outcome = if core.mailbox.has_trigger_items() {
            self.try_start(out)
        } else {
            StartOutcome::Held(HoldReason::Passive)
        };
        reply_to(out, reply, Ok(RescheduleReply { envelopes, outcome }));
    }

    pub(super) fn on_set_auto_start(&mut self, enabled: bool, out: &mut Vec<Effect>) {
        // 纯运行时开关，不落盘：它表达的是「当前这个进程还想不想干活」。
        self.core.auto_start = enabled;
        if enabled {
            // 重新武装时可能已经攒了待投递的输入，立刻兑现。
            let _ = self.try_start(out);
        }
    }

    /// 空闲且信箱有触发内容且允许自动开始 → 建一个 `Run` 并推进到第一个等待点。
    ///
    /// 这是所有轮外触发汇聚的地方：`Enqueue`、`Reschedule`、`SetAutoStart(true)`、
    /// `Boot`、以及上一轮收尾之后的再检查。
    pub(crate) fn try_start(&mut self, out: &mut Vec<Effect>) -> StartOutcome {
        if self.run.is_some() {
            // 有运行在跑就不必开新的：内容会在它的下一个边界进入。
            return StartOutcome::Pending;
        }
        let core = &mut self.core;
        if !core.auto_start {
            return StartOutcome::Held(HoldReason::AutoStartDisabled);
        }
        if core
            .effects
            .iter()
            .any(|(_, record)| record.state.requires_reconciliation())
        {
            return StartOutcome::Held(HoldReason::ReconciliationRequired);
        }
        if core.recovered.is_some() {
            return StartOutcome::Held(HoldReason::ResumeRequired);
        }
        if !core.mailbox.has_trigger_items() {
            return StartOutcome::Held(HoldReason::Passive);
        }
        let Some(next_run_seq) = core.run_seq.checked_add(1) else {
            return StartOutcome::Held(HoldReason::RunIdExhausted);
        };
        core.run_seq = next_run_seq;
        let run_id = RunId::new(format!("{}/run/{}", core.session_id.as_str(), next_run_seq));
        core.touch();
        let completion = self.start_run(run_id.clone(), out);
        StartOutcome::Started { run_id, completion }
    }
}

fn rejected(reason: MailboxRejectedReason, input: MailboxInput) -> MailboxRejected {
    MailboxRejected { reason, input }
}
