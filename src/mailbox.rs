//! 运行时输入信箱。
//!
//! 投递时机由状态机边界而非模型角色决定：
//! - `NextModelRequest` 是引导输入，在下次调用模型前排空；如果模型即将结束，
//!   则改为再调用一次模型。
//! - `WhenIdle` 是后续输入，仅在本次运行原本将要结束时排空。

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::run::RequestBoundaryKind;
use crate::types::TranscriptItem;

#[non_exhaustive]
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Delivery {
    /// 下一次请求模型之前投递；会话空闲时立刻开跑。
    NextModelRequest,
    /// 本轮本该结束时投递，排在 `NextModelRequest` 之后；会话空闲时立刻开跑。
    WhenIdle,
    /// 投递时机同 `WhenIdle`，但<b>不会把会话叫醒</b>。
    ///
    /// 给那些「模型知道就好，但不值得为它专门跑一轮」的内容用——典型是迟到的工具结果：
    /// 唤醒模型、它说一句「知道了」、然后结束，这一轮纯烧 token。留到下次有别的原因
    /// 开跑时捎带即可。
    Passive,
}

/// 一个原子信箱信封，其中的项目保持顺序并整体投递。
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MailboxInput {
    pub delivery: Delivery,
    pub items: Vec<TranscriptItem>,
}

impl MailboxInput {
    pub fn next_model_request(items: Vec<TranscriptItem>) -> Self {
        Self {
            delivery: Delivery::NextModelRequest,
            items,
        }
    }

    pub fn when_idle(items: Vec<TranscriptItem>) -> Self {
        Self {
            delivery: Delivery::WhenIdle,
            items,
        }
    }

    pub fn passive(items: Vec<TranscriptItem>) -> Self {
        Self {
            delivery: Delivery::Passive,
            items,
        }
    }
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MailboxRejectedReason {
    /// 输入针对的 run 已经结束或不是当前代际。仅 `enqueue_for` 会返回。
    WrongRun,
    /// 输入会超过 session 配置的 mailbox 项数或字节预算。
    Capacity,
    /// checkpoint store 未能提交该信封。
    Persistence,
    /// checkpoint 写入可能已经提交；必须重开检查，不能直接重投该信封。
    PersistenceCommitUnknown,
    /// 信封中的记录段包含孤立或未闭合的工具调用/结果，或者冒充内核的耐久事实
    /// （带保留 `kind` 的 JSON 段——那些只能由内核自己写进对话）。
    InvalidInput,
    /// 投递可能需要派发驱动器，因此当前线程必须处于 Tokio runtime 中。
    RuntimeUnavailable,
    /// 会话的驱动循环已经不在了（进程退出中，或内核缺陷）——不是持久化问题，
    /// 但这个实例也不能再用了。
    SessionClosed,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MailboxRejected {
    pub reason: MailboxRejectedReason,
    pub input: MailboxInput,
}

impl std::fmt::Display for MailboxRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mailbox input rejected: {:?}", self.reason)
    }
}

impl std::error::Error for MailboxRejected {}

#[derive(Clone, Default)]
pub(crate) struct Mailbox {
    next_request: VecDeque<MailboxInput>,
    when_idle: VecDeque<MailboxInput>,
    /// 与 `when_idle` 同时投递，但从不触发开跑。
    passive: VecDeque<MailboxInput>,
}

impl Mailbox {
    pub(crate) fn from_inputs(inputs: Vec<MailboxInput>) -> Self {
        let mut mailbox = Self::default();
        for input in inputs {
            mailbox.push(input);
        }
        mailbox
    }

    pub(crate) fn push(&mut self, input: MailboxInput) {
        self.queue_mut(input.delivery).push_back(input);
    }

    fn queue_mut(&mut self, delivery: Delivery) -> &mut VecDeque<MailboxInput> {
        match delivery {
            Delivery::NextModelRequest => &mut self.next_request,
            Delivery::WhenIdle => &mut self.when_idle,
            Delivery::Passive => &mut self.passive,
        }
    }

    /// 把当前排队中的某一类内容整体改成另一类交付语义，返回被移动的信封数。
    ///
    /// 移动保持相对顺序，并排在目标队列已有内容之后——那些内容更早就是按目标语义
    /// 排队的，不该被后来改主意的内容插队。
    pub(crate) fn reschedule(&mut self, from: Delivery, to: Delivery) -> usize {
        if from == to {
            return 0;
        }
        let moved: Vec<MailboxInput> = self.queue_mut(from).drain(..).collect();
        let count = moved.len();
        for mut input in moved {
            input.delivery = to;
            self.queue_mut(to).push_back(input);
        }
        count
    }

    /// 判断指定边界是否有实际记录可投递，但不改变队列。
    ///
    /// 完成边界用它把“空队列检查 + 封口”保持在同一次加锁中；只有确实要再请求模型时，
    /// 驱动器才会先释放锁执行轮间压缩。
    pub(crate) fn has_deliverable_items(&self, boundary: RequestBoundaryKind) -> bool {
        Self::any_items(&self.next_request)
            || (boundary == RequestBoundaryKind::BeforeCompletion
                && Self::any_items(&self.when_idle))
    }

    /// 是否有内容值得把空闲的会话叫醒。
    ///
    /// 与 `has_deliverable_items` 的区别只在 `Passive`：它照常投递，但从不构成开跑的理由。
    pub(crate) fn has_trigger_items(&self) -> bool {
        Self::any_items(&self.next_request) || Self::any_items(&self.when_idle)
    }

    fn all(&self) -> impl Iterator<Item = &MailboxInput> {
        self.next_request
            .iter()
            .chain(&self.when_idle)
            .chain(&self.passive)
    }

    fn any_items(queue: &VecDeque<MailboxInput>) -> bool {
        queue.iter().any(|input| !input.items.is_empty())
    }

    /// 排空操作是信箱的线性化点：检查与清除在一次操作中完成。
    /// 在完成边界，引导输入优先；每类输入内部保持先进先出。
    pub(crate) fn drain(&mut self, boundary: RequestBoundaryKind) -> Vec<TranscriptItem> {
        let mut drained = Self::flatten(&mut self.next_request);
        if boundary == RequestBoundaryKind::BeforeCompletion {
            drained.extend(Self::flatten(&mut self.when_idle));
        }
        // 被动内容只搭便车。模型请求边界之后必然还有一次请求，可以直接带上；
        // 完成边界只有在确实要再请求一次时才带——否则它自己就成了「再来一轮」的理由，
        // 而那正是这种投递方式要避免的。
        if boundary == RequestBoundaryKind::BeforeModelRequest || !drained.is_empty() {
            drained.extend(Self::flatten(&mut self.passive));
        }
        drained
    }

    /// 会话从空闲开跑时的第一个边界：三类内容一并投递。
    /// 「等它跑完再说」在没有「它」的时候就是「现在」；被动内容照例搭便车。
    pub(crate) fn drain_idle(&mut self) -> Vec<TranscriptItem> {
        let mut drained = Self::flatten(&mut self.next_request);
        drained.extend(Self::flatten(&mut self.when_idle));
        drained.extend(Self::flatten(&mut self.passive));
        drained
    }

    pub(crate) fn snapshot(&self) -> Vec<MailboxInput> {
        self.next_request
            .iter()
            .chain(&self.when_idle)
            .chain(&self.passive)
            .cloned()
            .collect()
    }

    pub(crate) fn item_count(&self) -> usize {
        // 三个队列都要算：`max_mailbox_*` 是整个信箱的上限，不是某一类投递的上限。
        // 漏掉 passive 会让「不叫醒模型」的内容既不计数也不计字节，从而无界增长。
        self.all().map(|input| input.items.len()).sum()
    }

    pub(crate) fn estimated_bytes(&self) -> usize {
        self.all()
            .map(|input| {
                serde_json::to_vec(input)
                    .map(|encoded| encoded.len())
                    .unwrap_or(usize::MAX)
            })
            .fold(0usize, usize::saturating_add)
    }

    fn flatten(queue: &mut VecDeque<MailboxInput>) -> Vec<TranscriptItem> {
        queue.drain(..).flat_map(|input| input.items).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::InputMessage;

    fn item(role: &str, text: &str) -> TranscriptItem {
        InputMessage::text(role, text).into()
    }

    #[test]
    fn next_request_and_idle_delivery_have_distinct_boundaries() {
        let mut mailbox = Mailbox::default();
        mailbox.push(MailboxInput::when_idle(vec![item("operator", "later")]));
        mailbox.push(MailboxInput::next_model_request(vec![item(
            "developer",
            "now",
        )]));

        let first = mailbox.drain(RequestBoundaryKind::BeforeModelRequest);
        assert_eq!(first, vec![item("developer", "now")]);

        let final_boundary = mailbox.drain(RequestBoundaryKind::BeforeCompletion);
        assert_eq!(final_boundary, vec![item("operator", "later")]);
    }

    #[test]
    fn completion_peek_ignores_empty_envelopes_and_includes_idle_items() {
        let mut mailbox = Mailbox::default();
        mailbox.push(MailboxInput::next_model_request(Vec::new()));
        assert!(!mailbox.has_deliverable_items(RequestBoundaryKind::BeforeModelRequest));
        assert!(!mailbox.has_deliverable_items(RequestBoundaryKind::BeforeCompletion));

        mailbox.push(MailboxInput::when_idle(vec![item("operator", "later")]));
        assert!(!mailbox.has_deliverable_items(RequestBoundaryKind::BeforeModelRequest));
        assert!(mailbox.has_deliverable_items(RequestBoundaryKind::BeforeCompletion));
    }

    #[test]
    fn rescheduling_preserves_order_and_yields_to_earlier_residents() {
        let mut mailbox = Mailbox::default();
        mailbox.push(MailboxInput::when_idle(vec![item("operator", "早就在等")]));
        mailbox.push(MailboxInput::next_model_request(vec![item("user", "一")]));
        mailbox.push(MailboxInput::next_model_request(vec![item("user", "二")]));

        assert_eq!(
            mailbox.reschedule(Delivery::NextModelRequest, Delivery::WhenIdle),
            2
        );
        // 改主意的内容排在原住民之后，彼此保持先后。
        assert_eq!(
            mailbox.drain(RequestBoundaryKind::BeforeCompletion),
            vec![
                item("operator", "早就在等"),
                item("user", "一"),
                item("user", "二"),
            ]
        );
    }

    #[test]
    fn rescheduled_content_leaves_the_boundary_it_came_from() {
        let mut mailbox = Mailbox::default();
        mailbox.push(MailboxInput::next_model_request(vec![item(
            "user",
            "先别急",
        )]));
        assert_eq!(
            mailbox.reschedule(Delivery::NextModelRequest, Delivery::WhenIdle),
            1
        );
        // 模型请求边界不该再看到它——这正是「先别急」的意思。
        assert!(mailbox
            .drain(RequestBoundaryKind::BeforeModelRequest)
            .is_empty());
        assert_eq!(
            mailbox.drain(RequestBoundaryKind::BeforeCompletion),
            vec![item("user", "先别急")]
        );
    }

    #[test]
    fn promoting_passive_content_makes_it_worth_waking_for() {
        let mut mailbox = Mailbox::default();
        mailbox.push(MailboxInput::passive(vec![item("tool", "迟到的结果")]));
        assert!(!mailbox.has_trigger_items(), "被动内容本身不构成开跑理由");

        assert_eq!(mailbox.reschedule(Delivery::Passive, Delivery::WhenIdle), 1);
        assert!(
            mailbox.has_trigger_items(),
            "显式提升之后它就值得叫醒会话了"
        );
    }

    #[test]
    fn rescheduling_a_class_onto_itself_changes_nothing() {
        let mut mailbox = Mailbox::default();
        mailbox.push(MailboxInput::next_model_request(vec![item("user", "一")]));
        assert_eq!(
            mailbox.reschedule(Delivery::NextModelRequest, Delivery::NextModelRequest),
            0
        );
        assert_eq!(
            mailbox.drain(RequestBoundaryKind::BeforeModelRequest),
            vec![item("user", "一")]
        );
    }
}
