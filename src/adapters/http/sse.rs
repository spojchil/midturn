//! 不带自动重连的 Server-Sent Events 增量解析器。
//!
//! 自动重连可能重复工具调用片段，因此 HTTP 模型只解析当前连接；连接中断由上层作为一次
//! 未完成的模型尝试处理。

use std::sync::Arc;

use crate::types::AgentError;

use super::transport::model_error;

const UTF8_BOM: &[u8; 3] = b"\xef\xbb\xbf";

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SseEvent {
    pub event: String,
    pub data: String,
    /// SSE 的 `id` 会跨事件继承；共享底层字符串可避免一个较大的持久 ID 在同一网络
    /// chunk 中的每个事件上重复分配。
    pub id: Arc<str>,
}

/// 按字节接收网络分块，避免 UTF-8 字符被 HTTP chunk 切开时提前解码。
pub(crate) struct SseParser {
    pending: Vec<u8>,
    event_name: Vec<u8>,
    data: Vec<u8>,
    data_seen: bool,
    last_event_id: Arc<str>,
    beginning_checked: bool,
    max_event_bytes: usize,
}

impl SseParser {
    pub(crate) fn new(max_event_bytes: usize) -> Self {
        Self {
            pending: Vec::new(),
            event_name: Vec::new(),
            data: Vec::new(),
            data_seen: false,
            last_event_id: Arc::from(""),
            beginning_checked: false,
            max_event_bytes,
        }
    }

    pub(crate) fn push(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, AgentError> {
        self.pending.extend_from_slice(chunk);
        if self.pending.len() > self.max_event_bytes && !contains_line_ending(&self.pending) {
            return Err(model_error("transport_sse_event_too_large"));
        }
        self.strip_optional_bom();
        let events = self.parse_lines(false)?;
        if self.pending.len() > self.max_event_bytes {
            return Err(model_error("transport_sse_event_too_large"));
        }
        Ok(events)
    }

    /// 解析最后一个完整行，并拒绝尚未由空行封口的 data 事件。
    pub(crate) fn finish(mut self) -> Result<Vec<SseEvent>, AgentError> {
        self.strip_optional_bom();
        let events = self.parse_lines(true)?;
        if self.data_seen {
            return Err(model_error("transport_sse_truncated_event"));
        }
        Ok(events)
    }

    fn strip_optional_bom(&mut self) {
        if self.beginning_checked {
            return;
        }
        let compared = self.pending.len().min(UTF8_BOM.len());
        if self.pending[..compared] != UTF8_BOM[..compared] {
            self.beginning_checked = true;
            return;
        }
        if self.pending.len() >= UTF8_BOM.len() {
            self.pending.drain(..UTF8_BOM.len());
            self.beginning_checked = true;
        }
    }

    fn parse_lines(&mut self, eof: bool) -> Result<Vec<SseEvent>, AgentError> {
        // 一次接管当前缓冲区并用游标扫描。旧实现为每行创建 Vec 后再从头 drain，单个
        // 大 chunk 含很多短行时会反复移动剩余字节，形成 O(n^2) CPU/内存带宽开销。
        let buffer = std::mem::take(&mut self.pending);
        let mut events = Vec::new();
        let mut cursor = 0;
        while cursor < buffer.len() {
            let Some(relative_end) = buffer[cursor..]
                .iter()
                .position(|byte| matches!(byte, b'\r' | b'\n'))
            else {
                break;
            };
            let line_end = cursor + relative_end;
            if buffer[line_end] == b'\r' && line_end + 1 == buffer.len() && !eof {
                break;
            }
            let delimiter_len =
                if buffer[line_end] == b'\r' && buffer.get(line_end + 1).copied() == Some(b'\n') {
                    2
                } else {
                    1
                };
            if let Some(event) = self.consume_line(&buffer[cursor..line_end])? {
                events.push(event);
            }
            cursor = line_end + delimiter_len;
        }
        if eof && cursor < buffer.len() {
            if let Some(event) = self.consume_line(&buffer[cursor..])? {
                events.push(event);
            }
            cursor = buffer.len();
        }
        if cursor < buffer.len() {
            self.pending.extend_from_slice(&buffer[cursor..]);
        }
        Ok(events)
    }

    fn consume_line(&mut self, line: &[u8]) -> Result<Option<SseEvent>, AgentError> {
        // SSE 整条字节流都必须是 UTF-8，未知字段和注释也不能成为绕过点。
        ensure_utf8(line)?;
        if line.is_empty() {
            return self.dispatch();
        }
        if line[0] == b':' {
            return Ok(None);
        }

        let (field, mut value) = match line.iter().position(|byte| *byte == b':') {
            Some(index) => (&line[..index], &line[index + 1..]),
            None => (line, &[][..]),
        };
        if value.first() == Some(&b' ') {
            value = &value[1..];
        }

        match field {
            b"event" => {
                ensure_utf8(value)?;
                if value.len() > self.max_event_bytes {
                    return Err(model_error("transport_sse_event_too_large"));
                }
                self.event_name.clear();
                self.event_name.extend_from_slice(value);
            }
            b"data" => {
                ensure_utf8(value)?;
                let new_len = self
                    .data
                    .len()
                    .checked_add(value.len().saturating_add(1))
                    .ok_or_else(|| model_error("transport_sse_event_too_large"))?;
                if new_len > self.max_event_bytes {
                    return Err(model_error("transport_sse_event_too_large"));
                }
                self.data.extend_from_slice(value);
                self.data.push(b'\n');
                self.data_seen = true;
            }
            b"id" if !value.contains(&0) => {
                let value = std::str::from_utf8(value)
                    .map_err(|_| model_error("transport_sse_not_utf8"))?;
                if value.len() > self.max_event_bytes {
                    return Err(model_error("transport_sse_event_too_large"));
                }
                self.last_event_id = Arc::from(value);
            }
            // retry 和未来字段不会改变当前连接中的协议语义。
            _ => {}
        }
        Ok(None)
    }

    fn dispatch(&mut self) -> Result<Option<SseEvent>, AgentError> {
        if !self.data_seen {
            self.event_name.clear();
            return Ok(None);
        }

        if self.data.last() == Some(&b'\n') {
            self.data.pop();
        }
        let event = if self.event_name.is_empty() {
            "message".to_owned()
        } else {
            String::from_utf8(std::mem::take(&mut self.event_name))
                .map_err(|_| model_error("transport_sse_not_utf8"))?
        };
        let data = String::from_utf8(std::mem::take(&mut self.data))
            .map_err(|_| model_error("transport_sse_not_utf8"))?;
        let id = Arc::clone(&self.last_event_id);
        self.data_seen = false;
        Ok(Some(SseEvent { event, data, id }))
    }
}

fn ensure_utf8(value: &[u8]) -> Result<(), AgentError> {
    std::str::from_utf8(value)
        .map(|_| ())
        .map_err(|_| model_error("transport_sse_not_utf8"))
}

fn contains_line_ending(bytes: &[u8]) -> bool {
    bytes.iter().any(|byte| matches!(byte, b'\r' | b'\n'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_every_network_split_including_utf8_boundaries() {
        let wire =
            "\u{feff}event: delta\r\nid: 7\r\ndata: {\"text\":\"你好😀\"}\r\n\r\n".as_bytes();
        for split in 0..=wire.len() {
            let mut parser = SseParser::new(1024);
            let mut events = parser.push(&wire[..split]).unwrap();
            events.extend(parser.push(&wire[split..]).unwrap());
            events.extend(parser.finish().unwrap());
            assert_eq!(
                events,
                vec![SseEvent {
                    event: "delta".to_owned(),
                    data: "{\"text\":\"你好😀\"}".to_owned(),
                    id: Arc::from("7"),
                }],
                "split={split}"
            );
        }
    }

    #[test]
    fn supports_comments_multiple_data_lines_and_all_line_endings() {
        let mut parser = SseParser::new(1024);
        let events = parser
            .push(b": ping\rdata: first\ndata: second\r\n\r\n")
            .unwrap();
        assert_eq!(
            events,
            vec![SseEvent {
                event: "message".to_owned(),
                data: "first\nsecond".to_owned(),
                id: Arc::from(""),
            }]
        );
        parser.finish().unwrap();
    }

    #[test]
    fn preserves_an_empty_data_event_and_persistent_id() {
        let mut parser = SseParser::new(1024);
        let events = parser.push(b"id: same\ndata:\n\ndata: next\n\n").unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "");
        assert_eq!(events[0].id.as_ref(), "same");
        assert_eq!(events[1].id.as_ref(), "same");
        assert!(Arc::ptr_eq(&events[0].id, &events[1].id));
    }

    #[test]
    fn shares_a_large_persistent_id_across_many_events_and_bounds_id_lines() {
        let id = "x".repeat(4096);
        let mut wire = format!("id: {id}\n").into_bytes();
        for _ in 0..2_000 {
            wire.extend_from_slice(b"data: x\n\n");
        }
        let mut parser = SseParser::new(8192);
        let events = parser.push(&wire).unwrap();
        assert_eq!(events.len(), 2_000);
        assert!(events
            .windows(2)
            .all(|events| Arc::ptr_eq(&events[0].id, &events[1].id)));

        let mut oversized = SseParser::new(4);
        assert_eq!(
            oversized.push(b"id: 12345\n").unwrap_err().summary,
            "transport_sse_event_too_large"
        );
    }

    #[test]
    fn rejects_invalid_utf8_truncated_events_and_event_overflow() {
        let mut invalid = SseParser::new(1024);
        assert_eq!(
            invalid.push(b"data: \xff\n\n").unwrap_err().summary,
            "transport_sse_not_utf8"
        );
        let mut invalid_comment = SseParser::new(1024);
        assert_eq!(
            invalid_comment.push(b": \xff\n\n").unwrap_err().summary,
            "transport_sse_not_utf8"
        );

        let mut truncated = SseParser::new(1024);
        truncated.push(b"data: partial\n").unwrap();
        assert_eq!(
            truncated.finish().unwrap_err().summary,
            "transport_sse_truncated_event"
        );

        let mut oversized = SseParser::new(4);
        assert_eq!(
            oversized.push(b"data: 12345").unwrap_err().summary,
            "transport_sse_event_too_large"
        );
    }
}
