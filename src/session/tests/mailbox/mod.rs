//! 信箱与触发：投递语义、边界排空、自动开始、预算、改期。

mod boundaries;
mod budget;
mod lifecycle;
mod reschedule;
mod validation;

use super::*;
