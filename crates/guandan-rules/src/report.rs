use serde::{Deserialize, Serialize};

use crate::RuleConfig;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReportDuty {
    None,
    AnswerOnRequest,
    AnnounceImmediately,
}

pub const fn report_duty(remaining: usize, rules: &RuleConfig) -> ReportDuty {
    if remaining == 0 {
        ReportDuty::None
    } else if rules.report_remaining_six && remaining <= 6 {
        ReportDuty::AnnounceImmediately
    } else if rules.report_remaining_ten_on_request && remaining <= 10 {
        ReportDuty::AnswerOnRequest
    } else {
        ReportDuty::None
    }
}
