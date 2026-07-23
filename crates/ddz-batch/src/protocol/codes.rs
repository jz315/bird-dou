//! Explicit protocol tags. Do not rely on Rust enum layout outside enums carrying `repr(u8)`.

use ddz_core::{
    DoubleAction, GameAction, LandlordSelectionState, Phase, PublicDoublingState, RevealAction,
    RevealTiming, Role, RobAction, SpringKind,
};

pub const NO_SEAT: i8 = -1;
pub const NO_RANK: u8 = u8::MAX;
pub const NO_U8: u8 = u8::MAX;
pub const NO_ACTION: u8 = u8::MAX;
pub const NO_EVENT_CODE: u8 = u8::MAX;

pub const ACTION_REVEAL: u8 = 0;
pub const ACTION_CALL: u8 = 1;
pub const ACTION_ROB: u8 = 2;
pub const ACTION_DOUBLE: u8 = 3;
pub const ACTION_PLAY: u8 = 4;

pub const EVENT_PLAYER: u8 = 0;
pub const EVENT_SYSTEM: u8 = 1;

pub const PHASE_PRE_DEAL: u8 = 0;
pub const PHASE_DEALING: u8 = 1;
pub const PHASE_CALLING: u8 = 2;
pub const PHASE_ROBBING: u8 = 3;
pub const PHASE_BOTTOM_REVEAL: u8 = 4;
pub const PHASE_POST_BOTTOM_REVEAL: u8 = 5;
pub const PHASE_DOUBLING: u8 = 6;
pub const PHASE_CARD_PLAY: u8 = 7;
pub const PHASE_TERMINAL: u8 = 8;

pub const ROLE_UNASSIGNED: u8 = 0;
pub const ROLE_LANDLORD: u8 = 1;
pub const ROLE_FARMER: u8 = 2;

pub const DECISION_NO: u8 = 0;
pub const DECISION_YES: u8 = 1;

pub const REVEAL_BEFORE_DEAL: u8 = 0;
pub const REVEAL_DURING_DEAL: u8 = 1;
pub const REVEAL_AFTER_BOTTOM: u8 = 2;

pub const LANDLORD_POST_BID: u8 = 0;
pub const LANDLORD_NOT_STARTED: u8 = 1;
pub const LANDLORD_CALLING: u8 = 2;
pub const LANDLORD_ROBBING: u8 = 3;
pub const LANDLORD_RESOLVED: u8 = 4;

pub const DOUBLING_DISABLED: u8 = 0;
pub const DOUBLING_NOT_STARTED: u8 = 1;
pub const DOUBLING_IN_PROGRESS: u8 = 2;
pub const DOUBLING_RESOLVED: u8 = 3;

pub const SPRING_NONE: u8 = 0;
pub const SPRING_LANDLORD: u8 = 1;
pub const SPRING_FARMER: u8 = 2;

pub const SYSTEM_DEAL_ROUND: u8 = 0;
pub const SYSTEM_REDEAL: u8 = 1;
pub const SYSTEM_LANDLORD_RESOLVED: u8 = 2;
pub const SYSTEM_BOTTOM_REVEALED: u8 = 3;
pub const SYSTEM_CARD_PLAY_STARTED: u8 = 4;

pub(crate) fn seat(value: ddz_core::Seat) -> i8 {
    i8::try_from(value.value()).expect("validated seat fits in i8")
}

pub(crate) const fn phase(value: Phase) -> u8 {
    match value {
        Phase::PreDeal => PHASE_PRE_DEAL,
        Phase::Dealing => PHASE_DEALING,
        Phase::Calling => PHASE_CALLING,
        Phase::Robbing => PHASE_ROBBING,
        Phase::BottomReveal => PHASE_BOTTOM_REVEAL,
        Phase::PostBottomReveal => PHASE_POST_BOTTOM_REVEAL,
        Phase::Doubling => PHASE_DOUBLING,
        Phase::CardPlay => PHASE_CARD_PLAY,
        Phase::Terminal => PHASE_TERMINAL,
    }
}

pub(crate) const fn role(value: Role) -> u8 {
    match value {
        Role::Unassigned => ROLE_UNASSIGNED,
        Role::Landlord => ROLE_LANDLORD,
        Role::Farmer => ROLE_FARMER,
    }
}

pub(crate) const fn action(value: GameAction) -> (u8, u8) {
    match value {
        GameAction::Reveal(RevealAction::Continue) => (ACTION_REVEAL, DECISION_NO),
        GameAction::Reveal(RevealAction::Reveal) => (ACTION_REVEAL, DECISION_YES),
        GameAction::Call(ddz_core::CallAction::Pass) => (ACTION_CALL, DECISION_NO),
        GameAction::Call(ddz_core::CallAction::CallLandlord) => (ACTION_CALL, DECISION_YES),
        GameAction::Rob(RobAction::Pass) => (ACTION_ROB, DECISION_NO),
        GameAction::Rob(RobAction::RobLandlord) => (ACTION_ROB, DECISION_YES),
        GameAction::Double(DoubleAction::Decline) => (ACTION_DOUBLE, DECISION_NO),
        GameAction::Double(DoubleAction::Double) => (ACTION_DOUBLE, DECISION_YES),
        GameAction::Play(_) => (ACTION_PLAY, NO_U8),
    }
}

pub(crate) const fn reveal_timing(value: RevealTiming) -> (u8, u8) {
    match value {
        RevealTiming::BeforeDeal => (REVEAL_BEFORE_DEAL, NO_U8),
        RevealTiming::DuringDeal { cards_received } => (REVEAL_DURING_DEAL, cards_received),
        RevealTiming::AfterBottom => (REVEAL_AFTER_BOTTOM, NO_U8),
    }
}

pub(crate) const fn landlord_state(value: &LandlordSelectionState) -> u8 {
    match value {
        LandlordSelectionState::PostBid { .. } => LANDLORD_POST_BID,
        LandlordSelectionState::NotStarted { .. } => LANDLORD_NOT_STARTED,
        LandlordSelectionState::Calling(_) => LANDLORD_CALLING,
        LandlordSelectionState::Robbing(_) => LANDLORD_ROBBING,
        LandlordSelectionState::Resolved(_) => LANDLORD_RESOLVED,
    }
}

pub(crate) const fn doubling_state(value: &PublicDoublingState) -> u8 {
    match value {
        PublicDoublingState::Disabled => DOUBLING_DISABLED,
        PublicDoublingState::NotStarted => DOUBLING_NOT_STARTED,
        PublicDoublingState::InProgress { .. } => DOUBLING_IN_PROGRESS,
        PublicDoublingState::Resolved { .. } => DOUBLING_RESOLVED,
    }
}

pub(crate) const fn spring(value: SpringKind) -> u8 {
    match value {
        SpringKind::None => SPRING_NONE,
        SpringKind::LandlordSpring => SPRING_LANDLORD,
        SpringKind::FarmerSpring => SPRING_FARMER,
    }
}
