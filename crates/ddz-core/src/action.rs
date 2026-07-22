use serde::{Deserialize, Serialize};

use crate::Move;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RevealAction {
    Continue,
    Reveal,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CallAction {
    Pass,
    CallLandlord,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RobAction {
    Pass,
    RobLandlord,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DoubleAction {
    Decline,
    Double,
}

/// One player-controlled action. Automatic deal/redeal events are represented by `SystemEvent`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GameAction {
    Reveal(RevealAction),
    Call(CallAction),
    Rob(RobAction),
    Double(DoubleAction),
    Play(Move),
}
