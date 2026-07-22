use serde::{Deserialize, Serialize};

use crate::{GameAction, Seat};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlayerEvent {
    pub actor: Seat,
    pub action: GameAction,
}

/// Automatic public events. Hidden cards never appear in the event stream.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SystemEvent {
    DealRound { round: u8 },
    Redeal { from_attempt: u32, to_attempt: u32 },
    LandlordResolved { landlord: Seat },
    BottomRevealed,
    CardPlayStarted,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GameEventKind {
    Player(PlayerEvent),
    System(SystemEvent),
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameEvent {
    pub sequence: u32,
    pub attempt: u32,
    pub kind: GameEventKind,
}
