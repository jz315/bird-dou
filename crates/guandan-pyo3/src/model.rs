use guandan_rules::{Card, Move, MoveKind, Rank, Seat, Team};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ActionView {
    pub index: usize,
    pub kind: &'static str,
    pub cards: Vec<Card>,
    pub total_cards: usize,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct EventView {
    pub sequence: usize,
    pub actor: Seat,
    pub kind: &'static str,
    pub cards: Vec<Card>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TargetView {
    pub actor: Seat,
    pub kind: &'static str,
    pub cards: Vec<Card>,
}

impl TargetView {
    pub(crate) fn from_move(actor: Seat, movement: &Move) -> Self {
        Self {
            actor,
            kind: kind_code(*movement.kind()),
            cards: movement.cards().to_vec(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ResultView {
    pub finish_order: [Seat; 4],
    pub winning_team: Team,
    pub level_advance: u8,
    pub human_won: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct StateView {
    pub schema_version: u32,
    pub phase: &'static str,
    pub human_seat: Seat,
    pub human_turn: bool,
    pub current_player: Option<Seat>,
    pub level: Rank,
    pub hand: Vec<Card>,
    pub cards_left: [usize; 4],
    pub target: Option<TargetView>,
    pub legal_actions: Vec<ActionView>,
    pub recent_actions: Vec<EventView>,
    pub finish_order: Vec<Seat>,
    pub result: Option<ResultView>,
}

pub(crate) const fn kind_code(kind: MoveKind) -> &'static str {
    match kind {
        MoveKind::Single { .. } => "single",
        MoveKind::Pair { .. } => "pair",
        MoveKind::Triple { .. } => "triple",
        MoveKind::FullHouse { .. } => "full_house",
        MoveKind::Straight { .. } => "straight",
        MoveKind::PairStraight { .. } => "pair_straight",
        MoveKind::TripleStraight { .. } => "triple_straight",
        MoveKind::Bomb { .. } => "bomb",
        MoveKind::StraightFlush { .. } => "straight_flush",
        MoveKind::FourJokers => "four_jokers",
    }
}
