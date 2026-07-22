//! Game-state, action, history, observation, and step-result data types.

use serde::{Deserialize, Serialize};

use crate::{Move, RankCounts, Seat};

/// Observation schema emitted by the first card-play engine.
pub const OBSERVATION_SCHEMA_VERSION: u32 = 1;

/// High-level phase of a complete game.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// Landlord selection is in progress.
    Bidding,
    /// Optional post-bid doubling is in progress.
    Doubling,
    /// Players are taking card-play turns.
    CardPlay,
    /// A player has emptied their hand and payoff is final.
    Terminal,
}

/// One bidding choice in a complete-game profile.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BidAction {
    /// Decline to call, bid, or rob.
    Pass,
    /// Submit a positive score bid.
    Score(u8),
    /// Call landlord in a call/rob sequence.
    Call,
    /// Rob landlord after an earlier call.
    Rob,
}

/// One optional post-bid doubling choice.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DoubleAction {
    /// Keep the current stake.
    Decline,
    /// Multiply the stake by the configured factor.
    Double,
}

/// A normalized action across every game phase.
///
/// Declaration order is the stable phase-first sort order.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GameAction {
    /// Landlord-selection action.
    Bid(BidAction),
    /// Optional doubling action.
    Double(DoubleAction),
    /// Card-play action, including Pass.
    Play(Move),
}

/// Bidding state represented by the post-bid engine.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BidState {
    /// Bidding was resolved externally and is disabled for this environment.
    DisabledPostBid,
    /// Integer score bidding is active.
    Score {
        /// Highest score bid so far, or zero before the first positive bid.
        highest_bid: u8,
        /// Seat owning `highest_bid`.
        highest_bidder: Option<Seat>,
        /// Number of seats that have acted in this bidding round.
        turns_taken: u8,
    },
    /// Call/rob bidding is active.
    Rob {
        /// Current provisional landlord after a call or rob.
        candidate: Option<Seat>,
        /// Number of seats that have acted in this bidding round.
        turns_taken: u8,
        /// Number of successful rob actions after the initial call.
        rob_count: u8,
    },
    /// Landlord selection is complete; optional doubling may still be active.
    Resolved {
        /// Winning score bid, or one for call/rob mode.
        winning_bid: u8,
        /// Seats that selected the configured two-times double.
        doubled: [bool; 3],
        /// Number of seats that have completed their doubling choice.
        double_turns: u8,
    },
    /// Every seat passed and the configured caller must redeal.
    AllPass,
}

/// Counters sufficient to determine spring and anti-spring at terminal time.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpringState {
    /// Number of non-Pass plays made by the landlord.
    pub landlord_non_pass_plays: u8,
    /// Number of non-Pass plays made by both farmers combined.
    pub farmer_non_pass_plays: u8,
}

/// One auditable state-transition event.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GameEvent {
    /// Zero-based position in the complete event history.
    pub sequence: u32,
    /// Seat that took the action.
    pub actor: Seat,
    /// Normalized action.
    pub action: GameAction,
}

/// Public bidding event carried by an observation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BidEvent {
    /// Zero-based position in bidding history.
    pub sequence: u32,
    /// Seat that acted.
    pub actor: Seat,
    /// Public bidding choice.
    pub action: BidAction,
}

/// Public event carried by an observation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicEvent {
    /// Zero-based position in complete public history.
    pub sequence: u32,
    /// Seat that acted.
    pub actor: Seat,
    /// Public normalized action.
    pub action: GameAction,
}

/// Landlord-team role of one seat.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// No landlord exists yet during complete-game bidding.
    Unassigned,
    /// The solo landlord seat.
    Landlord,
    /// Either member of the farmer team.
    Farmer,
}

/// Complete authoritative state of one game.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GameState {
    /// Stable ID of the rule configuration that owns this state.
    pub rule_config_id: u32,
    /// Current phase.
    pub phase: Phase,
    /// Seat that must act next, or the winning seat after terminal transition.
    pub current_player: Seat,
    /// Resolved landlord seat.
    pub landlord: Option<Seat>,
    /// Private current hands for all seats.
    pub hands: [RankCounts; 3],
    /// Original three bottom cards, retained as public metadata.
    pub bottom_cards: RankCounts,
    /// Cumulative non-Pass cards played by each seat.
    pub played_cards: [RankCounts; 3],
    /// Cached current hand sizes.
    pub cards_left: [u8; 3],
    /// Active target move within the current trick.
    pub last_non_pass: Option<Move>,
    /// Seat that produced the active target move.
    pub last_non_pass_player: Option<Seat>,
    /// Number of consecutive Pass actions after the active target.
    pub consecutive_passes: u8,
    /// Landlord-selection state.
    pub bid_state: BidState,
    /// Base-two stake exponent accumulated from bombs and the rocket.
    pub multiplier_exp: u8,
    /// Number of bombs and rockets played.
    pub bomb_count: u8,
    /// Public counters used by spring rules.
    pub spring_state: SpringState,
    /// Complete public action history.
    pub history: Vec<GameEvent>,
    /// Whether payoff is final and no more actions are legal.
    pub terminal: bool,
    /// Zero-sum seat payoff using landlord/farmer base stakes.
    pub raw_payoff: [i32; 3],
}

/// Information-set-safe view for one player.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    /// Observation schema version.
    pub schema_version: u32,
    /// Current game phase.
    pub phase: Phase,
    /// Seat receiving this observation.
    pub observer: Seat,
    /// Team role of `observer`.
    pub role: Role,
    /// Observer's private current hand.
    pub own_hand: RankCounts,
    /// Public cumulative played cards for every seat.
    pub public_played: [RankCounts; 3],
    /// Public original bottom cards.
    pub public_bottom_cards: RankCounts,
    /// Union of both opponents' current hands without their allocation.
    pub unknown_pool: RankCounts,
    /// Public hand sizes.
    pub cards_left: [u8; 3],
    /// Seat that acts next, or the winner after termination.
    pub current_player: Seat,
    /// Resolved landlord seat.
    pub landlord: Option<Seat>,
    /// Current target move.
    pub last_non_pass: Option<Move>,
    /// Consecutive passes after the target.
    pub consecutive_passes: u8,
    /// Public bidding history; empty in the post-bid profile.
    pub bid_history: Vec<BidEvent>,
    /// Complete public action history.
    pub history: Vec<PublicEvent>,
    /// Base-two stake exponent.
    pub multiplier_exp: u8,
    /// Number of bombs and rockets played.
    pub bomb_count: u8,
}

/// Result of one successful state transition.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StepResult {
    /// Event appended by this transition.
    pub event: GameEvent,
    /// Next acting seat, absent after terminal transition.
    pub next_player: Option<Seat>,
    /// Whether this transition ended the game.
    pub terminal: bool,
    /// Raw zero-sum seat payoff; zero before terminal time.
    pub raw_payoff: [i32; 3],
    /// Seat-wise terminal objective selected by the rule configuration.
    pub objective_payoff: [i32; 3],
}
