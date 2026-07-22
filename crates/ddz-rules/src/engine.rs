//! Deterministic post-bid game state and card-play transitions.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};

use ddz_core::{
    deserialize_game_state, serialize_game_state, validate_rank_counts, BidAction, BidEvent,
    BidState, CardError, DoubleAction, GameAction, GameEvent, GameState, Move, MoveKind,
    Observation, Phase, PublicEvent, RankCounts, Role, Seat, SpringState, StateCodecError,
    StepResult, EMPTY_RANK_COUNTS, OBSERVATION_SCHEMA_VERSION, RANK_COUNT,
};

use crate::{
    generate_follow_moves, generate_lead_moves, BiddingMode, GenerateMovesError, RewardMode,
    RuleConfig, RuleConfigError, RuleProfile,
};

const PLAYER_COUNT: usize = 3;
const PLAYER_COUNT_U8: u8 = 3;
const LANDLORD_CARD_COUNT: u8 = 20;
const FARMER_CARD_COUNT: u8 = 17;
const BOTTOM_CARD_COUNT: u8 = 3;
static NEXT_GAME_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

/// Executable card-play environment beginning after the landlord is assigned.
#[derive(Debug)]
pub struct PostBidGame {
    rules: RuleConfig,
    state: GameState,
    revision: u64,
    instance_id: u64,
}

impl Clone for PostBidGame {
    fn clone(&self) -> Self {
        Self {
            rules: self.rules,
            state: self.state.clone(),
            revision: self.revision,
            instance_id: next_game_instance_id(),
        }
    }
}

impl PartialEq for PostBidGame {
    fn eq(&self, other: &Self) -> bool {
        self.rules == other.rules && self.state == other.state && self.revision == other.revision
    }
}

impl Eq for PostBidGame {}

/// Compact inverse delta for one successful in-place transition.
///
/// The token owns no history vector or complete [`GameState`]. It must be
/// supplied to [`PostBidGame::undo`] in strict last-in-first-out order on the
/// game that produced it.
#[derive(Clone, Debug)]
#[must_use = "dropping an UndoToken makes that transition irreversible"]
pub struct UndoToken {
    instance_id: u64,
    rule_config_id: u32,
    before_revision: u64,
    after_revision: u64,
    history_len: usize,
    event: GameEvent,
    checkpoint: TransitionCheckpoint,
}

#[derive(Clone, Debug)]
struct TransitionCheckpoint {
    actor: Seat,
    phase: Phase,
    current_player: Seat,
    landlord: Option<Seat>,
    hands: [RankCounts; PLAYER_COUNT],
    played_cards: [RankCounts; PLAYER_COUNT],
    cards_left: [u8; PLAYER_COUNT],
    last_non_pass: Option<Move>,
    last_non_pass_player: Option<Seat>,
    consecutive_passes: u8,
    bid_state: BidState,
    multiplier_exp: u8,
    bomb_count: u8,
    spring_state: SpringState,
    terminal: bool,
    raw_payoff: [i32; PLAYER_COUNT],
}

impl PostBidGame {
    /// Build a post-bid game from a complete, already-dealt deck.
    ///
    /// `hands[landlord]` must already include `bottom_cards`. The landlord must
    /// hold 20 cards, each farmer 17, and the three hands must jointly contain
    /// every physical deck card by rank.
    ///
    /// # Errors
    ///
    /// Returns [`GameInitError`] for an invalid profile, seat, hand size, deck
    /// partition, or bottom-card assignment.
    pub fn new(
        hands: [RankCounts; PLAYER_COUNT],
        bottom_cards: RankCounts,
        landlord: Seat,
        rules: RuleConfig,
    ) -> Result<Self, GameInitError> {
        rules.validate().map_err(GameInitError::RuleConfig)?;
        if rules.profile != RuleProfile::DouzeroPostBid {
            return Err(GameInitError::UnsupportedProfile {
                profile: rules.profile,
            });
        }
        validate_seat(landlord).map_err(|_| GameInitError::InvalidLandlord { landlord })?;

        let landlord_index = usize::from(landlord);
        let mut cards_left = [0; PLAYER_COUNT];
        for (seat, hand) in (0_u8..).zip(hands.iter()) {
            validate_rank_counts(hand)
                .map_err(|source| GameInitError::InvalidHand { seat, source })?;
            let seat_index = usize::from(seat);
            cards_left[seat_index] = count_cards(hand);
            let expected = if seat_index == landlord_index {
                LANDLORD_CARD_COUNT
            } else {
                FARMER_CARD_COUNT
            };
            if cards_left[seat_index] != expected {
                return Err(GameInitError::WrongHandSize {
                    seat,
                    actual: cards_left[seat_index],
                    expected,
                });
            }
        }

        validate_rank_counts(&bottom_cards).map_err(GameInitError::InvalidBottomCards)?;
        let bottom_count = count_cards(&bottom_cards);
        if bottom_count != BOTTOM_CARD_COUNT {
            return Err(GameInitError::WrongBottomCardCount {
                actual: bottom_count,
            });
        }
        let landlord_hand = hands
            .get(landlord_index)
            .ok_or(GameInitError::InvalidLandlord { landlord })?;
        for (rank_id, rank) in (0_u8..).zip(0..RANK_COUNT) {
            if bottom_cards[rank] > landlord_hand[rank] {
                return Err(GameInitError::BottomCardNotHeldByLandlord {
                    rank_id,
                    bottom_count: bottom_cards[rank],
                    landlord_count: landlord_hand[rank],
                });
            }

            let actual = hands.iter().map(|hand| hand[rank]).sum::<u8>();
            let expected = if rank_id <= 12 { 4 } else { 1 };
            if actual != expected {
                return Err(GameInitError::DeckCountMismatch {
                    rank_id,
                    actual,
                    expected,
                });
            }
        }

        Ok(Self {
            rules,
            state: GameState {
                rule_config_id: rules.rule_config_id,
                phase: Phase::CardPlay,
                current_player: landlord,
                landlord: Some(landlord),
                hands,
                bottom_cards,
                played_cards: [EMPTY_RANK_COUNTS; PLAYER_COUNT],
                cards_left,
                last_non_pass: None,
                last_non_pass_player: None,
                consecutive_passes: 0,
                bid_state: BidState::DisabledPostBid,
                multiplier_exp: 0,
                bomb_count: 0,
                spring_state: SpringState::default(),
                history: Vec::new(),
                terminal: false,
                raw_payoff: [0; PLAYER_COUNT],
            },
            revision: 0,
            instance_id: next_game_instance_id(),
        })
    }

    /// Build a complete canonical game before landlord selection.
    ///
    /// Every player starts with 17 cards and the three bottom cards remain in a
    /// separate hidden container until bidding resolves.
    ///
    /// # Errors
    ///
    /// Returns [`GameInitError`] for an invalid profile, bidder, hand size,
    /// bottom-card container, or incomplete physical deck partition.
    pub fn new_complete(
        hands: [RankCounts; PLAYER_COUNT],
        bottom_cards: RankCounts,
        first_bidder: Seat,
        rules: RuleConfig,
    ) -> Result<Self, GameInitError> {
        rules.validate().map_err(GameInitError::RuleConfig)?;
        if rules.profile != RuleProfile::CanonicalFull {
            return Err(GameInitError::UnsupportedProfile {
                profile: rules.profile,
            });
        }
        validate_seat(first_bidder)
            .map_err(|_| GameInitError::InvalidFirstBidder { first_bidder })?;
        let mut cards_left = [0; PLAYER_COUNT];
        for (seat, hand) in (0_u8..).zip(hands.iter()) {
            validate_rank_counts(hand)
                .map_err(|source| GameInitError::InvalidHand { seat, source })?;
            cards_left[usize::from(seat)] = count_cards(hand);
            if cards_left[usize::from(seat)] != FARMER_CARD_COUNT {
                return Err(GameInitError::WrongHandSize {
                    seat,
                    actual: cards_left[usize::from(seat)],
                    expected: FARMER_CARD_COUNT,
                });
            }
        }
        validate_rank_counts(&bottom_cards).map_err(GameInitError::InvalidBottomCards)?;
        let bottom_count = count_cards(&bottom_cards);
        if bottom_count != BOTTOM_CARD_COUNT {
            return Err(GameInitError::WrongBottomCardCount {
                actual: bottom_count,
            });
        }
        for (rank_id, rank) in (0_u8..).zip(0..RANK_COUNT) {
            let actual = hands.iter().map(|hand| hand[rank]).sum::<u8>() + bottom_cards[rank];
            let expected = if rank_id <= 12 { 4 } else { 1 };
            if actual != expected {
                return Err(GameInitError::DeckCountMismatch {
                    rank_id,
                    actual,
                    expected,
                });
            }
        }
        let bid_state = match rules.bidding.mode {
            BiddingMode::Score => BidState::Score {
                highest_bid: 0,
                highest_bidder: None,
                turns_taken: 0,
            },
            BiddingMode::Rob => BidState::Rob {
                candidate: None,
                turns_taken: 0,
                rob_count: 0,
            },
            BiddingMode::Disabled => {
                return Err(GameInitError::UnsupportedBiddingMode);
            }
        };
        Ok(Self {
            rules,
            state: GameState {
                rule_config_id: rules.rule_config_id,
                phase: Phase::Bidding,
                current_player: first_bidder,
                landlord: None,
                hands,
                bottom_cards,
                played_cards: [EMPTY_RANK_COUNTS; PLAYER_COUNT],
                cards_left,
                last_non_pass: None,
                last_non_pass_player: None,
                consecutive_passes: 0,
                bid_state,
                multiplier_exp: 0,
                bomb_count: 0,
                spring_state: SpringState::default(),
                history: Vec::new(),
                terminal: false,
                raw_payoff: [0; PLAYER_COUNT],
            },
            revision: 0,
            instance_id: next_game_instance_id(),
        })
    }

    /// Immutable authoritative state.
    #[must_use]
    pub const fn state(&self) -> &GameState {
        &self.state
    }

    /// Rule configuration attached to this environment.
    #[must_use]
    pub const fn rules(&self) -> &RuleConfig {
        &self.rules
    }

    /// Whether no further actions may be taken.
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        self.state.terminal
    }

    /// Serialize the authoritative state with an explicit wire schema version.
    ///
    /// The execution-local undo revision is intentionally excluded. Restoring
    /// the bytes replays history and starts a fresh reversible engine instance.
    ///
    /// # Errors
    ///
    /// Returns [`StateCodecError`] if JSON envelope serialization fails.
    pub fn serialize_state(&self) -> Result<Vec<u8>, StateCodecError> {
        serialize_game_state(&self.state)
    }

    /// Restore a serialized state and prove every historical transition.
    ///
    /// # Errors
    ///
    /// Returns [`GameDeserializeError::Codec`] for an invalid wire envelope or
    /// [`GameDeserializeError::Restore`] when the decoded state cannot be
    /// reconstructed by replay under `rules`.
    pub fn deserialize_state(
        bytes: &[u8],
        rules: RuleConfig,
    ) -> Result<Self, GameDeserializeError> {
        let state = deserialize_game_state(bytes).map_err(GameDeserializeError::Codec)?;
        Self::from_state(&state, rules).map_err(GameDeserializeError::Restore)
    }

    /// Restore a decoded state by reconstructing its initial deal and replaying
    /// every recorded event under the supplied rules.
    ///
    /// # Errors
    ///
    /// Returns [`GameRestoreError`] for a configuration mismatch, malformed
    /// initial deal, discontinuous history, illegal replay action, or any final
    /// field that differs from the replayed authoritative state.
    pub fn from_state(state: &GameState, rules: RuleConfig) -> Result<Self, GameRestoreError> {
        if state.rule_config_id != rules.rule_config_id {
            return Err(GameRestoreError::RuleConfigIdMismatch {
                state: state.rule_config_id,
                rules: rules.rule_config_id,
            });
        }
        let mut initial_hands = [EMPTY_RANK_COUNTS; PLAYER_COUNT];
        for (seat, ((initial_hand, current_hand), played_hand)) in (0_u8..).zip(
            initial_hands
                .iter_mut()
                .zip(state.hands.iter())
                .zip(state.played_cards.iter()),
        ) {
            for (rank_id, ((initial_count, &current_count), &played_count)) in (0_u8..).zip(
                initial_hand
                    .iter_mut()
                    .zip(current_hand.iter())
                    .zip(played_hand.iter()),
            ) {
                *initial_count = current_count
                    .checked_add(played_count)
                    .ok_or(GameRestoreError::InitialCountOverflow { seat, rank_id })?;
            }
        }
        let mut replay = match rules.profile {
            RuleProfile::DouzeroPostBid => {
                let landlord = state.landlord.ok_or(GameRestoreError::MissingLandlord)?;
                Self::new(initial_hands, state.bottom_cards, landlord, rules)
                    .map_err(GameRestoreError::InitialState)?
            }
            RuleProfile::CanonicalFull => {
                if let Some(landlord) = state.landlord {
                    let landlord_hand = &mut initial_hands[usize::from(landlord)];
                    for (rank_id, count) in landlord_hand.iter_mut().enumerate() {
                        *count = count.checked_sub(state.bottom_cards[rank_id]).ok_or(
                            GameRestoreError::BottomCardSubtraction {
                                landlord,
                                rank_id: u8::try_from(rank_id).unwrap_or_default(),
                            },
                        )?;
                    }
                }
                let first_bidder = state
                    .history
                    .first()
                    .map_or(state.current_player, |event| event.actor);
                Self::new_complete(initial_hands, state.bottom_cards, first_bidder, rules)
                    .map_err(GameRestoreError::InitialState)?
            }
        };
        for (expected_sequence, event) in state.history.iter().enumerate() {
            let expected_sequence =
                u32::try_from(expected_sequence).map_err(|_| GameRestoreError::HistoryTooLong)?;
            if event.sequence != expected_sequence {
                return Err(GameRestoreError::HistorySequence {
                    expected: expected_sequence,
                    actual: event.sequence,
                });
            }
            if event.actor != replay.state.current_player {
                return Err(GameRestoreError::HistoryActor {
                    sequence: event.sequence,
                    expected: replay.state.current_player,
                    actual: event.actor,
                });
            }
            replay
                .step(&event.action)
                .map_err(|source| GameRestoreError::Replay {
                    sequence: event.sequence,
                    source,
                })?;
        }
        if &replay.state != state {
            return Err(GameRestoreError::StateMismatch);
        }
        Ok(replay)
    }

    /// Rebuild this card-play root with one information-set-consistent hidden allocation.
    ///
    /// Container A is the next absolute seat after `observer`; container B is
    /// inferred from the current two-opponent union. The resulting state is
    /// replay-validated, including public bottom-card ownership.
    ///
    /// # Errors
    ///
    /// Rejects invalid/non-acting observers, non-card-play states, invalid rank
    /// counts, capacity/rank conservation failures, and replay-invalid samples.
    pub fn with_hidden_assignment(
        &self,
        observer: Seat,
        assignment_a: RankCounts,
    ) -> Result<Self, HiddenSampleError> {
        validate_seat(observer).map_err(|_| HiddenSampleError::InvalidObserver { observer })?;
        if observer != self.state.current_player {
            return Err(HiddenSampleError::ObserverNotCurrent {
                observer,
                current_player: self.state.current_player,
            });
        }
        if self.state.phase != Phase::CardPlay {
            return Err(HiddenSampleError::WrongPhase {
                phase: self.state.phase,
            });
        }
        validate_rank_counts(&assignment_a).map_err(HiddenSampleError::InvalidAssignment)?;
        let seat_a = next_seat(observer);
        let seat_b = next_seat(seat_a);
        let index_a = usize::from(seat_a);
        let index_b = usize::from(seat_b);
        let actual_a = count_cards(&assignment_a);
        let expected_a = self.state.cards_left[index_a];
        if actual_a != expected_a {
            return Err(HiddenSampleError::CapacityMismatch {
                seat: seat_a,
                actual: actual_a,
                expected: expected_a,
            });
        }
        let mut assignment_b = EMPTY_RANK_COUNTS;
        for rank in 0..RANK_COUNT {
            let unknown = self.state.hands[index_a][rank] + self.state.hands[index_b][rank];
            assignment_b[rank] = unknown.checked_sub(assignment_a[rank]).ok_or(
                HiddenSampleError::RankConservation {
                    rank_id: u8::try_from(rank).unwrap_or_default(),
                    assigned_a: assignment_a[rank],
                    unknown,
                },
            )?;
        }
        let actual_b = count_cards(&assignment_b);
        let expected_b = self.state.cards_left[index_b];
        if actual_b != expected_b {
            return Err(HiddenSampleError::CapacityMismatch {
                seat: seat_b,
                actual: actual_b,
                expected: expected_b,
            });
        }
        let mut sampled = self.state.clone();
        sampled.hands[index_a] = assignment_a;
        sampled.hands[index_b] = assignment_b;
        Self::from_state(&sampled, self.rules).map_err(HiddenSampleError::Restore)
    }

    /// Generate legal card-play moves for the current player.
    ///
    /// Terminal states return an empty vector.
    ///
    /// # Errors
    ///
    /// Returns [`GameError::WrongPhase`] for a non-card-play, non-terminal state
    /// or [`GameError::MoveGeneration`] if the rule generator rejects state data.
    pub fn legal_moves(&self) -> Result<Vec<Move>, GameError> {
        if self.state.terminal {
            return Ok(Vec::new());
        }
        if self.state.phase != Phase::CardPlay {
            return Err(GameError::WrongPhase {
                phase: self.state.phase,
            });
        }

        let hand = self
            .state
            .hands
            .get(usize::from(self.state.current_player))
            .ok_or(GameError::StateInvariant(
                "current player is outside the hand array",
            ))?;
        match self.state.last_non_pass {
            Some(target) => generate_follow_moves(hand, &target, &self.rules),
            None => generate_lead_moves(hand, &self.rules),
        }
        .map_err(GameError::MoveGeneration)
    }

    /// Generate phase-tagged legal actions for the current player.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::legal_moves`].
    pub fn legal_actions(&self) -> Result<Vec<GameAction>, GameError> {
        match self.state.phase {
            Phase::Bidding => self.legal_bid_actions(),
            Phase::Doubling => Ok(vec![
                GameAction::Double(DoubleAction::Decline),
                GameAction::Double(DoubleAction::Double),
            ]),
            Phase::CardPlay => self
                .legal_moves()
                .map(|moves| moves.into_iter().map(GameAction::Play).collect::<Vec<_>>()),
            Phase::Terminal => Ok(Vec::new()),
        }
    }

    /// Build an information-set-safe observation for one seat.
    ///
    /// # Errors
    ///
    /// Returns [`GameError::InvalidSeat`] when `observer` is outside `0..=2`.
    pub fn observe(&self, observer: Seat) -> Result<Observation, GameError> {
        validate_seat(observer)?;
        let observer_index = usize::from(observer);
        let own_hand = *self
            .state
            .hands
            .get(observer_index)
            .ok_or(GameError::InvalidSeat { seat: observer })?;
        let mut unknown_pool = EMPTY_RANK_COUNTS;
        for (seat, hand) in self.state.hands.iter().enumerate() {
            if seat == observer_index {
                continue;
            }
            for rank in 0..RANK_COUNT {
                unknown_pool[rank] += hand[rank];
            }
        }
        if self.state.landlord.is_none() {
            for (unknown, bottom) in unknown_pool.iter_mut().zip(self.state.bottom_cards) {
                *unknown += bottom;
            }
        }

        let bid_history = self
            .state
            .history
            .iter()
            .filter_map(|event| match event.action {
                GameAction::Bid(action) => Some(BidEvent {
                    sequence: event.sequence,
                    actor: event.actor,
                    action,
                }),
                GameAction::Double(_) | GameAction::Play(_) => None,
            })
            .collect();
        let history = self
            .state
            .history
            .iter()
            .map(|event| PublicEvent {
                sequence: event.sequence,
                actor: event.actor,
                action: event.action,
            })
            .collect();

        Ok(Observation {
            schema_version: OBSERVATION_SCHEMA_VERSION,
            phase: self.state.phase,
            observer,
            role: match self.state.landlord {
                Some(landlord) if observer == landlord => Role::Landlord,
                Some(_) => Role::Farmer,
                None => Role::Unassigned,
            },
            own_hand,
            public_played: self.state.played_cards,
            public_bottom_cards: if self.state.landlord.is_some() && self.rules.bottom_cards_public
            {
                self.state.bottom_cards
            } else {
                EMPTY_RANK_COUNTS
            },
            unknown_pool,
            cards_left: self.state.cards_left,
            current_player: self.state.current_player,
            landlord: self.state.landlord,
            last_non_pass: self.state.last_non_pass,
            consecutive_passes: self.state.consecutive_passes,
            bid_history,
            history,
            multiplier_exp: self.state.multiplier_exp,
            bomb_count: self.state.bomb_count,
        })
    }

    /// Apply one legal action for the current player.
    ///
    /// # Errors
    ///
    /// Returns [`GameError::Terminal`] after game end,
    /// [`GameError::IllegalAction`] for an action outside the current legal set,
    /// or a generation/invariant error if the state cannot be advanced safely.
    pub fn step(&mut self, action: &GameAction) -> Result<StepResult, GameError> {
        self.apply_transition(action).map(|(result, _undo)| result)
    }

    /// Apply one legal action and return both its result and inverse delta.
    ///
    /// This is the transactional primitive used by batched callers that may
    /// need to roll back earlier environments if a later environment fails.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::step`].
    pub fn step_with_undo(
        &mut self,
        action: &GameAction,
    ) -> Result<(StepResult, UndoToken), GameError> {
        self.apply_transition(action)
    }

    /// Apply one action and return its compact inverse delta.
    ///
    /// The transition is transactional: on any error, state and history remain
    /// unchanged. The returned token must be passed to [`Self::undo`] before any
    /// other successful transition.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::step`].
    pub fn apply_in_place(&mut self, action: &GameAction) -> Result<UndoToken, GameError> {
        self.step_with_undo(action).map(|(_result, undo)| undo)
    }

    /// Revert the most recent transition using its inverse delta.
    ///
    /// # Errors
    ///
    /// Returns [`UndoError`] if the token belongs to another rule configuration,
    /// is stale or out of order, or no longer matches the history suffix.
    pub fn undo(&mut self, token: &UndoToken) -> Result<(), UndoError> {
        if token.instance_id != self.instance_id {
            return Err(UndoError::InstanceMismatch);
        }
        if token.rule_config_id != self.state.rule_config_id {
            return Err(UndoError::RuleConfigIdMismatch {
                token: token.rule_config_id,
                state: self.state.rule_config_id,
            });
        }
        if self.revision != token.after_revision {
            return Err(UndoError::RevisionMismatch {
                expected: token.after_revision,
                actual: self.revision,
            });
        }
        let expected_history_len = token
            .history_len
            .checked_add(1)
            .ok_or(UndoError::HistoryLengthOverflow)?;
        if self.state.history.len() != expected_history_len {
            return Err(UndoError::HistoryLengthMismatch {
                expected: expected_history_len,
                actual: self.state.history.len(),
            });
        }
        if self.state.history.last() != Some(&token.event) {
            return Err(UndoError::HistoryEventMismatch);
        }
        self.restore_checkpoint(token)
            .ok_or(UndoError::InvalidActor {
                actor: token.checkpoint.actor,
            })?;
        Ok(())
    }

    fn apply_transition(
        &mut self,
        action: &GameAction,
    ) -> Result<(StepResult, UndoToken), GameError> {
        if self.state.terminal {
            return Err(GameError::Terminal);
        }
        let actor = self.state.current_player;
        if !self.legal_actions()?.contains(action) {
            return Err(GameError::IllegalAction {
                actor,
                action: *action,
            });
        }

        let sequence = u32::try_from(self.state.history.len())
            .map_err(|_| GameError::StateInvariant("history length exceeds u32"))?;
        let event = GameEvent {
            sequence,
            actor,
            action: *action,
        };
        let after_revision = self
            .revision
            .checked_add(1)
            .ok_or(GameError::StateInvariant("undo revision counter overflow"))?;
        let undo = self.capture_undo(actor, event, after_revision)?;

        let transition = (|| {
            match *action {
                GameAction::Bid(bid) => self.apply_bid(actor, bid)?,
                GameAction::Double(double) => self.apply_double(actor, double)?,
                GameAction::Play(played_move) => {
                    if played_move.kind() == MoveKind::Pass {
                        self.apply_pass(actor)?;
                    } else {
                        self.apply_non_pass(actor, played_move)?;
                    }
                }
            }
            self.state.history.push(event);
            let objective_payoff = self.objective_payoff()?;
            Ok(StepResult {
                event,
                next_player: (!self.state.terminal).then_some(self.state.current_player),
                terminal: self.state.terminal,
                raw_payoff: self.state.raw_payoff,
                objective_payoff,
            })
        })();

        match transition {
            Ok(result) => {
                self.revision = after_revision;
                Ok((result, undo))
            }
            Err(error) => {
                if self.restore_checkpoint(&undo).is_none() {
                    return Err(GameError::StateInvariant(
                        "failed to roll back rejected transition",
                    ));
                }
                Err(error)
            }
        }
    }

    fn legal_bid_actions(&self) -> Result<Vec<GameAction>, GameError> {
        match self.state.bid_state {
            BidState::Score { highest_bid, .. } => {
                let maximum = self.rules.bidding.max_bid.ok_or(GameError::StateInvariant(
                    "score bidding has no configured maximum",
                ))?;
                let mut actions = Vec::with_capacity(usize::from(maximum - highest_bid) + 1);
                actions.push(GameAction::Bid(BidAction::Pass));
                for score in (highest_bid + 1)..=maximum {
                    actions.push(GameAction::Bid(BidAction::Score(score)));
                }
                Ok(actions)
            }
            BidState::Rob { candidate, .. } => Ok(vec![
                GameAction::Bid(BidAction::Pass),
                GameAction::Bid(if candidate.is_none() {
                    BidAction::Call
                } else {
                    BidAction::Rob
                }),
            ]),
            BidState::DisabledPostBid | BidState::Resolved { .. } | BidState::AllPass => Err(
                GameError::StateInvariant("bidding phase has a non-bidding bid state"),
            ),
        }
    }

    fn apply_bid(&mut self, actor: Seat, action: BidAction) -> Result<(), GameError> {
        match self.state.bid_state {
            BidState::Score {
                highest_bid,
                highest_bidder,
                turns_taken,
            } => {
                let (next_bid, next_bidder) = match action {
                    BidAction::Pass => (highest_bid, highest_bidder),
                    BidAction::Score(score) => (score, Some(actor)),
                    BidAction::Call | BidAction::Rob => {
                        return Err(GameError::StateInvariant(
                            "score bidding admitted a call/rob action",
                        ));
                    }
                };
                let next_turns = turns_taken
                    .checked_add(1)
                    .ok_or(GameError::StateInvariant("bidding turn counter overflow"))?;
                let maximum = self.rules.bidding.max_bid.ok_or(GameError::StateInvariant(
                    "score bidding has no configured maximum",
                ))?;
                if next_bid == maximum || next_turns == PLAYER_COUNT_U8 {
                    if let Some(landlord) = next_bidder {
                        self.resolve_landlord(landlord, next_bid, 0)
                    } else {
                        self.finish_all_pass(actor);
                        Ok(())
                    }
                } else {
                    self.state.bid_state = BidState::Score {
                        highest_bid: next_bid,
                        highest_bidder: next_bidder,
                        turns_taken: next_turns,
                    };
                    self.state.current_player = next_seat(actor);
                    Ok(())
                }
            }
            BidState::Rob {
                candidate,
                turns_taken,
                rob_count,
            } => {
                let (next_candidate, next_rob_count) = match action {
                    BidAction::Pass => (candidate, rob_count),
                    BidAction::Call if candidate.is_none() => (Some(actor), rob_count),
                    BidAction::Rob if candidate.is_some() => (
                        Some(actor),
                        rob_count
                            .checked_add(1)
                            .ok_or(GameError::StateInvariant("rob counter overflow"))?,
                    ),
                    BidAction::Score(_) | BidAction::Call | BidAction::Rob => {
                        return Err(GameError::StateInvariant(
                            "call/rob bidding admitted an inconsistent action",
                        ));
                    }
                };
                let next_turns = turns_taken
                    .checked_add(1)
                    .ok_or(GameError::StateInvariant("bidding turn counter overflow"))?;
                if next_turns == PLAYER_COUNT_U8 {
                    if let Some(landlord) = next_candidate {
                        self.resolve_landlord(landlord, 1, next_rob_count)
                    } else {
                        self.finish_all_pass(actor);
                        Ok(())
                    }
                } else {
                    self.state.bid_state = BidState::Rob {
                        candidate: next_candidate,
                        turns_taken: next_turns,
                        rob_count: next_rob_count,
                    };
                    self.state.current_player = next_seat(actor);
                    Ok(())
                }
            }
            BidState::DisabledPostBid | BidState::Resolved { .. } | BidState::AllPass => Err(
                GameError::StateInvariant("bid transition has a non-bidding bid state"),
            ),
        }
    }

    fn resolve_landlord(
        &mut self,
        landlord: Seat,
        winning_bid: u8,
        rob_count: u8,
    ) -> Result<(), GameError> {
        let landlord_index = usize::from(landlord);
        for rank in 0..RANK_COUNT {
            self.state.hands[landlord_index][rank] = self.state.hands[landlord_index][rank]
                .checked_add(self.state.bottom_cards[rank])
                .ok_or(GameError::StateInvariant("landlord hand count overflow"))?;
        }
        self.state.cards_left[landlord_index] = self.state.cards_left[landlord_index]
            .checked_add(BOTTOM_CARD_COUNT)
            .ok_or(GameError::StateInvariant("landlord card count overflow"))?;
        self.state.landlord = Some(landlord);
        self.state.bid_state = BidState::Resolved {
            winning_bid,
            doubled: [false; PLAYER_COUNT],
            double_turns: 0,
        };
        self.state.multiplier_exp = self
            .state
            .multiplier_exp
            .checked_add(rob_count)
            .ok_or(GameError::StateInvariant("rob multiplier overflow"))?;
        if self.rules.doubling_enabled {
            self.state.phase = Phase::Doubling;
            self.state.current_player = landlord;
        } else {
            self.start_card_play()?;
        }
        Ok(())
    }

    fn finish_all_pass(&mut self, actor: Seat) {
        self.state.bid_state = BidState::AllPass;
        self.state.phase = Phase::Terminal;
        self.state.current_player = actor;
        self.state.terminal = true;
        self.state.raw_payoff = [0; PLAYER_COUNT];
    }

    fn apply_double(&mut self, actor: Seat, action: DoubleAction) -> Result<(), GameError> {
        let BidState::Resolved {
            winning_bid,
            mut doubled,
            double_turns,
        } = self.state.bid_state
        else {
            return Err(GameError::StateInvariant(
                "doubling transition has an unresolved bid state",
            ));
        };
        if action == DoubleAction::Double {
            doubled[usize::from(actor)] = true;
            self.state.multiplier_exp = self
                .state
                .multiplier_exp
                .checked_add(1)
                .ok_or(GameError::StateInvariant("double multiplier overflow"))?;
        }
        let next_turns = double_turns
            .checked_add(1)
            .ok_or(GameError::StateInvariant("doubling turn counter overflow"))?;
        self.state.bid_state = BidState::Resolved {
            winning_bid,
            doubled,
            double_turns: next_turns,
        };
        if next_turns == PLAYER_COUNT_U8 {
            self.start_card_play()
        } else {
            self.state.current_player = next_seat(actor);
            Ok(())
        }
    }

    fn start_card_play(&mut self) -> Result<(), GameError> {
        let landlord = self
            .state
            .landlord
            .ok_or(GameError::StateInvariant("resolved bid has no landlord"))?;
        self.state.phase = Phase::CardPlay;
        self.state.current_player = if self.rules.landlord_plays_first {
            landlord
        } else {
            next_seat(landlord)
        };
        Ok(())
    }

    fn capture_undo(
        &self,
        actor: Seat,
        event: GameEvent,
        after_revision: u64,
    ) -> Result<UndoToken, GameError> {
        validate_seat(actor)?;
        Ok(UndoToken {
            instance_id: self.instance_id,
            rule_config_id: self.state.rule_config_id,
            before_revision: self.revision,
            after_revision,
            history_len: self.state.history.len(),
            event,
            checkpoint: TransitionCheckpoint {
                actor,
                phase: self.state.phase,
                current_player: self.state.current_player,
                landlord: self.state.landlord,
                hands: self.state.hands,
                played_cards: self.state.played_cards,
                cards_left: self.state.cards_left,
                last_non_pass: self.state.last_non_pass,
                last_non_pass_player: self.state.last_non_pass_player,
                consecutive_passes: self.state.consecutive_passes,
                bid_state: self.state.bid_state,
                multiplier_exp: self.state.multiplier_exp,
                bomb_count: self.state.bomb_count,
                spring_state: self.state.spring_state,
                terminal: self.state.terminal,
                raw_payoff: self.state.raw_payoff,
            },
        })
    }

    fn restore_checkpoint(&mut self, token: &UndoToken) -> Option<()> {
        validate_seat(token.checkpoint.actor).ok()?;
        self.state.landlord = token.checkpoint.landlord;
        self.state.hands = token.checkpoint.hands;
        self.state.played_cards = token.checkpoint.played_cards;
        self.state.cards_left = token.checkpoint.cards_left;
        self.state.phase = token.checkpoint.phase;
        self.state.current_player = token.checkpoint.current_player;
        self.state.last_non_pass = token.checkpoint.last_non_pass;
        self.state.last_non_pass_player = token.checkpoint.last_non_pass_player;
        self.state.consecutive_passes = token.checkpoint.consecutive_passes;
        self.state.bid_state = token.checkpoint.bid_state;
        self.state.multiplier_exp = token.checkpoint.multiplier_exp;
        self.state.bomb_count = token.checkpoint.bomb_count;
        self.state.spring_state = token.checkpoint.spring_state;
        self.state.terminal = token.checkpoint.terminal;
        self.state.raw_payoff = token.checkpoint.raw_payoff;
        self.state.history.truncate(token.history_len);
        self.revision = token.before_revision;
        Some(())
    }

    fn apply_pass(&mut self, actor: Seat) -> Result<(), GameError> {
        if self.state.last_non_pass.is_none() {
            return Err(GameError::StateInvariant(
                "free-lead state admitted a Pass action",
            ));
        }
        self.state.consecutive_passes =
            self.state
                .consecutive_passes
                .checked_add(1)
                .ok_or(GameError::StateInvariant(
                    "consecutive pass counter overflow",
                ))?;
        self.state.current_player = next_seat(actor);

        if self.state.consecutive_passes == 2 {
            self.state.last_non_pass = None;
            self.state.last_non_pass_player = None;
            self.state.consecutive_passes = 0;
        }
        Ok(())
    }

    fn apply_non_pass(&mut self, actor: Seat, played_move: Move) -> Result<(), GameError> {
        let actor_index = usize::from(actor);
        for rank in 0..RANK_COUNT {
            let required = played_move.cards()[rank];
            let available = self.state.hands[actor_index][rank];
            if required > available {
                return Err(GameError::StateInvariant(
                    "legal move consumes unavailable cards",
                ));
            }
            self.state.hands[actor_index][rank] = available - required;
            self.state.played_cards[actor_index][rank] += required;
        }
        self.state.cards_left[actor_index] = self.state.cards_left[actor_index]
            .checked_sub(played_move.total_cards())
            .ok_or(GameError::StateInvariant("cached card count underflow"))?;
        self.state.last_non_pass = Some(played_move);
        self.state.last_non_pass_player = Some(actor);
        self.state.consecutive_passes = 0;

        if matches!(played_move.kind(), MoveKind::Bomb | MoveKind::Rocket) {
            self.state.bomb_count = self
                .state
                .bomb_count
                .checked_add(1)
                .ok_or(GameError::StateInvariant("bomb counter overflow"))?;
            let factor = if played_move.kind() == MoveKind::Rocket {
                self.rules.rocket_multiplier
            } else {
                self.rules.bomb_multiplier
            };
            self.state.multiplier_exp = self
                .state
                .multiplier_exp
                .checked_add(multiplier_exponent(factor)?)
                .ok_or(GameError::StateInvariant("multiplier exponent overflow"))?;
        }

        let landlord = self
            .state
            .landlord
            .ok_or(GameError::StateInvariant("post-bid state has no landlord"))?;
        let play_counter = if actor == landlord {
            &mut self.state.spring_state.landlord_non_pass_plays
        } else {
            &mut self.state.spring_state.farmer_non_pass_plays
        };
        *play_counter = play_counter
            .checked_add(1)
            .ok_or(GameError::StateInvariant("spring play counter overflow"))?;

        if self.state.cards_left[actor_index] == 0 {
            self.finish(actor)?;
        } else {
            self.state.current_player = next_seat(actor);
        }
        Ok(())
    }

    fn finish(&mut self, winner: Seat) -> Result<(), GameError> {
        self.state.terminal = true;
        self.state.phase = Phase::Terminal;
        self.state.current_player = winner;

        let landlord = self
            .state
            .landlord
            .ok_or(GameError::StateInvariant("post-bid state has no landlord"))?;
        let landlord_won = winner == landlord;
        let spring = (landlord_won
            && self.rules.spring.landlord_spring_enabled
            && self.state.spring_state.farmer_non_pass_plays == 0)
            || (!landlord_won
                && self.rules.spring.anti_spring_enabled
                && self.state.spring_state.landlord_non_pass_plays == 1);
        if spring {
            self.state.multiplier_exp = self
                .state
                .multiplier_exp
                .checked_add(multiplier_exponent(self.rules.spring.multiplier)?)
                .ok_or(GameError::StateInvariant("spring multiplier overflow"))?;
        }
        let bid = match self.state.bid_state {
            BidState::DisabledPostBid => 1,
            BidState::Resolved { winning_bid, .. } => i32::from(winning_bid),
            BidState::Score { .. } | BidState::Rob { .. } | BidState::AllPass => {
                return Err(GameError::StateInvariant(
                    "terminal card play has unresolved bidding",
                ));
            }
        };
        let unit_stake = checked_power_of_two(self.state.multiplier_exp)?
            .checked_mul(bid)
            .ok_or(GameError::StateInvariant("raw score exceeds i32"))?;
        let unit_stake = self.rules.score_cap.map_or(unit_stake, |cap| {
            let landlord_cap = i32::try_from(cap).unwrap_or(i32::MAX);
            unit_stake.min(landlord_cap / 2)
        });
        for seat in 0..PLAYER_COUNT {
            let is_landlord = seat == usize::from(landlord);
            let won = if is_landlord {
                landlord_won
            } else {
                !landlord_won
            };
            let base = if is_landlord { 2 } else { 1 };
            self.state.raw_payoff[seat] = if won {
                base * unit_stake
            } else {
                -base * unit_stake
            };
        }
        Ok(())
    }

    fn objective_payoff(&self) -> Result<[i32; PLAYER_COUNT], GameError> {
        if !self.state.terminal {
            return Ok([0; PLAYER_COUNT]);
        }
        if self.rules.reward_mode == RewardMode::RawScore {
            return Ok(self.state.raw_payoff);
        }

        let landlord = self
            .state
            .landlord
            .ok_or(GameError::StateInvariant("post-bid state has no landlord"))?;
        let landlord_won = self.state.raw_payoff[usize::from(landlord)] > 0;
        let magnitude = match self.rules.reward_mode {
            RewardMode::WinPercentage => 1,
            RewardMode::AverageDifferencePoints => checked_power_of_two(self.state.bomb_count)?,
            RewardMode::LogAverageDifferencePoints => i32::from(self.state.bomb_count) + 1,
            RewardMode::RawScore => unreachable!("raw score returned before objective mapping"),
        };
        Ok(std::array::from_fn(|seat| {
            let seat_won = if seat == usize::from(landlord) {
                landlord_won
            } else {
                !landlord_won
            };
            if seat_won {
                magnitude
            } else {
                -magnitude
            }
        }))
    }
}

const fn next_seat(seat: Seat) -> Seat {
    (seat + 1) % PLAYER_COUNT_U8
}

fn validate_seat(seat: Seat) -> Result<(), GameError> {
    if usize::from(seat) < PLAYER_COUNT {
        Ok(())
    } else {
        Err(GameError::InvalidSeat { seat })
    }
}

fn count_cards(cards: &RankCounts) -> u8 {
    cards.iter().sum()
}

fn checked_power_of_two(exponent: u8) -> Result<i32, GameError> {
    1_i32
        .checked_shl(u32::from(exponent))
        .ok_or(GameError::StateInvariant(
            "multiplier exponent exceeds i32 payoff capacity",
        ))
}

fn multiplier_exponent(factor: u32) -> Result<u8, GameError> {
    if !factor.is_power_of_two() {
        return Err(GameError::StateInvariant(
            "configured multiplier is not a power of two",
        ));
    }
    u8::try_from(factor.trailing_zeros())
        .map_err(|_| GameError::StateInvariant("multiplier exponent exceeds u8"))
}

fn next_game_instance_id() -> u64 {
    NEXT_GAME_INSTANCE_ID.fetch_add(1, Ordering::Relaxed)
}

/// Errors produced while constructing a post-bid game.
#[derive(Debug)]
pub enum GameInitError {
    /// The supplied rule configuration is invalid.
    RuleConfig(RuleConfigError),
    /// E007 accepts only the externally resolved `DouZero` profile.
    UnsupportedProfile {
        /// Rejected profile.
        profile: RuleProfile,
    },
    /// Landlord seat fell outside `0..=2`.
    InvalidLandlord {
        /// Rejected seat.
        landlord: Seat,
    },
    /// First bidding seat fell outside `0..=2`.
    InvalidFirstBidder {
        /// Rejected seat.
        first_bidder: Seat,
    },
    /// Complete-game construction received disabled bidding.
    UnsupportedBiddingMode,
    /// One hand exceeded physical rank capacity.
    InvalidHand {
        /// Seat owning the invalid hand.
        seat: Seat,
        /// Card validation failure.
        source: CardError,
    },
    /// A hand did not contain 20 landlord cards or 17 farmer cards.
    WrongHandSize {
        /// Seat owning the hand.
        seat: Seat,
        /// Actual number of cards.
        actual: u8,
        /// Required number of cards.
        expected: u8,
    },
    /// Bottom-card counts exceeded physical capacity.
    InvalidBottomCards(CardError),
    /// Bottom metadata did not contain exactly three cards.
    WrongBottomCardCount {
        /// Actual card count.
        actual: u8,
    },
    /// Bottom metadata named a card rank absent from the landlord hand.
    BottomCardNotHeldByLandlord {
        /// Rank containing the mismatch.
        rank_id: u8,
        /// Count declared as bottom cards.
        bottom_count: u8,
        /// Count present in the landlord hand.
        landlord_count: u8,
    },
    /// The three hands did not partition a complete deck.
    DeckCountMismatch {
        /// Rank containing the mismatch.
        rank_id: u8,
        /// Cards present across all hands.
        actual: u8,
        /// Physical deck count required for the rank.
        expected: u8,
    },
}

impl Display for GameInitError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RuleConfig(error) => Display::fmt(error, formatter),
            Self::UnsupportedProfile { profile } => {
                write!(formatter, "post-bid engine does not support profile {profile:?}")
            }
            Self::InvalidLandlord { landlord } => {
                write!(formatter, "landlord seat {landlord} is outside 0..=2")
            }
            Self::InvalidFirstBidder { first_bidder } => {
                write!(formatter, "first bidder seat {first_bidder} is outside 0..=2")
            }
            Self::UnsupportedBiddingMode => {
                write!(formatter, "complete game requires score or call/rob bidding")
            }
            Self::InvalidHand { seat, source } => {
                write!(formatter, "invalid hand for seat {seat}: {source}")
            }
            Self::WrongHandSize {
                seat,
                actual,
                expected,
            } => write!(
                formatter,
                "seat {seat} holds {actual} cards; expected {expected}"
            ),
            Self::InvalidBottomCards(error) => {
                write!(formatter, "invalid bottom cards: {error}")
            }
            Self::WrongBottomCardCount { actual } => {
                write!(formatter, "bottom metadata contains {actual} cards; expected 3")
            }
            Self::BottomCardNotHeldByLandlord {
                rank_id,
                bottom_count,
                landlord_count,
            } => write!(
                formatter,
                "bottom rank {rank_id} has count {bottom_count}, but landlord holds {landlord_count}"
            ),
            Self::DeckCountMismatch {
                rank_id,
                actual,
                expected,
            } => write!(
                formatter,
                "dealt rank {rank_id} has {actual} cards across hands; expected {expected}"
            ),
        }
    }
}

impl Error for GameInitError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::RuleConfig(error) => Some(error),
            Self::InvalidHand { source, .. } | Self::InvalidBottomCards(source) => Some(source),
            Self::UnsupportedProfile { .. }
            | Self::InvalidLandlord { .. }
            | Self::InvalidFirstBidder { .. }
            | Self::UnsupportedBiddingMode
            | Self::WrongHandSize { .. }
            | Self::WrongBottomCardCount { .. }
            | Self::BottomCardNotHeldByLandlord { .. }
            | Self::DeckCountMismatch { .. } => None,
        }
    }
}

/// Errors returned when decoding and restoring a serialized engine state.
#[derive(Debug)]
pub enum GameDeserializeError {
    /// The versioned byte envelope is malformed or unsupported.
    Codec(StateCodecError),
    /// The decoded state fails replay-based engine validation.
    Restore(GameRestoreError),
}

impl Display for GameDeserializeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Codec(error) => Display::fmt(error, formatter),
            Self::Restore(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for GameDeserializeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Codec(error) => Some(error),
            Self::Restore(error) => Some(error),
        }
    }
}

/// Errors returned while materializing a sampled hidden allocation.
#[derive(Debug)]
pub enum HiddenSampleError {
    /// Observer fell outside the three seats.
    InvalidObserver { observer: Seat },
    /// Hidden samples may only be built for the acting information set.
    ObserverNotCurrent {
        observer: Seat,
        current_player: Seat,
    },
    /// Hidden sampling is restricted to card play after landlord resolution.
    WrongPhase { phase: Phase },
    /// Container-A counts are not physically valid.
    InvalidAssignment(CardError),
    /// A sampled container has the wrong total number of cards.
    CapacityMismatch {
        seat: Seat,
        actual: u8,
        expected: u8,
    },
    /// Container A assigned more of a rank than exists in the hidden union.
    RankConservation {
        rank_id: u8,
        assigned_a: u8,
        unknown: u8,
    },
    /// The sampled state failed authoritative replay validation.
    Restore(GameRestoreError),
}

impl Display for HiddenSampleError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidObserver { observer } => {
                write!(
                    formatter,
                    "hidden-sample observer {observer} is outside 0..=2"
                )
            }
            Self::ObserverNotCurrent {
                observer,
                current_player,
            } => write!(
                formatter,
                "hidden-sample observer {observer} is not current player {current_player}"
            ),
            Self::WrongPhase { phase } => {
                write!(
                    formatter,
                    "hidden sampling requires card play, got {phase:?}"
                )
            }
            Self::InvalidAssignment(error) => {
                write!(formatter, "invalid container-A hidden assignment: {error}")
            }
            Self::CapacityMismatch {
                seat,
                actual,
                expected,
            } => write!(
                formatter,
                "hidden assignment for seat {seat} has {actual} cards, expected {expected}"
            ),
            Self::RankConservation {
                rank_id,
                assigned_a,
                unknown,
            } => write!(
                formatter,
                "hidden assignment gives A {assigned_a} of rank {rank_id}, but union has {unknown}"
            ),
            Self::Restore(error) => {
                write!(formatter, "sampled hidden state failed replay: {error}")
            }
        }
    }
}

impl Error for HiddenSampleError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidAssignment(error) => Some(error),
            Self::Restore(error) => Some(error),
            Self::InvalidObserver { .. }
            | Self::ObserverNotCurrent { .. }
            | Self::WrongPhase { .. }
            | Self::CapacityMismatch { .. }
            | Self::RankConservation { .. } => None,
        }
    }
}

/// Errors returned while proving a decoded state by replay.
#[derive(Debug)]
pub enum GameRestoreError {
    /// State and supplied rules refer to different stable configuration IDs.
    RuleConfigIdMismatch {
        /// ID encoded in the state.
        state: u32,
        /// ID carried by the supplied rules.
        rules: u32,
    },
    /// A post-bid state omitted the resolved landlord.
    MissingLandlord,
    /// Reconstructing a seat's initial card count overflowed.
    InitialCountOverflow {
        /// Seat containing the invalid count.
        seat: Seat,
        /// Rank containing the invalid count.
        rank_id: u8,
    },
    /// Resolved landlord history could not remove the three original bottom cards.
    BottomCardSubtraction {
        /// Resolved landlord seat.
        landlord: Seat,
        /// Rank containing fewer reconstructed cards than bottom-card metadata.
        rank_id: u8,
    },
    /// Reconstructed initial deal failed post-bid validation.
    InitialState(GameInitError),
    /// History length could not be represented by its `u32` event schema.
    HistoryTooLong,
    /// An event sequence number was discontinuous.
    HistorySequence {
        /// Required zero-based sequence.
        expected: u32,
        /// Encoded sequence.
        actual: u32,
    },
    /// An event actor did not match the replayed current player.
    HistoryActor {
        /// Event sequence containing the mismatch.
        sequence: u32,
        /// Replayed acting seat.
        expected: Seat,
        /// Encoded acting seat.
        actual: Seat,
    },
    /// A historical action was illegal during replay.
    Replay {
        /// Event sequence that failed.
        sequence: u32,
        /// Transition failure.
        source: GameError,
    },
    /// Replayed final state differed from one or more encoded fields.
    StateMismatch,
}

impl Display for GameRestoreError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RuleConfigIdMismatch { state, rules } => write!(
                formatter,
                "serialized state uses rule configuration {state}, but supplied rules use {rules}"
            ),
            Self::MissingLandlord => write!(formatter, "post-bid state has no landlord"),
            Self::InitialCountOverflow { seat, rank_id } => write!(
                formatter,
                "reconstructed initial count overflowed at seat {seat}, rank {rank_id}"
            ),
            Self::BottomCardSubtraction { landlord, rank_id } => write!(
                formatter,
                "reconstructed landlord seat {landlord} lacks bottom-card rank {rank_id}"
            ),
            Self::InitialState(error) => {
                write!(formatter, "invalid reconstructed initial state: {error}")
            }
            Self::HistoryTooLong => write!(formatter, "history length exceeds u32 schema"),
            Self::HistorySequence { expected, actual } => write!(
                formatter,
                "history sequence is {actual}; expected {expected}"
            ),
            Self::HistoryActor {
                sequence,
                expected,
                actual,
            } => write!(
                formatter,
                "history event {sequence} actor is seat {actual}; expected seat {expected}"
            ),
            Self::Replay { sequence, source } => {
                write!(formatter, "history event {sequence} is illegal: {source}")
            }
            Self::StateMismatch => {
                write!(
                    formatter,
                    "serialized state differs from its replayed history"
                )
            }
        }
    }
}

impl Error for GameRestoreError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InitialState(error) => Some(error),
            Self::Replay { source, .. } => Some(source),
            Self::RuleConfigIdMismatch { .. }
            | Self::MissingLandlord
            | Self::InitialCountOverflow { .. }
            | Self::BottomCardSubtraction { .. }
            | Self::HistoryTooLong
            | Self::HistorySequence { .. }
            | Self::HistoryActor { .. }
            | Self::StateMismatch => None,
        }
    }
}

/// Errors returned when an inverse delta cannot be applied safely.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UndoError {
    /// Token was created by another engine instance or cloned branch.
    InstanceMismatch,
    /// Token and state refer to different rule configurations.
    RuleConfigIdMismatch {
        /// Configuration ID captured in the token.
        token: u32,
        /// Configuration ID in the current state.
        state: u32,
    },
    /// Another successful transition occurred after this token was produced.
    RevisionMismatch {
        /// Revision immediately after the token's transition.
        expected: u64,
        /// Current engine revision.
        actual: u64,
    },
    /// Adding the token's single expected history event overflowed `usize`.
    HistoryLengthOverflow,
    /// Current history length does not match the token's transition suffix.
    HistoryLengthMismatch {
        /// Token history length plus one.
        expected: usize,
        /// Current history length.
        actual: usize,
    },
    /// The most recent event differs from the token's event.
    HistoryEventMismatch,
    /// Token actor falls outside the three state arrays.
    InvalidActor {
        /// Rejected actor seat.
        actor: Seat,
    },
}

impl Display for UndoError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InstanceMismatch => {
                write!(formatter, "undo token belongs to another game instance")
            }
            Self::RuleConfigIdMismatch { token, state } => write!(
                formatter,
                "undo token uses rule configuration {token}, but state uses {state}"
            ),
            Self::RevisionMismatch { expected, actual } => write!(
                formatter,
                "undo token expects revision {expected}, but engine is at {actual}"
            ),
            Self::HistoryLengthOverflow => {
                write!(formatter, "undo token history length overflows usize")
            }
            Self::HistoryLengthMismatch { expected, actual } => write!(
                formatter,
                "undo token expects history length {expected}, but state has {actual}"
            ),
            Self::HistoryEventMismatch => {
                write!(
                    formatter,
                    "undo token does not match the latest history event"
                )
            }
            Self::InvalidActor { actor } => {
                write!(formatter, "undo token actor {actor} is outside 0..=2")
            }
        }
    }
}

impl Error for UndoError {}

/// Errors produced while querying or stepping a post-bid game.
#[derive(Debug)]
pub enum GameError {
    /// A seat fell outside `0..=2`.
    InvalidSeat {
        /// Rejected seat.
        seat: Seat,
    },
    /// An operation requiring card play was attempted in another phase.
    WrongPhase {
        /// Actual phase.
        phase: Phase,
    },
    /// No actions may be applied after terminal transition.
    Terminal,
    /// The current legal set did not contain the action.
    IllegalAction {
        /// Acting seat.
        actor: Seat,
        /// Rejected action.
        action: GameAction,
    },
    /// Legal move generation failed for authoritative state data.
    MoveGeneration(GenerateMovesError),
    /// An internal invariant prevented a safe state transition.
    StateInvariant(&'static str),
}

impl Display for GameError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSeat { seat } => write!(formatter, "seat {seat} is outside 0..=2"),
            Self::WrongPhase { phase } => {
                write!(
                    formatter,
                    "card-play operation is invalid in phase {phase:?}"
                )
            }
            Self::Terminal => write!(formatter, "game is terminal"),
            Self::IllegalAction { actor, action } => {
                write!(formatter, "action {action:?} is illegal for seat {actor}")
            }
            Self::MoveGeneration(error) => Display::fmt(error, formatter),
            Self::StateInvariant(reason) => {
                write!(formatter, "game-state invariant failed: {reason}")
            }
        }
    }
}

impl Error for GameError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::MoveGeneration(error) => Some(error),
            Self::InvalidSeat { .. }
            | Self::WrongPhase { .. }
            | Self::Terminal
            | Self::IllegalAction { .. }
            | Self::StateInvariant(_) => None,
        }
    }
}
