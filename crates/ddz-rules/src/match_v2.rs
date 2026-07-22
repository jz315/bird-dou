//! Deterministic Huanle match and deal-attempt lifecycle coordination.
//!
//! This module deliberately owns only match boundaries: deterministic attempt
//! seeds, all-pass redeals, action-budget accounting, and replayable lifecycle
//! decisions. R004 owns reveal/dealing and R005 owns initial calling; robbing,
//! bottom handling, doubling, card play, and settlement remain authoritative
//! phase implementations in their respective tickets.

use std::error::Error;
use std::fmt::{Display, Formatter};

use ddz_core::{CardId, Move, Seat, CARD_COUNT};
use serde::{Deserialize, Serialize};

use crate::{
    derive_attempt_seed, shuffled_deck_for_seed, RuleConfigError, RuleConfigV2, RuleProfile,
    ATTEMPT_SEED_DERIVATION_ALGORITHM, PLAYER_COUNT, SHUFFLE_ALGORITHM,
};

const HUANLE_CARDS_PER_PLAYER: usize = 17;
const HUANLE_BOTTOM_CARD_COUNT: usize = 3;
const HUANLE_DEALT_CARD_COUNT: usize = HUANLE_CARDS_PER_PLAYER * PLAYER_COUNT;

/// Stable derivation used for the randomized pre-deal declaration order.
pub const PRE_DEAL_REVEAL_ORDER_ALGORITHM: &str = "deal_seed_permutation_v1";

/// Authoritative phase reached by the R003–R005 Huanle lifecycle.
///
/// R005 owns `Calling` and stops at the `Robbing`/`BottomReveal` boundaries.
/// Rob queues, bottom-card ownership, doubling, card play, and settlement are
/// introduced by their owning tickets instead of being guessed here.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseV2 {
    /// Every seat makes its deterministic-order pre-deal reveal declaration.
    PreDealReveal,
    /// One card per seat has been dealt for the current round and unrevealed
    /// seats may reveal or continue receiving cards.
    DealingReveal,
    /// All seventeen cards have been dealt and R005 owns the next transition.
    Calling,
    /// A first positive call selected a provisional candidate; R006 owns the
    /// rob queue and its final landlord resolution.
    Robbing,
    /// Calling all-pass with a first revealer resolved a landlord; R007 owns
    /// the bottom-card transfer and all later phase transitions.
    BottomReveal,
}

/// Irrevocable reveal information accumulated before the call phase.
///
/// A zero entry in `reveal_factor_by_seat` means that the seat has not
/// revealed. `maximum_factor` is one until the first reveal, so it is safe to
/// use as the neutral multiplicative factor in later settlement code.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RevealStateV2 {
    /// Whether each seat has irrevocably revealed its hand.
    pub revealed: [bool; PLAYER_COUNT],
    /// Reveal factor for each seat, or zero when the seat is still hidden.
    pub reveal_factor_by_seat: [u32; PLAYER_COUNT],
    /// First seat to reveal, determined by accepted action-event order.
    pub first_revealer: Option<Seat>,
    /// Within-attempt action sequence of the first reveal.
    pub first_reveal_sequence: Option<u64>,
    /// Maximum of all accepted reveal factors, or one before any reveal.
    pub maximum_factor: u32,
}

/// Initial call-landlord state for one fully dealt Huanle attempt.
///
/// It deliberately captures only R005 facts. The R006 rob state derives its
/// eligibility from `declined` and its initial candidate from `caller`.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CallStateV2 {
    /// Seat that received the first call opportunity.
    pub first_caller: Seat,
    /// Seat that must make the next call decision while in `Calling`.
    pub current_player: Seat,
    /// Whether each seat has made its sole initial calling decision.
    pub acted: [bool; PLAYER_COUNT],
    /// Whether each seat explicitly passed the initial call.
    pub declined: [bool; PLAYER_COUNT],
    /// First positive caller, if calling ended by `CallLandlord`.
    pub caller: Option<Seat>,
}

/// Safe, reveal-phase observation for one player.
///
/// This is intentionally narrower than the full `ObservationV2` planned for
/// R009. It provides exactly the visibility R004 needs: the observer's own
/// partial/full hand and opponents' hands only after those opponents reveal.
/// It never includes bottom cards or an unrevealed opponent hand.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RevealObservationV2 {
    /// Current authoritative pre-call phase.
    pub phase: PhaseV2,
    /// Seat for which this projection was constructed.
    pub observer: Seat,
    /// The observer's own currently dealt partial or full hand.
    pub own_hand: Vec<CardId>,
    /// Number of cards currently dealt to each seat.
    pub cards_received: [u8; PLAYER_COUNT],
    /// Current hand of each revealed seat; every hidden seat has an empty list.
    pub public_revealed_hands: [Vec<CardId>; PLAYER_COUNT],
    /// Public per-seat reveal flags.
    pub revealed: [bool; PLAYER_COUNT],
    /// First revealed seat, if any.
    pub first_revealer: Option<Seat>,
    /// Public maximum reveal factor rather than a product of reveal factors.
    pub maximum_reveal_factor: u32,
    /// Seats that may make a during-deal reveal decision in the current round.
    pub pending_during_deal_reveal: [bool; PLAYER_COUNT],
    /// The only seat that may make the next deterministic pre-deal declaration.
    pub pre_deal_reveal_actor: Option<Seat>,
    /// First caller resolved at the R004/R005 boundary once dealing is complete.
    pub first_caller: Option<Seat>,
    /// Bottom cards remain hidden until R007 resolves a landlord.
    pub bottom_visible: bool,
}

/// Lifecycle status for the current Huanle deal attempt.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptStatusV2 {
    /// No landlord has been resolved yet; subsequent phase tickets own the details.
    Unresolved,
    /// A phase state machine resolved the landlord and card play may eventually follow.
    LandlordResolved {
        /// Resolved landlord seat.
        landlord: Seat,
    },
}

/// A reveal decision at one of the three Huanle reveal opportunities.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RevealDecisionV2 {
    /// Reveal the currently permitted hand information.
    Reveal,
    /// Decline the current reveal opportunity.
    Decline,
}

/// One initial calling decision.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CallDecisionV2 {
    /// Call landlord and immediately enter the robbing phase.
    CallLandlord,
    /// Decline to call landlord.
    PassCall,
}

/// One robbing decision.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RobDecisionV2 {
    /// Replace the provisional landlord candidate.
    Rob,
    /// Decline this seat's one robbing opportunity.
    PassRob,
}

/// One per-seat doubling decision.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DoubleDecisionV2 {
    /// Apply this seat's configured pairwise double factor.
    Double,
    /// Decline to double.
    Decline,
}

/// Normalized player action across every Huanle phase.
///
/// R003 records this protocol for deterministic attempt audit and replay. Later
/// phase tickets own its legality and state transitions.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GameActionV2 {
    /// Reveal decision before cards are dealt.
    PreDealReveal(RevealDecisionV2),
    /// Reveal decision while receiving cards.
    DuringDealReveal(RevealDecisionV2),
    /// Initial call-landlord decision.
    Call(CallDecisionV2),
    /// Rob-landlord decision.
    Rob(RobDecisionV2),
    /// Reveal decision after the landlord receives the bottom cards.
    PostBottomReveal(RevealDecisionV2),
    /// Per-seat doubling decision.
    Double(DoubleDecisionV2),
    /// Canonical card-play action.
    Play(Move),
}

/// One accepted player action retained inside its deal attempt.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AttemptActionRecordV2 {
    /// Monotonic sequence within this attempt.
    pub sequence: u64,
    /// Seat whose action was accepted.
    pub actor: Seat,
    /// Exact normalized action accepted by its owning phase state machine.
    pub action: GameActionV2,
}

/// One deterministic deal attempt retained inside an active Huanle match.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DealAttemptStateV2 {
    /// Zero-based position inside the enclosing match.
    pub attempt_index: u32,
    /// Deterministic child seed derived from the match seed and attempt index.
    pub deal_seed: u64,
    /// Stable physical-card shuffle implementation identifier.
    pub shuffle_algorithm: String,
    /// Randomized fallback first-call seat retained for the future calling phase.
    pub first_caller_candidate: Seat,
    /// Authoritative physical-card order for later partial dealing and replay.
    pub deck: Vec<CardId>,
    /// Current authoritative phase before the R005 call state machine starts.
    pub phase: PhaseV2,
    /// Deterministic declaration order used by the pre-deal reveal phase.
    pub pre_deal_reveal_order: [Seat; PLAYER_COUNT],
    /// Stable identifier for the declaration-order derivation.
    pub pre_deal_reveal_order_algorithm: String,
    /// Number of declarations already accepted from `pre_deal_reveal_order`.
    pub pre_deal_reveal_cursor: u8,
    /// Number of cards dealt to each seat so far; R004 always advances them together.
    pub cards_received: [u8; PLAYER_COUNT],
    /// R004 reveal state, including the first revealer and maximum factor.
    pub reveal: RevealStateV2,
    /// Seats still entitled to choose reveal/continue for the current dealt round.
    pub pending_during_deal_reveal: [bool; PLAYER_COUNT],
    /// First calling seat, available only after all seventeen rounds complete.
    pub first_caller: Option<Seat>,
    /// Explicit initial calling state while R005 owns the phase and retained at
    /// the R006/R007 boundary for deterministic replay and audit.
    pub call: Option<CallStateV2>,
    /// Server-authoritative partial/full physical hands. This is deliberately
    /// private so network code must use `reveal_observation` rather than
    /// accidentally returning hidden opponent cards.
    hands: [Vec<CardId>; PLAYER_COUNT],
    /// Server-authoritative bottom cards. They remain absent from every R004
    /// observation and are owned by the R007 bottom-reveal implementation.
    bottom_cards: [CardId; HUANLE_BOTTOM_CARD_COUNT],
    /// Count of accepted player actions in this attempt across all implemented phases.
    pub accepted_action_count: u64,
    /// Complete accepted player-action history for this attempt.
    pub action_history: Vec<AttemptActionRecordV2>,
    /// Lifecycle result known so far for this attempt.
    pub status: AttemptStatusV2,
}

/// Why a completed attempt was replaced by another deterministic attempt.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptCompletionReasonV2 {
    /// Every call candidate passed while no first revealer existed.
    AllPass,
}

/// Immutable audit summary for an attempt that has been redealt.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AttemptSummaryV2 {
    /// Zero-based position inside the enclosing match.
    pub attempt_index: u32,
    /// Deterministic seed that reconstructs this attempt's deck.
    pub deal_seed: u64,
    /// Shuffle implementation used to create the deck.
    pub shuffle_algorithm: String,
    /// Stored fallback first-call seat for audit and replay.
    pub first_caller_candidate: Seat,
    /// Phase at which the completed attempt was closed.
    pub phase_at_completion: PhaseV2,
    /// Dealing progress retained for audit; all-pass attempts must be fully dealt.
    pub cards_received: [u8; PLAYER_COUNT],
    /// Reveal state retained for audit; an all-pass summary contains no reveal.
    pub reveal: RevealStateV2,
    /// First caller resolved before the all-pass disposition.
    pub first_caller: Option<Seat>,
    /// Complete call state at the all-pass disposition.
    pub call: Option<CallStateV2>,
    /// All accepted player actions attributed to this attempt.
    pub accepted_action_count: u64,
    /// Complete accepted player-action history for this closed attempt.
    pub action_history: Vec<AttemptActionRecordV2>,
    /// Reason the attempt was closed.
    pub completion_reason: AttemptCompletionReasonV2,
}

/// Match-level terminal metadata available before R008 adds pairwise settlement.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MatchCompletionV2 {
    /// Seat reported by the authoritative future card-play state machine as the winner.
    pub winner: Seat,
    /// Landlord resolved for the final attempt.
    pub landlord: Seat,
    /// Final active attempt index.
    pub final_attempt_index: u32,
    /// Total accepted player actions across every attempt in the match.
    pub total_accepted_action_count: u64,
}

/// Complete lifecycle state for one Huanle match.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MatchStateV2 {
    /// Stable seed from which every attempt seed is derived.
    pub match_seed: u64,
    /// Rule configuration identifier used to initialize this match.
    pub rule_config_id: u32,
    /// Stable hash of the complete v2 rule configuration.
    pub rules_hash: String,
    /// Explicit child-seed derivation identifier.
    pub attempt_seed_derivation_algorithm: String,
    /// Index of [`Self::current_attempt`].
    pub attempt_index: u32,
    /// Active attempt, including the authoritative physical deck.
    pub current_attempt: DealAttemptStateV2,
    /// Every all-pass attempt retained instead of being discarded on redeal.
    pub completed_attempts: Vec<AttemptSummaryV2>,
    /// Cumulative accepted player-action budget for all attempts.
    pub total_accepted_action_count: u64,
    /// Whether card play has reported a final winner for the match.
    pub terminal: bool,
    /// Terminal metadata without settlement payoffs; R008 enriches this contract.
    pub final_result: Option<MatchCompletionV2>,
}

/// Replayable decision events that mutate match lifecycle state.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchDecisionEventV2 {
    /// One phase accepted an exact player action for the active attempt.
    PlayerActionAccepted {
        /// Monotonic sequence within the decision event log.
        sequence: u64,
        /// Attempt that received the action.
        attempt_index: u32,
        /// Accepted action record, including the within-attempt sequence and actor.
        action: AttemptActionRecordV2,
    },
    /// The active attempt ended with all call candidates passing and starts a redeal.
    AllPass {
        /// Monotonic sequence within the decision event log.
        sequence: u64,
        /// Attempt that was closed.
        attempt_index: u32,
    },
    /// A later landlord-selection phase resolved the landlord for the active attempt.
    LandlordResolved {
        /// Monotonic sequence within the decision event log.
        sequence: u64,
        /// Attempt whose landlord was resolved.
        attempt_index: u32,
        /// Resolved landlord seat.
        landlord: Seat,
    },
    /// A later card-play phase reported a final winner for the active attempt.
    MatchCompleted {
        /// Monotonic sequence within the decision event log.
        sequence: u64,
        /// Attempt that produced the terminal result.
        attempt_index: u32,
        /// Seat reported as winner by authoritative card play.
        winner: Seat,
    },
}

impl MatchDecisionEventV2 {
    const fn sequence(&self) -> u64 {
        match self {
            Self::PlayerActionAccepted { sequence, .. }
            | Self::AllPass { sequence, .. }
            | Self::LandlordResolved { sequence, .. }
            | Self::MatchCompleted { sequence, .. } => *sequence,
        }
    }
}

/// Automatically generated lifecycle events kept separately from player decisions.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SystemEventV2 {
    /// A deterministic attempt became active.
    AttemptStarted {
        /// Attempt index that became active.
        attempt_index: u32,
        /// Deterministic seed used to shuffle its physical deck.
        deal_seed: u64,
        /// Fallback first-call candidate retained for later calling.
        first_caller_candidate: Seat,
    },
    /// Rust dealt one deterministic card to every seat for a reveal round.
    /// The event contains no card identities; those remain server state and
    /// are derived from the attempt seed during replay.
    DealingRoundDealt {
        /// Attempt receiving the round.
        attempt_index: u32,
        /// One-based number of cards now held by every seat.
        cards_received: u8,
    },
    /// Seventeen rounds completed and the first caller became public.
    CallingOpened {
        /// Attempt that reached the call boundary.
        attempt_index: u32,
        /// First caller selected from the first revealer or seeded fallback.
        first_caller: Seat,
    },
    /// The first positive call ended Calling and opened the R006 rob boundary.
    CallingEndedWithCall {
        /// Attempt whose call phase ended.
        attempt_index: u32,
        /// Provisional landlord candidate selected by the first call.
        caller: Seat,
    },
    /// Every seat passed Calling, so the first revealer directly became landlord.
    CallingAllPassAssignedLandlord {
        /// Attempt whose call phase exhausted all seats.
        attempt_index: u32,
        /// First revealer assigned as landlord by the frozen policy.
        landlord: Seat,
    },
    /// A no-reveal all-pass transitioned within the same match to another attempt.
    Redeal {
        /// Attempt that closed with all pass.
        from_attempt: u32,
        /// Attempt that became active after deterministic reseeding.
        to_attempt: u32,
    },
    /// A phase state machine resolved the landlord.
    LandlordResolved {
        /// Resolved landlord seat.
        landlord: Seat,
    },
    /// A future authoritative card-play phase reported a final winner.
    MatchCompleted {
        /// Seat reported as winner.
        winner: Seat,
    },
}

/// One ordered system event record.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SystemEventRecordV2 {
    /// Monotonic sequence within the system event log.
    pub sequence: u64,
    /// Generated lifecycle event.
    pub event: SystemEventV2,
}

/// Authoritative R003 match coordinator for the Huanle v2 profile.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct HuanleMatchV2 {
    /// Validated immutable rules used by the authoritative reveal transition.
    /// State/replay identity remains the public config ID and hash in
    /// `MatchStateV2`; this retained copy prevents any caller-owned config
    /// object from changing a live match after construction.
    rules: RuleConfigV2,
    state: MatchStateV2,
    decision_events: Vec<MatchDecisionEventV2>,
    system_events: Vec<SystemEventRecordV2>,
}

impl HuanleMatchV2 {
    /// Initialize a deterministic Huanle match at attempt zero.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError::RuleConfig`] unless `rules` is a valid v2 Huanle profile.
    pub fn new(match_seed: u64, rules: &RuleConfigV2) -> Result<Self, MatchError> {
        rules.validate().map_err(MatchError::RuleConfig)?;
        if rules.profile != RuleProfile::HuanleClassicV1 {
            return Err(MatchError::RuleConfig(
                RuleConfigError::IncompatibleProfile {
                    profile: rules.profile,
                    field: "profile",
                    expected: "huanle_classic_v1 for MatchStateV2",
                },
            ));
        }

        let current_attempt = new_attempt(match_seed, 0);
        let state = MatchStateV2 {
            match_seed,
            rule_config_id: rules.rule_config_id,
            rules_hash: rules.rules_hash().map_err(MatchError::RuleConfig)?,
            attempt_seed_derivation_algorithm: ATTEMPT_SEED_DERIVATION_ALGORITHM.to_owned(),
            attempt_index: 0,
            current_attempt,
            completed_attempts: Vec::new(),
            total_accepted_action_count: 0,
            terminal: false,
            final_result: None,
        };
        let mut game = Self {
            rules: *rules,
            state,
            decision_events: Vec::new(),
            system_events: Vec::new(),
        };
        game.push_attempt_started_system_event()?;
        game.validate()?;
        Ok(game)
    }

    /// Return immutable authoritative lifecycle state for replay and server-side audit.
    ///
    /// This contains the deterministic deck and must never be serialized as a
    /// player or web response. Use [`Self::reveal_observation`] for an
    /// information-safe R004 projection.
    #[must_use]
    pub const fn state(&self) -> &MatchStateV2 {
        &self.state
    }

    /// Return every replayable lifecycle decision in order.
    #[must_use]
    pub fn decision_events(&self) -> &[MatchDecisionEventV2] {
        &self.decision_events
    }

    /// Return every derived system event in order.
    #[must_use]
    pub fn system_events(&self) -> &[SystemEventRecordV2] {
        &self.system_events
    }

    /// Return the current pre-call phase.
    #[must_use]
    pub const fn phase(&self) -> PhaseV2 {
        self.state.current_attempt.phase
    }

    /// Build the information-safe R004 reveal projection for one observer.
    ///
    /// The result includes the observer's own partial/full hand and only hands
    /// made public by an irrevocable reveal. Bottom cards and every unrevealed
    /// opponent hand are omitted by construction.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError::InvalidSeat`] when `observer` is outside the
    /// three-seat Huanle table.
    pub fn reveal_observation(&self, observer: Seat) -> Result<RevealObservationV2, MatchError> {
        validate_seat(observer)?;
        let attempt = &self.state.current_attempt;
        let observer_index = usize::from(observer);
        let public_revealed_hands = std::array::from_fn(|seat| {
            if attempt.reveal.revealed[seat] {
                attempt.hands[seat].clone()
            } else {
                Vec::new()
            }
        });
        let pre_deal_reveal_actor = if attempt.phase == PhaseV2::PreDealReveal {
            attempt
                .pre_deal_reveal_order
                .get(usize::from(attempt.pre_deal_reveal_cursor))
                .copied()
        } else {
            None
        };

        Ok(RevealObservationV2 {
            phase: attempt.phase,
            observer,
            own_hand: attempt.hands[observer_index].clone(),
            cards_received: attempt.cards_received,
            public_revealed_hands,
            revealed: attempt.reveal.revealed,
            first_revealer: attempt.reveal.first_revealer,
            maximum_reveal_factor: attempt.reveal.maximum_factor,
            pending_during_deal_reveal: attempt.pending_during_deal_reveal,
            pre_deal_reveal_actor,
            first_caller: attempt.first_caller,
            bottom_visible: false,
        })
    }

    /// Enumerate reveal actions that are currently legal for one seat.
    ///
    /// Calling and later-phase actions intentionally remain absent: their
    /// authoritative state machines belong to later tickets.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid seat or a terminal match.
    pub fn legal_reveal_actions(&self, actor: Seat) -> Result<Vec<GameActionV2>, MatchError> {
        self.ensure_not_terminal()?;
        validate_seat(actor)?;
        let attempt = &self.state.current_attempt;
        let permitted = match attempt.phase {
            PhaseV2::PreDealReveal => attempt
                .pre_deal_reveal_order
                .get(usize::from(attempt.pre_deal_reveal_cursor))
                .is_some_and(|expected| *expected == actor),
            PhaseV2::DealingReveal => attempt.pending_during_deal_reveal[usize::from(actor)],
            PhaseV2::Calling | PhaseV2::Robbing | PhaseV2::BottomReveal => false,
        };
        if !permitted {
            return Ok(Vec::new());
        }
        let actions = match attempt.phase {
            PhaseV2::PreDealReveal => vec![
                GameActionV2::PreDealReveal(RevealDecisionV2::Reveal),
                GameActionV2::PreDealReveal(RevealDecisionV2::Decline),
            ],
            PhaseV2::DealingReveal => vec![
                GameActionV2::DuringDealReveal(RevealDecisionV2::Reveal),
                GameActionV2::DuringDealReveal(RevealDecisionV2::Decline),
            ],
            PhaseV2::Calling | PhaseV2::Robbing | PhaseV2::BottomReveal => Vec::new(),
        };
        Ok(actions)
    }

    /// Enumerate the two initial calling actions for the current caller.
    ///
    /// R005 deliberately exposes no rob action here: a successful first call
    /// opens the `Robbing` boundary for R006, while a full initial pass uses
    /// the explicit all-pass disposition in this ticket.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid seat, terminal match, or corrupted
    /// calling boundary state.
    pub fn legal_call_actions(&self, actor: Seat) -> Result<Vec<GameActionV2>, MatchError> {
        self.ensure_not_terminal()?;
        validate_seat(actor)?;
        if self.phase() != PhaseV2::Calling {
            return Ok(Vec::new());
        }
        let call = self
            .state
            .current_attempt
            .call
            .ok_or(MatchError::StateInvariant(
                "calling phase must retain an explicit call state",
            ))?;
        if actor != call.current_player {
            return Ok(Vec::new());
        }
        Ok(vec![
            GameActionV2::Call(CallDecisionV2::CallLandlord),
            GameActionV2::Call(CallDecisionV2::PassCall),
        ])
    }

    /// Apply the next deterministic-order pre-deal reveal declaration.
    ///
    /// Once all three declarations are accepted, Rust automatically deals the
    /// first round and opens the relevant during-deal reveal decisions.
    ///
    /// # Errors
    ///
    /// Returns an error for an out-of-order actor, incorrect phase, invalid
    /// seat, terminal match, or arithmetic/event-sequence overflow. Errors are
    /// transactional and leave the match unchanged.
    pub fn apply_pre_deal_reveal(
        &mut self,
        actor: Seat,
        decision: RevealDecisionV2,
    ) -> Result<(), MatchError> {
        let mut next = self.clone();
        next.apply_pre_deal_reveal_inner(actor, decision)?;
        next.validate()?;
        *self = next;
        Ok(())
    }

    /// Apply one currently pending during-deal reveal decision.
    ///
    /// Every unrevealed seat receives one decision after each authoritative
    /// dealing round. After the final pending decision, Rust either deals the
    /// next round or opens `Calling` once all seventeen cards are dealt.
    ///
    /// # Errors
    ///
    /// Returns an error for a non-pending seat, incorrect phase, invalid seat,
    /// terminal match, or arithmetic/event-sequence overflow. Errors are
    /// transactional and leave the match unchanged.
    pub fn apply_during_deal_reveal(
        &mut self,
        actor: Seat,
        decision: RevealDecisionV2,
    ) -> Result<(), MatchError> {
        let mut next = self.clone();
        next.apply_during_deal_reveal_inner(actor, decision)?;
        next.validate()?;
        *self = next;
        Ok(())
    }

    /// Apply one authoritative initial call-landlord decision.
    ///
    /// A first `CallLandlord` immediately ends `Calling`, records its caller,
    /// and opens `Robbing` for R006. A `PassCall` advances to the next seat;
    /// after all three pass, the first revealer becomes landlord or a hidden
    /// all-pass attempt redeals within the same match.
    ///
    /// # Errors
    ///
    /// Returns an error for an incorrect phase or actor, an already-acted
    /// caller, invalid seat, terminal match, or event-sequence overflow.
    /// Errors are transactional.
    pub fn apply_call(&mut self, actor: Seat, decision: CallDecisionV2) -> Result<(), MatchError> {
        let mut next = self.clone();
        next.apply_call_inner(actor, decision)?;
        next.validate()?;
        *self = next;
        Ok(())
    }

    /// Reject an attempt to bypass an owning phase state machine.
    ///
    /// This former R003 audit seam remains public only to give old callers a
    /// deterministic error. It never accepts an action: R004 and R005 now own
    /// reveal/call legality, and future tickets must likewise expose their own
    /// authoritative transitions instead of fabricating accepted actions.
    ///
    /// # Errors
    ///
    /// Always returns the error for the action's owning state machine after
    /// validating `actor`; it leaves the match unchanged.
    pub fn record_accepted_action(
        &mut self,
        actor: Seat,
        action: &GameActionV2,
    ) -> Result<(), MatchError> {
        let mut next = self.clone();
        next.record_accepted_action_inner(actor, action)?;
        next.validate()?;
        *self = next;
        Ok(())
    }

    /// Close a no-reveal all-pass attempt and automatically begin the next deterministic attempt.
    ///
    /// R005 invokes this only after its explicit `CallStateV2` records three
    /// `PassCall` decisions and confirms that no player revealed.
    ///
    /// # Errors
    ///
    /// Returns an error if no accepted action supports the disposition, a landlord was already
    /// resolved, the match is terminal, or an index/sequence would overflow.
    pub fn resolve_no_reveal_all_pass(&mut self) -> Result<(), MatchError> {
        let mut next = self.clone();
        next.resolve_no_reveal_all_pass_inner()?;
        next.validate()?;
        *self = next;
        Ok(())
    }

    fn resolve_no_reveal_all_pass_inner(&mut self) -> Result<(), MatchError> {
        self.ensure_not_terminal()?;
        self.ensure_unresolved_attempt()?;
        if self.state.current_attempt.accepted_action_count == 0 {
            return Err(MatchError::AllPassWithoutAcceptedActions);
        }
        if self
            .state
            .current_attempt
            .action_history
            .iter()
            .any(AttemptActionRecordV2::is_reveal)
        {
            return Err(MatchError::AllPassAfterReveal);
        }
        self.require_phase(PhaseV2::Calling)?;
        let call = self
            .state
            .current_attempt
            .call
            .ok_or(MatchError::StateInvariant(
                "calling phase must retain an explicit call state",
            ))?;
        if call.caller.is_some()
            || !call.acted.iter().all(|acted| *acted)
            || !call.declined.iter().all(|declined| *declined)
        {
            return Err(MatchError::NoRevealAllPassRequiresCompleteCallPass);
        }

        let from_attempt = self.state.attempt_index;
        let to_attempt = from_attempt
            .checked_add(1)
            .ok_or(MatchError::AttemptIndexOverflow)?;
        let summary = AttemptSummaryV2 {
            attempt_index: from_attempt,
            deal_seed: self.state.current_attempt.deal_seed,
            shuffle_algorithm: self.state.current_attempt.shuffle_algorithm.clone(),
            first_caller_candidate: self.state.current_attempt.first_caller_candidate,
            phase_at_completion: self.state.current_attempt.phase,
            cards_received: self.state.current_attempt.cards_received,
            reveal: self.state.current_attempt.reveal,
            first_caller: self.state.current_attempt.first_caller,
            call: self.state.current_attempt.call,
            accepted_action_count: self.state.current_attempt.accepted_action_count,
            action_history: self.state.current_attempt.action_history.clone(),
            completion_reason: AttemptCompletionReasonV2::AllPass,
        };
        let next_attempt = new_attempt(self.state.match_seed, to_attempt);
        let decision = MatchDecisionEventV2::AllPass {
            sequence: self.next_decision_sequence()?,
            attempt_index: from_attempt,
        };
        let redeal_sequence = self.next_system_sequence()?;
        let start_sequence = redeal_sequence
            .checked_add(1)
            .ok_or(MatchError::EventSequenceOverflow)?;

        self.state.completed_attempts.push(summary);
        self.state.attempt_index = to_attempt;
        self.state.current_attempt = next_attempt;
        self.decision_events.push(decision);
        self.system_events.push(SystemEventRecordV2 {
            sequence: redeal_sequence,
            event: SystemEventV2::Redeal {
                from_attempt,
                to_attempt,
            },
        });
        self.system_events.push(SystemEventRecordV2 {
            sequence: start_sequence,
            event: SystemEventV2::AttemptStarted {
                attempt_index: to_attempt,
                deal_seed: self.state.current_attempt.deal_seed,
                first_caller_candidate: self.state.current_attempt.first_caller_candidate,
            },
        });
        Ok(())
    }

    /// Record a landlord resolution produced by a future validated call/rob state machine.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid seat, absent accepted action evidence, an already resolved
    /// attempt, or a terminal match.
    pub fn record_landlord_resolution(&mut self, landlord: Seat) -> Result<(), MatchError> {
        let mut next = self.clone();
        next.record_landlord_resolution_inner(landlord)?;
        next.validate()?;
        *self = next;
        Ok(())
    }

    fn record_landlord_resolution_inner(&mut self, landlord: Seat) -> Result<(), MatchError> {
        self.ensure_not_terminal()?;
        self.ensure_unresolved_attempt()?;
        validate_seat(landlord)?;
        let call = self
            .state
            .current_attempt
            .call
            .ok_or(MatchError::StateInvariant(
                "landlord resolution requires the retained R005 call state",
            ))?;
        match self.phase() {
            PhaseV2::Calling => {
                if call.caller.is_some()
                    || !call.acted.iter().all(|acted| *acted)
                    || !call.declined.iter().all(|declined| *declined)
                    || self.state.current_attempt.reveal.first_revealer != Some(landlord)
                {
                    return Err(MatchError::LandlordResolutionRequiresCallOutcome);
                }
            }
            PhaseV2::Robbing => {
                if call.caller.is_none() {
                    return Err(MatchError::LandlordResolutionRequiresCallOutcome);
                }
            }
            actual => {
                return Err(MatchError::UnexpectedPhase {
                    expected: PhaseV2::Calling,
                    actual,
                });
            }
        }
        if self.state.current_attempt.accepted_action_count == 0 {
            return Err(MatchError::LandlordResolutionWithoutAcceptedActions);
        }

        let decision = MatchDecisionEventV2::LandlordResolved {
            sequence: self.next_decision_sequence()?,
            attempt_index: self.state.attempt_index,
            landlord,
        };
        let system = SystemEventRecordV2 {
            sequence: self.next_system_sequence()?,
            event: SystemEventV2::LandlordResolved { landlord },
        };
        self.state.current_attempt.status = AttemptStatusV2::LandlordResolved { landlord };
        self.decision_events.push(decision);
        self.system_events.push(system);
        Ok(())
    }

    /// Mark the match terminal after a future authoritative card-play state machine reports a winner.
    ///
    /// This lifecycle method intentionally records no payoff. R008 owns pairwise settlement and
    /// will enrich [`MatchCompletionV2`] rather than inventing a score in the match coordinator.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid winner, a missing landlord resolution, or a terminal match.
    pub fn complete_after_authoritative_card_play(
        &mut self,
        winner: Seat,
    ) -> Result<(), MatchError> {
        let mut next = self.clone();
        next.complete_after_authoritative_card_play_inner(winner)?;
        next.validate()?;
        *self = next;
        Ok(())
    }

    fn complete_after_authoritative_card_play_inner(
        &mut self,
        winner: Seat,
    ) -> Result<(), MatchError> {
        self.ensure_not_terminal()?;
        validate_seat(winner)?;
        let landlord = match self.state.current_attempt.status {
            AttemptStatusV2::Unresolved => return Err(MatchError::LandlordNotResolved),
            AttemptStatusV2::LandlordResolved { landlord } => landlord,
        };
        let decision = MatchDecisionEventV2::MatchCompleted {
            sequence: self.next_decision_sequence()?,
            attempt_index: self.state.attempt_index,
            winner,
        };
        let system = SystemEventRecordV2 {
            sequence: self.next_system_sequence()?,
            event: SystemEventV2::MatchCompleted { winner },
        };
        self.state.terminal = true;
        self.state.final_result = Some(MatchCompletionV2 {
            winner,
            landlord,
            final_attempt_index: self.state.attempt_index,
            total_accepted_action_count: self.state.total_accepted_action_count,
        });
        self.decision_events.push(decision);
        self.system_events.push(system);
        Ok(())
    }

    /// Reconstruct a match from its deterministic seed, v2 rules, and decision history.
    ///
    /// # Errors
    ///
    /// Returns a configuration error, an invalid transition error, or a replay mismatch when an
    /// event cannot reproduce the exact lifecycle state.
    pub fn replay(
        match_seed: u64,
        rules: &RuleConfigV2,
        decision_events: &[MatchDecisionEventV2],
    ) -> Result<Self, MatchError> {
        let mut replay = Self::new(match_seed, rules)?;
        let mut cursor = 0;
        while cursor < decision_events.len() {
            let expected = &decision_events[cursor];
            let before = replay.decision_events.len();
            match expected.clone() {
                MatchDecisionEventV2::PlayerActionAccepted {
                    attempt_index,
                    action,
                    ..
                } => {
                    replay.require_current_attempt(attempt_index)?;
                    let actor = action.actor;
                    match action.action {
                        GameActionV2::PreDealReveal(decision) => {
                            replay.apply_pre_deal_reveal(actor, decision)?;
                        }
                        GameActionV2::DuringDealReveal(decision) => {
                            replay.apply_during_deal_reveal(actor, decision)?;
                        }
                        GameActionV2::Call(decision) => replay.apply_call(actor, decision)?,
                        action => replay.record_accepted_action(actor, &action)?,
                    }
                }
                MatchDecisionEventV2::AllPass { attempt_index, .. } => {
                    replay.require_current_attempt(attempt_index)?;
                    replay.resolve_no_reveal_all_pass()?;
                }
                MatchDecisionEventV2::LandlordResolved {
                    attempt_index,
                    landlord,
                    ..
                } => {
                    replay.require_current_attempt(attempt_index)?;
                    replay.record_landlord_resolution(landlord)?;
                }
                MatchDecisionEventV2::MatchCompleted {
                    attempt_index,
                    winner,
                    ..
                } => {
                    replay.require_current_attempt(attempt_index)?;
                    replay.complete_after_authoritative_card_play(winner)?;
                }
            }
            let generated =
                replay
                    .decision_events
                    .get(before..)
                    .ok_or(MatchError::StateInvariant(
                        "replay decision-log boundary must remain valid",
                    ))?;
            let next_cursor = cursor
                .checked_add(generated.len())
                .ok_or(MatchError::EventSequenceOverflow)?;
            if generated.is_empty() || next_cursor > decision_events.len() {
                return Err(MatchError::ReplayEventMismatch {
                    sequence: expected.sequence(),
                });
            }
            for (actual, expected) in generated.iter().zip(&decision_events[cursor..next_cursor]) {
                if actual != expected {
                    return Err(MatchError::ReplayEventMismatch {
                        sequence: expected.sequence(),
                    });
                }
            }
            cursor = next_cursor;
        }
        replay.validate()?;
        Ok(replay)
    }

    /// Validate all internal lifecycle accounting invariants.
    ///
    /// # Errors
    ///
    /// Returns a descriptive invariant error if externally deserialized or otherwise corrupted
    /// state cannot represent a deterministic attempt lifecycle.
    pub fn validate(&self) -> Result<(), MatchError> {
        self.validate_state()?;
        validate_sequences(&self.decision_events, MatchDecisionEventV2::sequence)?;
        validate_sequences(&self.system_events, |record| record.sequence)?;
        Ok(())
    }

    fn validate_state(&self) -> Result<(), MatchError> {
        let state = &self.state;
        if state.rules_hash.is_empty() {
            return Err(MatchError::StateInvariant("rules_hash must be non-empty"));
        }
        if state.attempt_seed_derivation_algorithm != ATTEMPT_SEED_DERIVATION_ALGORITHM {
            return Err(MatchError::StateInvariant(
                "attempt seed derivation algorithm must be pinned",
            ));
        }
        if state.current_attempt.attempt_index != state.attempt_index {
            return Err(MatchError::StateInvariant(
                "current attempt index must equal match attempt index",
            ));
        }
        let completed_count = u32::try_from(state.completed_attempts.len())
            .map_err(|_| MatchError::StateInvariant("too many completed attempts"))?;
        if completed_count != state.attempt_index {
            return Err(MatchError::StateInvariant(
                "each preceding attempt must have exactly one all-pass summary",
            ));
        }

        let mut total_actions = state.current_attempt.accepted_action_count;
        validate_attempt(state.match_seed, &state.current_attempt, &self.rules)?;
        for (position, summary) in state.completed_attempts.iter().enumerate() {
            let expected_index = u32::try_from(position)
                .map_err(|_| MatchError::StateInvariant("attempt position exceeds u32"))?;
            if summary.attempt_index != expected_index {
                return Err(MatchError::StateInvariant(
                    "completed attempt summaries must be contiguous",
                ));
            }
            validate_summary(state.match_seed, summary, &self.rules)?;
            total_actions = total_actions
                .checked_add(summary.accepted_action_count)
                .ok_or(MatchError::ActionCountOverflow)?;
        }
        if total_actions != state.total_accepted_action_count {
            return Err(MatchError::StateInvariant(
                "total action count must equal current plus completed attempts",
            ));
        }
        if state.terminal != state.final_result.is_some() {
            return Err(MatchError::StateInvariant(
                "terminal and final_result must agree",
            ));
        }
        if let Some(final_result) = state.final_result {
            validate_seat(final_result.winner)?;
            validate_seat(final_result.landlord)?;
            if final_result.final_attempt_index != state.attempt_index
                || final_result.total_accepted_action_count != state.total_accepted_action_count
            {
                return Err(MatchError::StateInvariant(
                    "final result must describe the active attempt and total action count",
                ));
            }
            if state.current_attempt.status
                != (AttemptStatusV2::LandlordResolved {
                    landlord: final_result.landlord,
                })
            {
                return Err(MatchError::StateInvariant(
                    "terminal match must retain its resolved landlord",
                ));
            }
        }
        Ok(())
    }

    fn ensure_not_terminal(&self) -> Result<(), MatchError> {
        if self.state.terminal {
            Err(MatchError::TerminalMatch)
        } else {
            Ok(())
        }
    }

    fn ensure_unresolved_attempt(&self) -> Result<(), MatchError> {
        match self.state.current_attempt.status {
            AttemptStatusV2::Unresolved => Ok(()),
            AttemptStatusV2::LandlordResolved { landlord } => {
                Err(MatchError::AttemptAlreadyHasLandlord { landlord })
            }
        }
    }

    fn require_current_attempt(&self, attempt_index: u32) -> Result<(), MatchError> {
        if self.state.attempt_index == attempt_index {
            Ok(())
        } else {
            Err(MatchError::ReplayAttemptMismatch {
                expected: self.state.attempt_index,
                actual: attempt_index,
            })
        }
    }

    fn apply_pre_deal_reveal_inner(
        &mut self,
        actor: Seat,
        decision: RevealDecisionV2,
    ) -> Result<(), MatchError> {
        self.ensure_not_terminal()?;
        validate_seat(actor)?;
        self.require_phase(PhaseV2::PreDealReveal)?;

        let cursor = usize::from(self.state.current_attempt.pre_deal_reveal_cursor);
        let expected = self
            .state
            .current_attempt
            .pre_deal_reveal_order
            .get(cursor)
            .copied()
            .ok_or(MatchError::StateInvariant(
                "pre-deal reveal cursor must point to a declaration seat",
            ))?;
        if actor != expected {
            return Err(MatchError::PreDealRevealOutOfTurn {
                expected,
                actual: actor,
            });
        }

        let action = GameActionV2::PreDealReveal(decision);
        let action_record = self.append_accepted_action(actor, action)?;
        if decision == RevealDecisionV2::Reveal {
            self.accept_reveal(
                actor,
                self.rules.reveal.before_deal_factor,
                action_record.sequence,
            )?;
        }

        self.state.current_attempt.pre_deal_reveal_cursor = self
            .state
            .current_attempt
            .pre_deal_reveal_cursor
            .checked_add(1)
            .ok_or(MatchError::EventSequenceOverflow)?;
        if usize::from(self.state.current_attempt.pre_deal_reveal_cursor) == PLAYER_COUNT {
            self.state.current_attempt.phase = PhaseV2::DealingReveal;
            self.deal_until_decision_or_calling()?;
        }
        Ok(())
    }

    fn apply_during_deal_reveal_inner(
        &mut self,
        actor: Seat,
        decision: RevealDecisionV2,
    ) -> Result<(), MatchError> {
        self.ensure_not_terminal()?;
        validate_seat(actor)?;
        self.require_phase(PhaseV2::DealingReveal)?;
        let actor_index = usize::from(actor);
        if !self.state.current_attempt.pending_during_deal_reveal[actor_index] {
            return Err(MatchError::DuringDealRevealNotPending { seat: actor });
        }
        let factor_index = usize::from(self.state.current_attempt.cards_received[actor_index]);
        let factor = *self
            .rules
            .reveal
            .factor_by_cards_received
            .get(factor_index)
            .ok_or(MatchError::StateInvariant(
                "during-deal reveal factor index must be within the explicit schedule",
            ))?;

        let action = GameActionV2::DuringDealReveal(decision);
        let action_record = self.append_accepted_action(actor, action)?;
        if decision == RevealDecisionV2::Reveal {
            self.accept_reveal(actor, factor, action_record.sequence)?;
        }
        self.state.current_attempt.pending_during_deal_reveal[actor_index] = false;

        if self
            .state
            .current_attempt
            .pending_during_deal_reveal
            .iter()
            .any(|pending| *pending)
        {
            return Ok(());
        }
        if self.state.current_attempt.cards_received[0]
            == u8::try_from(HUANLE_CARDS_PER_PLAYER).expect("Huanle card count fits in u8")
        {
            self.open_calling()?;
        } else {
            self.deal_until_decision_or_calling()?;
        }
        Ok(())
    }

    fn apply_call_inner(
        &mut self,
        actor: Seat,
        decision: CallDecisionV2,
    ) -> Result<(), MatchError> {
        self.ensure_not_terminal()?;
        validate_seat(actor)?;
        self.require_phase(PhaseV2::Calling)?;
        let call = self
            .state
            .current_attempt
            .call
            .ok_or(MatchError::StateInvariant(
                "calling phase must retain an explicit call state",
            ))?;
        if actor != call.current_player {
            return Err(MatchError::CallOutOfTurn {
                expected: call.current_player,
                actual: actor,
            });
        }
        if call.acted[usize::from(actor)] {
            return Err(MatchError::CallAlreadyActed { seat: actor });
        }

        self.append_accepted_action(actor, GameActionV2::Call(decision))?;
        match decision {
            CallDecisionV2::CallLandlord => {
                let call =
                    self.state
                        .current_attempt
                        .call
                        .as_mut()
                        .ok_or(MatchError::StateInvariant(
                            "calling phase must retain an explicit call state",
                        ))?;
                call.acted[usize::from(actor)] = true;
                call.caller = Some(actor);
                self.state.current_attempt.phase = PhaseV2::Robbing;
                self.push_system_event(SystemEventV2::CallingEndedWithCall {
                    attempt_index: self.state.attempt_index,
                    caller: actor,
                })?;
            }
            CallDecisionV2::PassCall => {
                let all_acted = {
                    let call = self.state.current_attempt.call.as_mut().ok_or(
                        MatchError::StateInvariant(
                            "calling phase must retain an explicit call state",
                        ),
                    )?;
                    let actor_index = usize::from(actor);
                    call.acted[actor_index] = true;
                    call.declined[actor_index] = true;
                    call.acted.iter().all(|acted| *acted)
                };
                if !all_acted {
                    let next_player = next_unacted_call_seat(
                        actor,
                        self.state
                            .current_attempt
                            .call
                            .as_ref()
                            .ok_or(MatchError::StateInvariant(
                                "calling phase must retain an explicit call state",
                            ))?
                            .acted,
                    )?;
                    self.state
                        .current_attempt
                        .call
                        .as_mut()
                        .ok_or(MatchError::StateInvariant(
                            "calling phase must retain an explicit call state",
                        ))?
                        .current_player = next_player;
                    return Ok(());
                }

                if let Some(landlord) = self.state.current_attempt.reveal.first_revealer {
                    self.record_landlord_resolution_inner(landlord)?;
                    self.state.current_attempt.phase = PhaseV2::BottomReveal;
                    self.push_system_event(SystemEventV2::CallingAllPassAssignedLandlord {
                        attempt_index: self.state.attempt_index,
                        landlord,
                    })?;
                } else {
                    self.resolve_no_reveal_all_pass_inner()?;
                }
            }
        }
        Ok(())
    }

    fn record_accepted_action_inner(
        &mut self,
        actor: Seat,
        action: &GameActionV2,
    ) -> Result<(), MatchError> {
        self.ensure_not_terminal()?;
        validate_seat(actor)?;
        match action {
            GameActionV2::PreDealReveal(_) | GameActionV2::DuringDealReveal(_) => {
                Err(MatchError::RevealActionRequiresStateMachine)
            }
            GameActionV2::Call(_) => Err(MatchError::CallActionRequiresStateMachine),
            GameActionV2::Rob(_) => {
                Err(MatchError::ActionRequiresOwningStateMachine { phase: "robbing" })
            }
            GameActionV2::PostBottomReveal(_) => {
                Err(MatchError::ActionRequiresOwningStateMachine {
                    phase: "post_bottom_reveal",
                })
            }
            GameActionV2::Double(_) => {
                Err(MatchError::ActionRequiresOwningStateMachine { phase: "doubling" })
            }
            GameActionV2::Play(_) => {
                Err(MatchError::ActionRequiresOwningStateMachine { phase: "card_play" })
            }
        }
    }

    fn append_accepted_action(
        &mut self,
        actor: Seat,
        action: GameActionV2,
    ) -> Result<AttemptActionRecordV2, MatchError> {
        let attempt_count = self
            .state
            .current_attempt
            .accepted_action_count
            .checked_add(1)
            .ok_or(MatchError::ActionCountOverflow)?;
        let total_count = self
            .state
            .total_accepted_action_count
            .checked_add(1)
            .ok_or(MatchError::ActionCountOverflow)?;
        let action_record = AttemptActionRecordV2 {
            sequence: u64::try_from(self.state.current_attempt.action_history.len())
                .map_err(|_| MatchError::EventSequenceOverflow)?,
            actor,
            action,
        };
        let event = MatchDecisionEventV2::PlayerActionAccepted {
            sequence: self.next_decision_sequence()?,
            attempt_index: self.state.attempt_index,
            action: action_record.clone(),
        };

        self.state.current_attempt.accepted_action_count = attempt_count;
        self.state.total_accepted_action_count = total_count;
        self.state
            .current_attempt
            .action_history
            .push(action_record.clone());
        self.decision_events.push(event);
        Ok(action_record)
    }

    fn accept_reveal(
        &mut self,
        actor: Seat,
        factor: u32,
        action_sequence: u64,
    ) -> Result<(), MatchError> {
        let actor_index = usize::from(actor);
        let reveal = &mut self.state.current_attempt.reveal;
        if reveal.revealed[actor_index] {
            return Err(MatchError::RevealAlreadyIrrevocable { seat: actor });
        }
        if factor <= 1 {
            return Err(MatchError::StateInvariant(
                "accepted reveal factor must be a configured multiplier above one",
            ));
        }
        reveal.revealed[actor_index] = true;
        reveal.reveal_factor_by_seat[actor_index] = factor;
        reveal.maximum_factor = reveal.maximum_factor.max(factor);
        if reveal.first_revealer.is_none() {
            reveal.first_revealer = Some(actor);
            reveal.first_reveal_sequence = Some(action_sequence);
        }
        Ok(())
    }

    fn deal_until_decision_or_calling(&mut self) -> Result<(), MatchError> {
        loop {
            self.deal_one_round()?;
            if self
                .state
                .current_attempt
                .pending_during_deal_reveal
                .iter()
                .any(|pending| *pending)
            {
                return Ok(());
            }
            if self.state.current_attempt.cards_received[0]
                == u8::try_from(HUANLE_CARDS_PER_PLAYER).expect("Huanle card count fits in u8")
            {
                return self.open_calling();
            }
        }
    }

    fn deal_one_round(&mut self) -> Result<(), MatchError> {
        self.require_phase(PhaseV2::DealingReveal)?;
        let current_cards_received = self.state.current_attempt.cards_received;
        if !current_cards_received
            .iter()
            .all(|count| *count == current_cards_received[0])
        {
            return Err(MatchError::StateInvariant(
                "R004 deals one card to every seat in lockstep",
            ));
        }
        let current = usize::from(current_cards_received[0]);
        if current >= HUANLE_CARDS_PER_PLAYER {
            return Err(MatchError::StateInvariant(
                "cannot deal more than seventeen Huanle cards to one seat",
            ));
        }
        let next = u8::try_from(current + 1).expect("Huanle card count fits in u8");
        let start = current
            .checked_mul(PLAYER_COUNT)
            .ok_or(MatchError::StateInvariant(
                "dealing position must fit the authoritative deck",
            ))?;
        let cards: [CardId; PLAYER_COUNT] =
            std::array::from_fn(|seat| self.state.current_attempt.deck[start + seat]);
        for (seat, card) in cards.into_iter().enumerate() {
            self.state.current_attempt.hands[seat].push(card);
            self.state.current_attempt.cards_received[seat] = next;
            self.state.current_attempt.pending_during_deal_reveal[seat] =
                !self.state.current_attempt.reveal.revealed[seat];
        }
        self.push_system_event(SystemEventV2::DealingRoundDealt {
            attempt_index: self.state.attempt_index,
            cards_received: next,
        })
    }

    fn open_calling(&mut self) -> Result<(), MatchError> {
        if self.state.current_attempt.cards_received
            != [u8::try_from(HUANLE_CARDS_PER_PLAYER).expect("Huanle card count fits in u8");
                PLAYER_COUNT]
        {
            return Err(MatchError::StateInvariant(
                "calling opens only after every seat has seventeen cards",
            ));
        }
        if self
            .state
            .current_attempt
            .pending_during_deal_reveal
            .iter()
            .any(|pending| *pending)
        {
            return Err(MatchError::StateInvariant(
                "calling cannot open while a reveal decision remains pending",
            ));
        }
        let first_caller = self
            .state
            .current_attempt
            .reveal
            .first_revealer
            .unwrap_or(self.state.current_attempt.first_caller_candidate);
        self.state.current_attempt.phase = PhaseV2::Calling;
        self.state.current_attempt.first_caller = Some(first_caller);
        self.state.current_attempt.call = Some(new_call_state(first_caller));
        self.push_system_event(SystemEventV2::CallingOpened {
            attempt_index: self.state.attempt_index,
            first_caller,
        })
    }

    fn require_phase(&self, expected: PhaseV2) -> Result<(), MatchError> {
        let actual = self.state.current_attempt.phase;
        if actual == expected {
            Ok(())
        } else {
            Err(MatchError::UnexpectedPhase { expected, actual })
        }
    }

    fn next_decision_sequence(&self) -> Result<u64, MatchError> {
        u64::try_from(self.decision_events.len()).map_err(|_| MatchError::EventSequenceOverflow)
    }

    fn next_system_sequence(&self) -> Result<u64, MatchError> {
        u64::try_from(self.system_events.len()).map_err(|_| MatchError::EventSequenceOverflow)
    }

    fn push_system_event(&mut self, event: SystemEventV2) -> Result<(), MatchError> {
        let sequence = self.next_system_sequence()?;
        self.system_events
            .push(SystemEventRecordV2 { sequence, event });
        Ok(())
    }

    fn push_attempt_started_system_event(&mut self) -> Result<(), MatchError> {
        let attempt = &self.state.current_attempt;
        self.push_system_event(SystemEventV2::AttemptStarted {
            attempt_index: attempt.attempt_index,
            deal_seed: attempt.deal_seed,
            first_caller_candidate: attempt.first_caller_candidate,
        })
    }
}

fn new_attempt(match_seed: u64, attempt_index: u32) -> DealAttemptStateV2 {
    let deal_seed = derive_attempt_seed(match_seed, attempt_index);
    let deck = shuffled_deck_for_seed(deal_seed);
    DealAttemptStateV2 {
        attempt_index,
        deal_seed,
        shuffle_algorithm: SHUFFLE_ALGORITHM.to_owned(),
        first_caller_candidate: u8::try_from(deal_seed % (PLAYER_COUNT as u64))
            .expect("seat candidate is in 0..3"),
        deck: deck.to_vec(),
        phase: PhaseV2::PreDealReveal,
        pre_deal_reveal_order: pre_deal_reveal_order_for_seed(deal_seed),
        pre_deal_reveal_order_algorithm: PRE_DEAL_REVEAL_ORDER_ALGORITHM.to_owned(),
        pre_deal_reveal_cursor: 0,
        cards_received: [0; PLAYER_COUNT],
        reveal: RevealStateV2 {
            revealed: [false; PLAYER_COUNT],
            reveal_factor_by_seat: [0; PLAYER_COUNT],
            first_revealer: None,
            first_reveal_sequence: None,
            maximum_factor: 1,
        },
        pending_during_deal_reveal: [false; PLAYER_COUNT],
        first_caller: None,
        call: None,
        hands: std::array::from_fn(|_| Vec::with_capacity(HUANLE_CARDS_PER_PLAYER)),
        bottom_cards: [
            deck[HUANLE_DEALT_CARD_COUNT],
            deck[HUANLE_DEALT_CARD_COUNT + 1],
            deck[HUANLE_DEALT_CARD_COUNT + 2],
        ],
        accepted_action_count: 0,
        action_history: Vec::new(),
        status: AttemptStatusV2::Unresolved,
    }
}

fn pre_deal_reveal_order_for_seed(deal_seed: u64) -> [Seat; PLAYER_COUNT] {
    // A complete six-permutation table preserves a randomized declaration
    // order without deriving it from hidden physical card identities.
    const ORDERS: [[Seat; PLAYER_COUNT]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let position = usize::try_from(deal_seed % (ORDERS.len() as u64))
        .expect("pre-deal order index is below the permutation table length");
    ORDERS[position]
}

const fn new_call_state(first_caller: Seat) -> CallStateV2 {
    CallStateV2 {
        first_caller,
        current_player: first_caller,
        acted: [false; PLAYER_COUNT],
        declined: [false; PLAYER_COUNT],
        caller: None,
    }
}

fn next_unacted_call_seat(actor: Seat, acted: [bool; PLAYER_COUNT]) -> Result<Seat, MatchError> {
    for offset in 1..=PLAYER_COUNT {
        let candidate = (usize::from(actor) + offset) % PLAYER_COUNT;
        if !acted[candidate] {
            return u8::try_from(candidate)
                .map_err(|_| MatchError::StateInvariant("Huanle seat index must fit in u8"));
        }
    }
    Err(MatchError::StateInvariant(
        "calling must have an unacted seat before selecting the next caller",
    ))
}

fn validate_attempt(
    match_seed: u64,
    attempt: &DealAttemptStateV2,
    rules: &RuleConfigV2,
) -> Result<(), MatchError> {
    validate_seat(attempt.first_caller_candidate)?;
    if attempt.shuffle_algorithm != SHUFFLE_ALGORITHM {
        return Err(MatchError::StateInvariant(
            "attempt shuffle algorithm must be pinned",
        ));
    }
    let expected_seed = derive_attempt_seed(match_seed, attempt.attempt_index);
    if attempt.deal_seed != expected_seed {
        return Err(MatchError::StateInvariant(
            "attempt seed must derive from match seed and index",
        ));
    }
    if attempt.first_caller_candidate
        != u8::try_from(attempt.deal_seed % (PLAYER_COUNT as u64))
            .expect("seat candidate is in 0..3")
    {
        return Err(MatchError::StateInvariant(
            "first caller candidate must derive from deal seed",
        ));
    }
    if attempt.deck.len() != CARD_COUNT {
        return Err(MatchError::StateInvariant(
            "attempt deck must contain all fifty-four physical cards",
        ));
    }
    let expected_deck = shuffled_deck_for_seed(attempt.deal_seed);
    if attempt.deck != expected_deck.to_vec() {
        return Err(MatchError::StateInvariant(
            "attempt deck must match the deterministic deal seed",
        ));
    }
    let expected_bottom = [
        expected_deck[HUANLE_DEALT_CARD_COUNT],
        expected_deck[HUANLE_DEALT_CARD_COUNT + 1],
        expected_deck[HUANLE_DEALT_CARD_COUNT + 2],
    ];
    if attempt.bottom_cards != expected_bottom {
        return Err(MatchError::StateInvariant(
            "attempt bottom cards must remain the undisclosed final deck partition",
        ));
    }
    let received = attempt.cards_received;
    if !received.iter().all(|count| *count == received[0])
        || usize::from(received[0]) > HUANLE_CARDS_PER_PLAYER
    {
        return Err(MatchError::StateInvariant(
            "R004 cards received must be an equal 0..17 count for every seat",
        ));
    }
    for seat in 0..PLAYER_COUNT {
        let expected_hand = (0..usize::from(received[seat]))
            .map(|round| expected_deck[round * PLAYER_COUNT + seat])
            .collect::<Vec<_>>();
        if attempt.hands[seat] != expected_hand {
            return Err(MatchError::StateInvariant(
                "partial hands must be the deterministic round-robin deck partition",
            ));
        }
    }
    if attempt.pre_deal_reveal_order != pre_deal_reveal_order_for_seed(attempt.deal_seed) {
        return Err(MatchError::StateInvariant(
            "pre-deal reveal order must derive from the deterministic deal seed",
        ));
    }
    if attempt.pre_deal_reveal_order_algorithm != PRE_DEAL_REVEAL_ORDER_ALGORITHM {
        return Err(MatchError::StateInvariant(
            "pre-deal reveal order algorithm must be pinned",
        ));
    }
    if let AttemptStatusV2::LandlordResolved { landlord } = attempt.status {
        validate_seat(landlord)?;
    }
    validate_action_history(&attempt.action_history, attempt.accepted_action_count)?;
    let progress = replay_reveal_progress(
        &attempt.action_history,
        attempt.pre_deal_reveal_order,
        attempt.first_caller_candidate,
        rules,
    )?;
    if progress.phase != attempt.phase
        || progress.pre_deal_reveal_cursor != attempt.pre_deal_reveal_cursor
        || progress.cards_received != attempt.cards_received
        || progress.reveal != attempt.reveal
        || progress.pending_during_deal_reveal != attempt.pending_during_deal_reveal
        || progress.first_caller != attempt.first_caller
        || progress.call != attempt.call
    {
        return Err(MatchError::StateInvariant(
            "attempt reveal/dealing state must replay exactly from accepted actions",
        ));
    }
    Ok(())
}

fn validate_summary(
    match_seed: u64,
    summary: &AttemptSummaryV2,
    rules: &RuleConfigV2,
) -> Result<(), MatchError> {
    validate_seat(summary.first_caller_candidate)?;
    if summary.shuffle_algorithm != SHUFFLE_ALGORITHM {
        return Err(MatchError::StateInvariant(
            "completed attempt shuffle algorithm must be pinned",
        ));
    }
    if summary.deal_seed != derive_attempt_seed(match_seed, summary.attempt_index) {
        return Err(MatchError::StateInvariant(
            "completed attempt seed must derive from match seed and index",
        ));
    }
    if summary.first_caller_candidate
        != u8::try_from(summary.deal_seed % (PLAYER_COUNT as u64))
            .expect("seat candidate is in 0..3")
    {
        return Err(MatchError::StateInvariant(
            "completed first caller candidate must derive from deal seed",
        ));
    }
    validate_action_history(&summary.action_history, summary.accepted_action_count)?;
    let progress = replay_reveal_progress(
        &summary.action_history,
        pre_deal_reveal_order_for_seed(summary.deal_seed),
        summary.first_caller_candidate,
        rules,
    )?;
    if progress.phase != summary.phase_at_completion
        || progress.cards_received != summary.cards_received
        || progress.reveal != summary.reveal
        || progress.first_caller != summary.first_caller
        || progress.call != summary.call
    {
        return Err(MatchError::StateInvariant(
            "completed attempt reveal/dealing summary must replay exactly",
        ));
    }
    if summary.completion_reason == AttemptCompletionReasonV2::AllPass
        && (summary.phase_at_completion != PhaseV2::Calling
            || summary.reveal.first_revealer.is_some()
            || summary.reveal.revealed.iter().any(|revealed| *revealed)
            || !summary.call.is_some_and(|call| {
                call.caller.is_none()
                    && call.acted.iter().all(|acted| *acted)
                    && call.declined.iter().all(|declined| *declined)
            }))
    {
        return Err(MatchError::StateInvariant(
            "no-reveal all-pass summary must close only a fully dealt hidden attempt",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RevealProgressV2 {
    phase: PhaseV2,
    pre_deal_reveal_cursor: u8,
    cards_received: [u8; PLAYER_COUNT],
    reveal: RevealStateV2,
    pending_during_deal_reveal: [bool; PLAYER_COUNT],
    first_caller: Option<Seat>,
    call: Option<CallStateV2>,
}

fn replay_reveal_progress(
    action_history: &[AttemptActionRecordV2],
    pre_deal_reveal_order: [Seat; PLAYER_COUNT],
    first_caller_candidate: Seat,
    rules: &RuleConfigV2,
) -> Result<RevealProgressV2, MatchError> {
    let mut progress = RevealProgressV2 {
        phase: PhaseV2::PreDealReveal,
        pre_deal_reveal_cursor: 0,
        cards_received: [0; PLAYER_COUNT],
        reveal: RevealStateV2 {
            revealed: [false; PLAYER_COUNT],
            reveal_factor_by_seat: [0; PLAYER_COUNT],
            first_revealer: None,
            first_reveal_sequence: None,
            maximum_factor: 1,
        },
        pending_during_deal_reveal: [false; PLAYER_COUNT],
        first_caller: None,
        call: None,
    };
    for record in action_history {
        match &record.action {
            GameActionV2::PreDealReveal(decision) => {
                replay_pre_deal_reveal_action(
                    &mut progress,
                    record,
                    *decision,
                    pre_deal_reveal_order,
                    first_caller_candidate,
                    rules,
                )?;
            }
            GameActionV2::DuringDealReveal(decision) => {
                replay_during_deal_reveal_action(
                    &mut progress,
                    record,
                    *decision,
                    first_caller_candidate,
                    rules,
                )?;
            }
            GameActionV2::Call(decision) => {
                replay_call_action(&mut progress, record, *decision)?;
            }
            _ => {
                if !matches!(
                    progress.phase,
                    PhaseV2::Calling | PhaseV2::Robbing | PhaseV2::BottomReveal
                ) {
                    return Err(MatchError::StateInvariant(
                        "later-phase action occurred before R004/R005 opened its boundary",
                    ));
                }
            }
        }
    }
    Ok(progress)
}

fn replay_pre_deal_reveal_action(
    progress: &mut RevealProgressV2,
    record: &AttemptActionRecordV2,
    decision: RevealDecisionV2,
    pre_deal_reveal_order: [Seat; PLAYER_COUNT],
    first_caller_candidate: Seat,
    rules: &RuleConfigV2,
) -> Result<(), MatchError> {
    if progress.phase != PhaseV2::PreDealReveal {
        return Err(MatchError::StateInvariant(
            "pre-deal reveal action occurred outside the pre-deal phase",
        ));
    }
    let expected = pre_deal_reveal_order
        .get(usize::from(progress.pre_deal_reveal_cursor))
        .copied()
        .ok_or(MatchError::StateInvariant(
            "pre-deal reveal action exceeded the declaration order",
        ))?;
    if record.actor != expected {
        return Err(MatchError::StateInvariant(
            "pre-deal reveal actions must follow the declaration order",
        ));
    }
    if decision == RevealDecisionV2::Reveal {
        accept_reveal_progress(
            &mut progress.reveal,
            record.actor,
            rules.reveal.before_deal_factor,
            record.sequence,
        )?;
    }
    progress.pre_deal_reveal_cursor = progress
        .pre_deal_reveal_cursor
        .checked_add(1)
        .ok_or(MatchError::EventSequenceOverflow)?;
    if usize::from(progress.pre_deal_reveal_cursor) == PLAYER_COUNT {
        progress.phase = PhaseV2::DealingReveal;
        advance_reveal_progress_until_decision_or_calling(progress, first_caller_candidate)?;
    }
    Ok(())
}

fn replay_during_deal_reveal_action(
    progress: &mut RevealProgressV2,
    record: &AttemptActionRecordV2,
    decision: RevealDecisionV2,
    first_caller_candidate: Seat,
    rules: &RuleConfigV2,
) -> Result<(), MatchError> {
    if progress.phase != PhaseV2::DealingReveal {
        return Err(MatchError::StateInvariant(
            "during-deal reveal action occurred outside a dealing round",
        ));
    }
    let actor_index = usize::from(record.actor);
    if !progress.pending_during_deal_reveal[actor_index] {
        return Err(MatchError::StateInvariant(
            "during-deal reveal action was not pending for its seat",
        ));
    }
    if decision == RevealDecisionV2::Reveal {
        let factor = *rules
            .reveal
            .factor_by_cards_received
            .get(usize::from(progress.cards_received[actor_index]))
            .ok_or(MatchError::StateInvariant(
                "during-deal factor must use the explicit 0..17 schedule",
            ))?;
        accept_reveal_progress(&mut progress.reveal, record.actor, factor, record.sequence)?;
    }
    progress.pending_during_deal_reveal[actor_index] = false;
    if !progress
        .pending_during_deal_reveal
        .iter()
        .any(|pending| *pending)
    {
        if progress.cards_received[0]
            == u8::try_from(HUANLE_CARDS_PER_PLAYER).expect("Huanle card count fits in u8")
        {
            open_calling_progress(progress, first_caller_candidate)?;
        } else {
            advance_reveal_progress_until_decision_or_calling(progress, first_caller_candidate)?;
        }
    }
    Ok(())
}

fn replay_call_action(
    progress: &mut RevealProgressV2,
    record: &AttemptActionRecordV2,
    decision: CallDecisionV2,
) -> Result<(), MatchError> {
    if progress.phase != PhaseV2::Calling {
        return Err(MatchError::StateInvariant(
            "initial call action occurred outside the calling phase",
        ));
    }
    let call = progress.call.as_mut().ok_or(MatchError::StateInvariant(
        "calling phase must retain an explicit call state",
    ))?;
    if record.actor != call.current_player || call.acted[usize::from(record.actor)] {
        return Err(MatchError::StateInvariant(
            "call action must come once from the current caller",
        ));
    }
    let actor_index = usize::from(record.actor);
    call.acted[actor_index] = true;
    match decision {
        CallDecisionV2::CallLandlord => {
            call.caller = Some(record.actor);
            progress.phase = PhaseV2::Robbing;
        }
        CallDecisionV2::PassCall => {
            call.declined[actor_index] = true;
            if call.acted.iter().all(|acted| *acted) {
                if progress.reveal.first_revealer.is_some() {
                    progress.phase = PhaseV2::BottomReveal;
                }
            } else {
                call.current_player = next_unacted_call_seat(record.actor, call.acted)?;
            }
        }
    }
    Ok(())
}

fn accept_reveal_progress(
    reveal: &mut RevealStateV2,
    actor: Seat,
    factor: u32,
    action_sequence: u64,
) -> Result<(), MatchError> {
    let actor_index = usize::from(actor);
    if reveal.revealed[actor_index] || factor <= 1 {
        return Err(MatchError::StateInvariant(
            "replay reveal state must be an irrevocable configured multiplier",
        ));
    }
    reveal.revealed[actor_index] = true;
    reveal.reveal_factor_by_seat[actor_index] = factor;
    reveal.maximum_factor = reveal.maximum_factor.max(factor);
    if reveal.first_revealer.is_none() {
        reveal.first_revealer = Some(actor);
        reveal.first_reveal_sequence = Some(action_sequence);
    }
    Ok(())
}

fn advance_reveal_progress_until_decision_or_calling(
    progress: &mut RevealProgressV2,
    first_caller_candidate: Seat,
) -> Result<(), MatchError> {
    loop {
        if progress.phase != PhaseV2::DealingReveal {
            return Err(MatchError::StateInvariant(
                "only the dealing phase may advance its reveal progress",
            ));
        }
        if progress.cards_received[0]
            >= u8::try_from(HUANLE_CARDS_PER_PLAYER).expect("Huanle card count fits in u8")
        {
            return Err(MatchError::StateInvariant(
                "reveal progress cannot deal more than seventeen rounds",
            ));
        }
        let next = progress.cards_received[0]
            .checked_add(1)
            .ok_or(MatchError::StateInvariant(
                "dealing progress card count overflowed",
            ))?;
        progress.cards_received = [next; PLAYER_COUNT];
        progress.pending_during_deal_reveal =
            std::array::from_fn(|seat| !progress.reveal.revealed[seat]);
        if progress
            .pending_during_deal_reveal
            .iter()
            .any(|pending| *pending)
        {
            return Ok(());
        }
        if next == u8::try_from(HUANLE_CARDS_PER_PLAYER).expect("Huanle card count fits in u8") {
            return open_calling_progress(progress, first_caller_candidate);
        }
    }
}

fn open_calling_progress(
    progress: &mut RevealProgressV2,
    first_caller_candidate: Seat,
) -> Result<(), MatchError> {
    if progress.cards_received
        != [u8::try_from(HUANLE_CARDS_PER_PLAYER).expect("Huanle card count fits in u8");
            PLAYER_COUNT]
        || progress
            .pending_during_deal_reveal
            .iter()
            .any(|pending| *pending)
    {
        return Err(MatchError::StateInvariant(
            "replay may open calling only after all reveal decisions finish",
        ));
    }
    progress.phase = PhaseV2::Calling;
    progress.first_caller = Some(
        progress
            .reveal
            .first_revealer
            .unwrap_or(first_caller_candidate),
    );
    progress.call = progress.first_caller.map(new_call_state);
    Ok(())
}

fn validate_action_history(
    action_history: &[AttemptActionRecordV2],
    accepted_action_count: u64,
) -> Result<(), MatchError> {
    let history_count =
        u64::try_from(action_history.len()).map_err(|_| MatchError::ActionCountOverflow)?;
    if history_count != accepted_action_count {
        return Err(MatchError::StateInvariant(
            "accepted action count must equal action history length",
        ));
    }
    for (position, record) in action_history.iter().enumerate() {
        let expected_sequence =
            u64::try_from(position).map_err(|_| MatchError::EventSequenceOverflow)?;
        if record.sequence != expected_sequence {
            return Err(MatchError::StateInvariant(
                "attempt action sequences must be contiguous",
            ));
        }
        validate_seat(record.actor)?;
    }
    Ok(())
}

fn validate_sequences<T>(events: &[T], sequence: impl Fn(&T) -> u64) -> Result<(), MatchError> {
    for (position, event) in events.iter().enumerate() {
        let expected = u64::try_from(position).map_err(|_| MatchError::EventSequenceOverflow)?;
        if sequence(event) != expected {
            return Err(MatchError::StateInvariant(
                "event log sequences must be contiguous",
            ));
        }
    }
    Ok(())
}

fn validate_seat(seat: Seat) -> Result<(), MatchError> {
    if usize::from(seat) < PLAYER_COUNT {
        Ok(())
    } else {
        Err(MatchError::InvalidSeat { seat })
    }
}

impl AttemptActionRecordV2 {
    const fn is_reveal(&self) -> bool {
        matches!(
            self.action,
            GameActionV2::PreDealReveal(RevealDecisionV2::Reveal)
                | GameActionV2::DuringDealReveal(RevealDecisionV2::Reveal)
                | GameActionV2::PostBottomReveal(RevealDecisionV2::Reveal)
        )
    }
}

/// Errors produced by the v2 match lifecycle coordinator.
#[derive(Debug)]
pub enum MatchError {
    /// The supplied v2 profile is invalid or incompatible with the Huanle coordinator.
    RuleConfig(RuleConfigError),
    /// A seat outside the three-player table was supplied.
    InvalidSeat {
        /// Rejected seat.
        seat: Seat,
    },
    /// An action was attempted after terminal completion.
    TerminalMatch,
    /// An operation was applied outside its authoritative phase.
    UnexpectedPhase {
        /// Phase required by the operation.
        expected: PhaseV2,
        /// Active phase that rejected the operation.
        actual: PhaseV2,
    },
    /// A pre-deal declaration did not follow the seeded declaration order.
    PreDealRevealOutOfTurn {
        /// Seat that owns the next declaration.
        expected: Seat,
        /// Seat that attempted to declare instead.
        actual: Seat,
    },
    /// A seat tried to decide during a dealing round without a pending offer.
    DuringDealRevealNotPending {
        /// Rejected seat.
        seat: Seat,
    },
    /// A seat attempted to reveal after its prior reveal became irrevocable.
    RevealAlreadyIrrevocable {
        /// Rejected seat.
        seat: Seat,
    },
    /// A caller bypassed the authoritative R004 reveal state machine.
    RevealActionRequiresStateMachine,
    /// A call action was attempted through the generic audit seam instead of R005.
    CallActionRequiresStateMachine,
    /// A later phase action was attempted before its authoritative ticket exists.
    ActionRequiresOwningStateMachine {
        /// Name of the phase that must own the action.
        phase: &'static str,
    },
    /// A call decision came from a seat other than the current caller.
    CallOutOfTurn {
        /// Seat entitled to make the next call decision.
        expected: Seat,
        /// Seat that attempted the decision instead.
        actual: Seat,
    },
    /// A seat attempted a second initial calling decision.
    CallAlreadyActed {
        /// Rejected seat.
        seat: Seat,
    },
    /// A no-reveal all-pass was reported without any accepted player-action evidence.
    AllPassWithoutAcceptedActions,
    /// A no-reveal all-pass was reported after an accepted reveal action.
    AllPassAfterReveal,
    /// A no-reveal redeal was requested before every caller explicitly passed.
    NoRevealAllPassRequiresCompleteCallPass,
    /// A landlord resolution was reported without any accepted player-action evidence.
    LandlordResolutionWithoutAcceptedActions,
    /// A landlord resolution did not follow a valid R005 calling outcome.
    LandlordResolutionRequiresCallOutcome,
    /// An operation requires an unresolved attempt but a landlord already exists.
    AttemptAlreadyHasLandlord {
        /// Existing resolved landlord.
        landlord: Seat,
    },
    /// Completion was attempted before a landlord was resolved.
    LandlordNotResolved,
    /// Attempt action or match action total overflowed `u64`.
    ActionCountOverflow,
    /// A redeal would exceed the representable attempt index.
    AttemptIndexOverflow,
    /// An event log cannot represent another monotonic sequence number.
    EventSequenceOverflow,
    /// Replay addressed an attempt other than the active deterministic attempt.
    ReplayAttemptMismatch {
        /// Attempt expected from replay state.
        expected: u32,
        /// Attempt encoded by the event.
        actual: u32,
    },
    /// Replaying an event did not reproduce its exact authoritative event record.
    ReplayEventMismatch {
        /// Sequence at which replay diverged.
        sequence: u64,
    },
    /// Serialized or externally constructed state violated a lifecycle invariant.
    StateInvariant(&'static str),
}

impl Display for MatchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RuleConfig(error) => Display::fmt(error, formatter),
            Self::InvalidSeat { seat } => {
                write!(formatter, "seat {seat} is outside the Huanle table")
            }
            Self::TerminalMatch => write!(formatter, "match is terminal"),
            Self::UnexpectedPhase { expected, actual } => write!(
                formatter,
                "operation requires {expected:?}, but the active phase is {actual:?}"
            ),
            Self::PreDealRevealOutOfTurn { expected, actual } => write!(
                formatter,
                "pre-deal reveal belongs to seat {expected}, not seat {actual}"
            ),
            Self::DuringDealRevealNotPending { seat } => write!(
                formatter,
                "seat {seat} has no pending during-deal reveal decision"
            ),
            Self::RevealAlreadyIrrevocable { seat } => {
                write!(formatter, "seat {seat} has already irrevocably revealed")
            }
            Self::RevealActionRequiresStateMachine => write!(
                formatter,
                "pre-deal and during-deal reveal actions require the R004 state machine"
            ),
            Self::CallActionRequiresStateMachine => {
                write!(
                    formatter,
                    "initial call actions require the R005 state machine"
                )
            }
            Self::ActionRequiresOwningStateMachine { phase } => {
                write!(
                    formatter,
                    "{phase} actions require their owning state machine"
                )
            }
            Self::CallOutOfTurn { expected, actual } => write!(
                formatter,
                "initial call belongs to seat {expected}, not seat {actual}"
            ),
            Self::CallAlreadyActed { seat } => {
                write!(
                    formatter,
                    "seat {seat} has already made its initial call decision"
                )
            }
            Self::AllPassWithoutAcceptedActions => {
                write!(
                    formatter,
                    "all-pass attempt requires accepted action evidence"
                )
            }
            Self::AllPassAfterReveal => {
                write!(
                    formatter,
                    "no-reveal all-pass is invalid after a reveal action"
                )
            }
            Self::NoRevealAllPassRequiresCompleteCallPass => write!(
                formatter,
                "no-reveal all-pass requires three explicit initial PassCall decisions"
            ),
            Self::LandlordResolutionWithoutAcceptedActions => {
                write!(
                    formatter,
                    "landlord resolution requires accepted action evidence"
                )
            }
            Self::LandlordResolutionRequiresCallOutcome => write!(
                formatter,
                "landlord resolution requires a completed all-pass reveal branch or robbing caller"
            ),
            Self::AttemptAlreadyHasLandlord { landlord } => {
                write!(
                    formatter,
                    "attempt already resolved landlord seat {landlord}"
                )
            }
            Self::LandlordNotResolved => write!(formatter, "attempt landlord is not resolved"),
            Self::ActionCountOverflow => write!(formatter, "accepted action count overflow"),
            Self::AttemptIndexOverflow => write!(formatter, "attempt index overflow"),
            Self::EventSequenceOverflow => write!(formatter, "event sequence overflow"),
            Self::ReplayAttemptMismatch { expected, actual } => write!(
                formatter,
                "replay event targets attempt {actual}, but active attempt is {expected}"
            ),
            Self::ReplayEventMismatch { sequence } => {
                write!(formatter, "replay event mismatch at sequence {sequence}")
            }
            Self::StateInvariant(reason) => {
                write!(formatter, "invalid Huanle match state: {reason}")
            }
        }
    }
}

impl Error for MatchError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::RuleConfig(error) => Some(error),
            Self::InvalidSeat { .. }
            | Self::TerminalMatch
            | Self::UnexpectedPhase { .. }
            | Self::PreDealRevealOutOfTurn { .. }
            | Self::DuringDealRevealNotPending { .. }
            | Self::RevealAlreadyIrrevocable { .. }
            | Self::RevealActionRequiresStateMachine
            | Self::CallActionRequiresStateMachine
            | Self::ActionRequiresOwningStateMachine { .. }
            | Self::CallOutOfTurn { .. }
            | Self::CallAlreadyActed { .. }
            | Self::AllPassWithoutAcceptedActions
            | Self::AllPassAfterReveal
            | Self::NoRevealAllPassRequiresCompleteCallPass
            | Self::LandlordResolutionWithoutAcceptedActions
            | Self::LandlordResolutionRequiresCallOutcome
            | Self::AttemptAlreadyHasLandlord { .. }
            | Self::LandlordNotResolved
            | Self::ActionCountOverflow
            | Self::AttemptIndexOverflow
            | Self::EventSequenceOverflow
            | Self::ReplayAttemptMismatch { .. }
            | Self::ReplayEventMismatch { .. }
            | Self::StateInvariant(_) => None,
        }
    }
}
