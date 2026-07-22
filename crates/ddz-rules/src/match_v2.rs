//! Deterministic Huanle match and deal-attempt lifecycle coordination.
//!
//! This module deliberately owns only match boundaries: deterministic attempt
//! seeds, all-pass redeals, action-budget accounting, and replayable lifecycle
//! decisions. Reveal, calling, robbing, doubling, card play, and settlement
//! remain authoritative phase implementations in their respective tickets.

use std::error::Error;
use std::fmt::{Display, Formatter};

use ddz_core::{CardId, Move, Seat};
use serde::{Deserialize, Serialize};

use crate::{
    derive_attempt_seed, shuffled_deck_for_seed, RuleConfigError, RuleConfigV2, RuleProfile,
    ATTEMPT_SEED_DERIVATION_ALGORITHM, PLAYER_COUNT, SHUFFLE_ALGORITHM,
};

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
            state,
            decision_events: Vec::new(),
            system_events: Vec::new(),
        };
        game.push_attempt_started_system_event()?;
        game.validate()?;
        Ok(game)
    }

    /// Return immutable lifecycle state for observations, replay, and audit.
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

    /// Record one accepted player action without imposing a game-rule cap.
    ///
    /// Phase state machines call this after, and only after, an action has been accepted. It is
    /// an audit budget, not a rollout or trainer safety limit.
    ///
    /// # Errors
    ///
    /// Returns an error for a terminal match, invalid actor, or arithmetic/sequence overflow.
    pub fn record_accepted_action(
        &mut self,
        actor: Seat,
        action: GameActionV2,
    ) -> Result<(), MatchError> {
        self.ensure_not_terminal()?;
        validate_seat(actor)?;
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
            .push(action_record);
        self.decision_events.push(event);
        self.validate()
    }

    /// Close a no-reveal all-pass attempt and automatically begin the next deterministic attempt.
    ///
    /// Calling logic is intentionally outside R003. Its authoritative implementation must first
    /// record each accepted action through [`Self::record_accepted_action`] and then call
    /// this method only for the frozen no-reveal all-pass branch.
    ///
    /// # Errors
    ///
    /// Returns an error if no accepted action supports the disposition, a landlord was already
    /// resolved, the match is terminal, or an index/sequence would overflow.
    pub fn resolve_no_reveal_all_pass(&mut self) -> Result<(), MatchError> {
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

        let from_attempt = self.state.attempt_index;
        let to_attempt = from_attempt
            .checked_add(1)
            .ok_or(MatchError::AttemptIndexOverflow)?;
        let summary = AttemptSummaryV2 {
            attempt_index: from_attempt,
            deal_seed: self.state.current_attempt.deal_seed,
            shuffle_algorithm: self.state.current_attempt.shuffle_algorithm.clone(),
            first_caller_candidate: self.state.current_attempt.first_caller_candidate,
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
        self.validate()
    }

    /// Record a landlord resolution produced by a future validated call/rob state machine.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid seat, absent accepted action evidence, an already resolved
    /// attempt, or a terminal match.
    pub fn record_landlord_resolution(&mut self, landlord: Seat) -> Result<(), MatchError> {
        self.ensure_not_terminal()?;
        self.ensure_unresolved_attempt()?;
        validate_seat(landlord)?;
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
        self.validate()
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
        self.validate()
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
        for expected in decision_events {
            match expected.clone() {
                MatchDecisionEventV2::PlayerActionAccepted {
                    attempt_index,
                    action,
                    ..
                } => {
                    replay.require_current_attempt(attempt_index)?;
                    replay.record_accepted_action(action.actor, action.action)?;
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
            let actual =
                replay
                    .decision_events
                    .last()
                    .cloned()
                    .ok_or(MatchError::StateInvariant(
                        "replay did not record a decision event",
                    ))?;
            if actual != *expected {
                return Err(MatchError::ReplayEventMismatch {
                    sequence: expected.sequence(),
                });
            }
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
        validate_attempt(state.match_seed, &state.current_attempt)?;
        for (position, summary) in state.completed_attempts.iter().enumerate() {
            let expected_index = u32::try_from(position)
                .map_err(|_| MatchError::StateInvariant("attempt position exceeds u32"))?;
            if summary.attempt_index != expected_index {
                return Err(MatchError::StateInvariant(
                    "completed attempt summaries must be contiguous",
                ));
            }
            validate_summary(state.match_seed, summary)?;
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

    fn next_decision_sequence(&self) -> Result<u64, MatchError> {
        u64::try_from(self.decision_events.len()).map_err(|_| MatchError::EventSequenceOverflow)
    }

    fn next_system_sequence(&self) -> Result<u64, MatchError> {
        u64::try_from(self.system_events.len()).map_err(|_| MatchError::EventSequenceOverflow)
    }

    fn push_attempt_started_system_event(&mut self) -> Result<(), MatchError> {
        let sequence = self.next_system_sequence()?;
        let attempt = &self.state.current_attempt;
        self.system_events.push(SystemEventRecordV2 {
            sequence,
            event: SystemEventV2::AttemptStarted {
                attempt_index: attempt.attempt_index,
                deal_seed: attempt.deal_seed,
                first_caller_candidate: attempt.first_caller_candidate,
            },
        });
        Ok(())
    }
}

fn new_attempt(match_seed: u64, attempt_index: u32) -> DealAttemptStateV2 {
    let deal_seed = derive_attempt_seed(match_seed, attempt_index);
    DealAttemptStateV2 {
        attempt_index,
        deal_seed,
        shuffle_algorithm: SHUFFLE_ALGORITHM.to_owned(),
        first_caller_candidate: u8::try_from(deal_seed % (PLAYER_COUNT as u64))
            .expect("seat candidate is in 0..3"),
        deck: shuffled_deck_for_seed(deal_seed).to_vec(),
        accepted_action_count: 0,
        action_history: Vec::new(),
        status: AttemptStatusV2::Unresolved,
    }
}

fn validate_attempt(match_seed: u64, attempt: &DealAttemptStateV2) -> Result<(), MatchError> {
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
    if attempt.deck != shuffled_deck_for_seed(attempt.deal_seed).to_vec() {
        return Err(MatchError::StateInvariant(
            "attempt deck must match the deterministic deal seed",
        ));
    }
    if let AttemptStatusV2::LandlordResolved { landlord } = attempt.status {
        validate_seat(landlord)?;
    }
    validate_action_history(&attempt.action_history, attempt.accepted_action_count)?;
    Ok(())
}

fn validate_summary(match_seed: u64, summary: &AttemptSummaryV2) -> Result<(), MatchError> {
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
    /// A no-reveal all-pass was reported without any accepted player-action evidence.
    AllPassWithoutAcceptedActions,
    /// A no-reveal all-pass was reported after an accepted reveal action.
    AllPassAfterReveal,
    /// A landlord resolution was reported without any accepted player-action evidence.
    LandlordResolutionWithoutAcceptedActions,
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
            Self::LandlordResolutionWithoutAcceptedActions => {
                write!(
                    formatter,
                    "landlord resolution requires accepted action evidence"
                )
            }
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
            | Self::AllPassWithoutAcceptedActions
            | Self::AllPassAfterReveal
            | Self::LandlordResolutionWithoutAcceptedActions
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
