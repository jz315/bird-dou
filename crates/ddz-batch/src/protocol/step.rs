//! Packed result of one atomic batch step.

use ddz_core::{GameState, Seat};
use ddz_rules::StepResult;

use crate::protocol::codes::{self, NO_SEAT};
use crate::{BatchError, SlotVersion, BATCH_SCHEMA_VERSION};

use super::{PackedEvents, PackedObservation};

pub(crate) struct StepPackRow<'a> {
    pub(crate) state: &'a GameState,
    pub(crate) version: SlotVersion,
    pub(crate) selected_local_index: i64,
    pub(crate) result: Option<&'a StepResult>,
    pub(crate) transition_reward: [i64; 3],
}

/// Transition metadata, reward, authoritative event deltas, and next fixed observation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedStepResult {
    /// Packed-buffer protocol version.
    pub schema_version: u32,
    /// Number of environment slots.
    pub batch_size: usize,
    /// Whether the slot committed an action in this call.
    pub transitioned: Vec<u8>,
    /// Selected local legal-action index, or `-1` for skipped slots.
    pub selected_local_index: Vec<i64>,
    /// Acting seat, or `-1` for skipped slots.
    pub actor: Vec<i8>,
    /// Phase before the action.
    pub phase_before: Vec<u8>,
    /// Phase after the action.
    pub phase_after: Vec<u8>,
    /// Whether this action caused the slot to become terminal.
    pub became_terminal: Vec<u8>,
    /// Current terminal flag after the call.
    pub done: Vec<u8>,
    /// Slot generation after the call.
    pub generation: Vec<u64>,
    /// Slot revision after the call.
    pub revision: Vec<u64>,
    /// Flattened `[B, 3]` learner reward emitted by this transition only.
    pub reward: Vec<i64>,
    /// Flattened `[B, 3]` raw terminal payoff snapshot, zero before terminal.
    pub raw_payoff: Vec<i64>,
    /// Raw authoritative events emitted by the engine.
    ///
    /// During an unresolved doubling round these events may include choices that are hidden from
    /// another player's information set. Model history must come from `public_history_packed`.
    pub authoritative_events: PackedEvents,
    /// Next fixed-width current-player observation. Terminal rows are marked invalid.
    pub observation: PackedObservation,
}

impl PackedStepResult {
    pub(crate) fn from_rows(
        rows: &[StepPackRow<'_>],
        observation: PackedObservation,
    ) -> Result<Self, BatchError> {
        let event_rows = rows
            .iter()
            .map(|row| {
                row.result
                    .map_or(&[][..], |result| result.emitted_events.as_slice())
            })
            .collect::<Vec<_>>();
        let mut result = Self {
            schema_version: BATCH_SCHEMA_VERSION,
            batch_size: rows.len(),
            transitioned: Vec::with_capacity(rows.len()),
            selected_local_index: Vec::with_capacity(rows.len()),
            actor: Vec::with_capacity(rows.len()),
            phase_before: Vec::with_capacity(rows.len()),
            phase_after: Vec::with_capacity(rows.len()),
            became_terminal: Vec::with_capacity(rows.len()),
            done: Vec::with_capacity(rows.len()),
            generation: Vec::with_capacity(rows.len()),
            revision: Vec::with_capacity(rows.len()),
            reward: Vec::with_capacity(rows.len().saturating_mul(3)),
            raw_payoff: Vec::with_capacity(rows.len().saturating_mul(3)),
            authoritative_events: PackedEvents::from_rows(event_rows)?,
            observation,
        };
        for row in rows {
            result.transitioned.push(u8::from(row.result.is_some()));
            result.selected_local_index.push(row.selected_local_index);
            result.actor.push(
                row.result
                    .map_or(NO_SEAT, |value| codes::seat(value.actor)),
            );
            let phase_before = row
                .result
                .map_or(row.state.phase, |value| value.phase_before);
            let phase_after = row
                .result
                .map_or(row.state.phase, |value| value.phase_after);
            result.phase_before.push(codes::phase(phase_before));
            result.phase_after.push(codes::phase(phase_after));
            result.became_terminal.push(u8::from(
                row.result.is_some_and(|value| value.terminal),
            ));
            result.done.push(u8::from(row.state.is_terminal()));
            result.generation.push(row.version.generation);
            result.revision.push(row.version.revision);
            result.reward.extend_from_slice(&row.transition_reward);
            match &row.state.outcome {
                Some(outcome) => {
                    for seat in Seat::ALL {
                        result.raw_payoff.push(outcome.payoff[seat]);
                    }
                }
                None => result.raw_payoff.extend_from_slice(&[0; 3]),
            }
        }
        result.validate()?;
        Ok(result)
    }

    pub fn validate(&self) -> Result<(), BatchError> {
        let batch = self.batch_size;
        let scalar_lengths = [
            self.transitioned.len(),
            self.selected_local_index.len(),
            self.actor.len(),
            self.phase_before.len(),
            self.phase_after.len(),
            self.became_terminal.len(),
            self.done.len(),
            self.generation.len(),
            self.revision.len(),
        ];
        if scalar_lengths.into_iter().any(|length| length != batch)
            || self.reward.len() != batch.saturating_mul(3)
            || self.raw_payoff.len() != batch.saturating_mul(3)
            || self.authoritative_events.batch_size != batch
            || self.observation.batch_size != batch
        {
            return Err(BatchError::InternalInvariant(
                "packed step-result columns have inconsistent lengths",
            ));
        }
        self.authoritative_events.validate()?;
        self.observation.validate()
    }
}
