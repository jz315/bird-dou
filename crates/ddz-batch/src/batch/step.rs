use ddz_core::{GameAction, Seat};
use ddz_rules::{StepResult, UndoToken};

use crate::protocol::{PackedStepResult, StepPackRow};
use crate::{BatchError, SlotVersion, SKIP_ACTION_INDEX};

use super::{observe, BatchEnv};

#[derive(Clone, Copy, Debug)]
struct Selection {
    local_index: i64,
    actor: Option<Seat>,
    action: Option<GameAction>,
}

impl BatchEnv {
    /// Atomically apply local action indices. `-1` leaves a slot unchanged.
    pub fn step_packed(
        &mut self,
        local_action_indices: &[i64],
    ) -> Result<PackedStepResult, BatchError> {
        self.step_impl(local_action_indices, None, None)
    }

    /// Atomically apply local action indices and reject stale asynchronous policy responses.
    pub fn step_packed_checked(
        &mut self,
        local_action_indices: &[i64],
        expected_generation: &[u64],
        expected_revision: &[u64],
    ) -> Result<PackedStepResult, BatchError> {
        self.step_impl(
            local_action_indices,
            Some(expected_generation),
            Some(expected_revision),
        )
    }

    fn step_impl(
        &mut self,
        local_action_indices: &[i64],
        expected_generation: Option<&[u64]>,
        expected_revision: Option<&[u64]>,
    ) -> Result<PackedStepResult, BatchError> {
        self.require_initialized()?;
        self.require_batch_length("local_action_indices", local_action_indices.len())?;
        if let Some(values) = expected_generation {
            self.require_batch_length("expected_generation", values.len())?;
        }
        if let Some(values) = expected_revision {
            self.require_batch_length("expected_revision", values.len())?;
        }
        self.validate_versions(
            local_action_indices,
            expected_generation,
            expected_revision,
        )?;

        let selections = self.select_actions(local_action_indices)?;
        let next_versions = selections
            .iter()
            .enumerate()
            .map(|(slot, selection)| {
                if selection.action.is_some() {
                    self.slots[slot]
                        .version
                        .advanced()
                        .ok_or(BatchError::RevisionOverflow { slot })
                } else {
                    Ok(self.slots[slot].version)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let old_versions = self
            .slots
            .iter()
            .map(|slot| slot.version)
            .collect::<Vec<_>>();
        let mut applied = Vec::<(usize, UndoToken)>::new();
        let mut results = vec![None; self.slots.len()];

        for (slot, selection) in selections.iter().copied().enumerate() {
            let Some(action) = selection.action else {
                continue;
            };
            let actor = selection.actor.expect("selected actions always carry an actor");
            match self.slots[slot].game.apply_with_undo(actor, action) {
                Ok((result, undo)) => {
                    applied.push((slot, undo));
                    results[slot] = Some(result);
                }
                Err(source) => {
                    self.rollback(&mut applied)?;
                    return Err(BatchError::Step { slot, source });
                }
            }
        }

        for (slot, version) in next_versions.iter().copied().enumerate() {
            self.slots[slot].version = version;
        }

        let transition_rewards = match self.transition_rewards(&results) {
            Ok(values) => values,
            Err(error) => {
                self.restore_after_failed_pack(&mut applied, &old_versions)?;
                return Err(error);
            }
        };
        let observation = match observe::pack_current(&self.slots) {
            Ok(value) => value,
            Err(error) => {
                self.restore_after_failed_pack(&mut applied, &old_versions)?;
                return Err(error);
            }
        };
        let rows = self
            .slots
            .iter()
            .enumerate()
            .map(|(slot, value)| StepPackRow {
                state: value.game.state(),
                version: value.version,
                selected_local_index: selections[slot].local_index,
                result: results[slot].as_ref(),
                transition_reward: transition_rewards[slot],
            })
            .collect::<Vec<_>>();
        let packed = match PackedStepResult::from_rows(&rows, observation) {
            Ok(value) => value,
            Err(error) => {
                self.restore_after_failed_pack(&mut applied, &old_versions)?;
                return Err(error);
            }
        };

        self.legal_cache.invalidate_many(
            selections
                .iter()
                .enumerate()
                .filter_map(|(slot, selection)| selection.action.is_some().then_some(slot)),
        );
        Ok(packed)
    }

    fn select_actions(&mut self, indices: &[i64]) -> Result<Vec<Selection>, BatchError> {
        let mut selections = Vec::with_capacity(self.slots.len());
        for (slot, &index) in indices.iter().enumerate() {
            if index == SKIP_ACTION_INDEX {
                selections.push(Selection {
                    local_index: index,
                    actor: None,
                    action: None,
                });
                continue;
            }
            if index < SKIP_ACTION_INDEX {
                return Err(BatchError::InvalidActionIndex { slot, index });
            }
            if self.slots[slot].game.state().is_terminal() {
                return Err(BatchError::TerminalActionIndex { slot, index });
            }
            let actor = self.slots[slot]
                .game
                .state()
                .current_player
                .ok_or(BatchError::InternalInvariant(
                    "non-terminal slot has no current player",
                ))?;
            self.ensure_legal_slot(slot)?;
            let actions = self
                .legal_cache
                .get(slot, self.slots[slot].version)
                .ok_or(BatchError::InternalInvariant(
                    "current legal-action cache entry disappeared",
                ))?;
            let local = usize::try_from(index)
                .map_err(|_| BatchError::InvalidActionIndex { slot, index })?;
            let action = actions
                .get(local)
                .copied()
                .ok_or(BatchError::ActionIndexOutOfRange {
                    slot,
                    index,
                    legal_count: actions.len(),
                })?;
            selections.push(Selection {
                local_index: index,
                actor: Some(actor),
                action: Some(action),
            });
        }
        Ok(selections)
    }

    fn validate_versions(
        &self,
        indices: &[i64],
        generations: Option<&[u64]>,
        revisions: Option<&[u64]>,
    ) -> Result<(), BatchError> {
        for (slot, value) in self.slots.iter().enumerate() {
            if indices[slot] == SKIP_ACTION_INDEX {
                continue;
            }
            if let Some(generations) = generations {
                if generations[slot] != value.version.generation {
                    return Err(BatchError::StaleGeneration {
                        slot,
                        expected: value.version.generation,
                        actual: generations[slot],
                    });
                }
            }
            if let Some(revisions) = revisions {
                if revisions[slot] != value.version.revision {
                    return Err(BatchError::StaleRevision {
                        slot,
                        expected: value.version.revision,
                        actual: revisions[slot],
                    });
                }
            }
        }
        Ok(())
    }

    fn transition_rewards(
        &self,
        results: &[Option<StepResult>],
    ) -> Result<Vec<[i64; 3]>, BatchError> {
        let mut rewards = vec![[0; 3]; self.slots.len()];
        for (slot, result) in results.iter().enumerate() {
            if !result.as_ref().is_some_and(|value| value.terminal) {
                continue;
            }
            for seat in Seat::ALL {
                rewards[slot][seat.index()] = self.slots[slot]
                    .game
                    .terminal_reward(seat)
                    .map_err(|source| BatchError::Step { slot, source })?;
            }
        }
        Ok(rewards)
    }

    fn require_batch_length(&self, field: &'static str, actual: usize) -> Result<(), BatchError> {
        if actual == self.slots.len() {
            Ok(())
        } else {
            Err(BatchError::BatchSizeMismatch {
                field,
                expected: self.slots.len(),
                actual,
            })
        }
    }

    fn rollback(&mut self, applied: &mut Vec<(usize, UndoToken)>) -> Result<(), BatchError> {
        while let Some((slot, token)) = applied.pop() {
            self.slots[slot]
                .game
                .undo(token)
                .map_err(|source| BatchError::Rollback { slot, source })?;
        }
        Ok(())
    }

    fn restore_after_failed_pack(
        &mut self,
        applied: &mut Vec<(usize, UndoToken)>,
        versions: &[SlotVersion],
    ) -> Result<(), BatchError> {
        self.rollback(applied)?;
        for (slot, version) in versions.iter().copied().enumerate() {
            self.slots[slot].version = version;
        }
        Ok(())
    }
}
