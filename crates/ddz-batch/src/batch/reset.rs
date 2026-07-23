use std::collections::BTreeSet;

use ddz_rules::{Game, RuleProfile};

use crate::protocol::PackedObservation;
use crate::{BatchError, ResetSpec, SlotReset, SlotVersion};

use super::{observe, BatchEnv, Slot};

impl BatchEnv {
    /// Transactionally replace the entire batch.
    pub fn reset_all(&mut self, specs: &[ResetSpec]) -> Result<PackedObservation, BatchError> {
        if specs.is_empty() {
            return Err(BatchError::EmptyBatch);
        }
        let (generations, next_generation) = self.checked_generations(specs.len())?;
        let slots = specs
            .iter()
            .copied()
            .zip(generations)
            .enumerate()
            .map(|(slot, (spec, generation))| {
                build_game(&self.rules, spec, slot).map(|game| Slot {
                    game,
                    version: SlotVersion::initial(generation),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let observation = observe::pack_current(&slots)?;

        self.slots = slots;
        self.legal_cache.reset(self.slots.len());
        self.next_generation = next_generation;
        Ok(observation)
    }

    /// Transactionally reset selected existing slots while all others continue unchanged.
    pub fn reset_slots(&mut self, resets: &[SlotReset]) -> Result<PackedObservation, BatchError> {
        self.require_initialized()?;
        if resets.is_empty() {
            return self.observations_current();
        }
        let mut seen = BTreeSet::new();
        for reset in resets {
            if reset.slot >= self.slots.len() {
                return Err(BatchError::SlotOutOfRange {
                    slot: reset.slot,
                    batch_size: self.slots.len(),
                });
            }
            if !seen.insert(reset.slot) {
                return Err(BatchError::DuplicateSlot { slot: reset.slot });
            }
        }

        let (generations, next_generation) = self.checked_generations(resets.len())?;
        let replacements = resets
            .iter()
            .zip(generations)
            .map(|(reset, generation)| {
                build_game(&self.rules, reset.spec, reset.slot).map(|game| {
                    (
                        reset.slot,
                        Slot {
                            game,
                            version: SlotVersion::initial(generation),
                        },
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut previous = Vec::with_capacity(replacements.len());
        for (slot, replacement) in replacements {
            previous.push((slot, std::mem::replace(&mut self.slots[slot], replacement)));
        }
        match observe::pack_current(&self.slots) {
            Ok(observation) => {
                self.legal_cache
                    .invalidate_many(previous.iter().map(|(slot, _)| *slot));
                self.next_generation = next_generation;
                Ok(observation)
            }
            Err(error) => {
                for (slot, old) in previous {
                    self.slots[slot] = old;
                }
                Err(error)
            }
        }
    }
}

fn build_game(
    rules: &ddz_rules::RuleConfig,
    spec: ResetSpec,
    slot: usize,
) -> Result<Game, BatchError> {
    match (rules.profile, spec) {
        (
            RuleProfile::HuanleClassic,
            ResetSpec::Huanle {
                match_seed,
                economy,
            },
        ) => Game::new_huanle(rules.clone(), match_seed, economy)
            .map_err(|source| BatchError::Reset { slot, source }),
        (
            RuleProfile::DouzeroPostBid,
            ResetSpec::PostBid {
                match_seed,
                landlord,
            },
        ) => Game::new_post_bid(rules.clone(), match_seed, landlord)
            .map_err(|source| BatchError::Reset { slot, source }),
        (RuleProfile::HuanleClassic, ResetSpec::PostBid { .. }) => {
            Err(BatchError::ResetProfileMismatch {
                slot,
                profile: rules.profile,
                expected: "ResetSpec::Huanle",
            })
        }
        (RuleProfile::DouzeroPostBid, ResetSpec::Huanle { .. }) => {
            Err(BatchError::ResetProfileMismatch {
                slot,
                profile: rules.profile,
                expected: "ResetSpec::PostBid",
            })
        }
    }
}
