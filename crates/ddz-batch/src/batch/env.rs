use ddz_core::GameState;
use ddz_rules::RuleConfig;

use crate::cache::LegalCache;
use crate::{BatchError, SlotVersion};

use super::Slot;

/// One Rust owner of multiple independent authoritative environments.
///
/// Every slot shares one immutable [`RuleConfig`]. Reset, restore, legal-action generation,
/// observation packing, and stepping cross the language boundary once per batch rather than once
/// per environment.
#[derive(Clone, Debug)]
pub struct BatchEnv {
    pub(crate) rules: RuleConfig,
    pub(crate) slots: Vec<Slot>,
    pub(crate) legal_cache: LegalCache,
    pub(crate) next_generation: u64,
}

impl BatchEnv {
    /// Construct an empty batch under one immutable rule configuration.
    pub fn new(rules: RuleConfig) -> Result<Self, BatchError> {
        rules.validate().map_err(BatchError::RuleConfig)?;
        Ok(Self {
            rules,
            slots: Vec::new(),
            legal_cache: LegalCache::default(),
            next_generation: 1,
        })
    }

    /// Shared immutable rules.
    #[must_use]
    pub const fn rules(&self) -> &RuleConfig {
        &self.rules
    }

    /// Number of initialized slots.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether no slots have been initialized.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Whether every slot is terminal.
    #[must_use]
    pub fn all_terminal(&self) -> bool {
        !self.slots.is_empty() && self.slots.iter().all(|slot| slot.game.state().is_terminal())
    }

    /// Number of non-terminal slots.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| !slot.game.state().is_terminal())
            .count()
    }

    /// Read one authoritative state without allowing callers to bypass version/cache updates.
    #[must_use]
    pub fn state(&self, slot: usize) -> Option<&GameState> {
        self.slots.get(slot).map(|value| value.game.state())
    }

    /// Current slot identity.
    #[must_use]
    pub fn version(&self, slot: usize) -> Option<SlotVersion> {
        self.slots.get(slot).map(|value| value.version)
    }

    pub(crate) fn require_initialized(&self) -> Result<(), BatchError> {
        if self.slots.is_empty() {
            Err(BatchError::Uninitialized)
        } else {
            Ok(())
        }
    }

    pub(crate) fn checked_generations(
        &self,
        count: usize,
    ) -> Result<(Vec<u64>, u64), BatchError> {
        let count = u64::try_from(count).map_err(|_| BatchError::GenerationOverflow)?;
        let end = self
            .next_generation
            .checked_add(count)
            .ok_or(BatchError::GenerationOverflow)?;
        Ok(((self.next_generation..end).collect(), end))
    }
}
