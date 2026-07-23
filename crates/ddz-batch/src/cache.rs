//! Revision-aware legal-action cache.

use ddz_core::GameAction;

use crate::SlotVersion;

#[derive(Clone, Debug)]
pub(crate) struct LegalEntry {
    pub(crate) version: SlotVersion,
    pub(crate) actions: Vec<GameAction>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct LegalCache {
    entries: Vec<Option<LegalEntry>>,
}

impl LegalCache {
    pub(crate) fn reset(&mut self, size: usize) {
        self.entries.clear();
        self.entries.resize_with(size, || None);
    }

    pub(crate) fn get(&self, slot: usize, version: SlotVersion) -> Option<&[GameAction]> {
        self.entries
            .get(slot)
            .and_then(Option::as_ref)
            .filter(|entry| entry.version == version)
            .map(|entry| entry.actions.as_slice())
    }

    pub(crate) fn insert(
        &mut self,
        slot: usize,
        version: SlotVersion,
        actions: Vec<GameAction>,
    ) {
        self.entries[slot] = Some(LegalEntry { version, actions });
    }

    pub(crate) fn invalidate(&mut self, slot: usize) {
        if let Some(entry) = self.entries.get_mut(slot) {
            *entry = None;
        }
    }

    pub(crate) fn invalidate_many(&mut self, slots: impl IntoIterator<Item = usize>) {
        for slot in slots {
            self.invalidate(slot);
        }
    }
}
