//! Packed player actions and ragged legal-action ranges.

use ddz_core::{GameAction, RankCounts, RANK_COUNT};

use crate::{BatchError, SlotVersion, BATCH_SCHEMA_VERSION};

use super::codes::{self, NO_ACTION, NO_RANK, NO_U8};

/// Structure-of-arrays representation of player actions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedActionData {
    /// High-level action class: reveal/call/rob/double/play.
    pub kind: Vec<u8>,
    /// Binary phase action code, or [`NO_U8`] for card play.
    pub subcode: Vec<u8>,
    /// `MoveKind` numeric tag, or [`NO_ACTION`] for non-play actions.
    pub move_kind: Vec<u8>,
    /// Flattened `[N, 15]` rank counts. Non-play rows are zero.
    pub move_cards: Vec<u8>,
    /// Move main rank. Non-play rows use [`NO_RANK`].
    pub move_main_rank: Vec<u8>,
    /// Move body-chain length. Non-play rows use zero.
    pub move_chain_len: Vec<u8>,
    /// Move total card count. Non-play rows use zero.
    pub move_total_cards: Vec<u8>,
}

impl PackedActionData {
    #[must_use]
    pub fn with_capacity(actions: usize) -> Self {
        Self {
            kind: Vec::with_capacity(actions),
            subcode: Vec::with_capacity(actions),
            move_kind: Vec::with_capacity(actions),
            move_cards: Vec::with_capacity(actions.saturating_mul(RANK_COUNT)),
            move_main_rank: Vec::with_capacity(actions),
            move_chain_len: Vec::with_capacity(actions),
            move_total_cards: Vec::with_capacity(actions),
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.kind.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.kind.is_empty()
    }

    pub(crate) fn push(&mut self, action: GameAction) {
        let (kind, subcode) = codes::action(action);
        self.kind.push(kind);
        self.subcode.push(subcode);
        match action {
            GameAction::Play(movement) => {
                self.move_kind.push(movement.kind() as u8);
                self.move_cards.extend_from_slice(movement.cards().as_array());
                self.move_main_rank.push(movement.main_rank());
                self.move_chain_len.push(movement.chain_len());
                self.move_total_cards.push(movement.total_cards());
            }
            GameAction::Reveal(_)
            | GameAction::Call(_)
            | GameAction::Rob(_)
            | GameAction::Double(_) => self.push_no_move(),
        }
    }

    pub(crate) fn push_none(&mut self) {
        self.kind.push(NO_ACTION);
        self.subcode.push(NO_U8);
        self.push_no_move();
    }

    fn push_no_move(&mut self) {
        self.move_kind.push(NO_ACTION);
        self.move_cards
            .extend_from_slice(RankCounts::empty().as_array());
        self.move_main_rank.push(NO_RANK);
        self.move_chain_len.push(0);
        self.move_total_cards.push(0);
    }

    pub(crate) fn validate(&self, expected: usize) -> Result<(), BatchError> {
        let scalar_lengths = [
            self.kind.len(),
            self.subcode.len(),
            self.move_kind.len(),
            self.move_main_rank.len(),
            self.move_chain_len.len(),
            self.move_total_cards.len(),
        ];
        if scalar_lengths.into_iter().any(|length| length != expected) {
            return Err(BatchError::InternalInvariant(
                "packed action scalar columns have inconsistent lengths",
            ));
        }
        if self.move_cards.len() != expected.saturating_mul(RANK_COUNT) {
            return Err(BatchError::InternalInvariant(
                "packed action card buffer has an inconsistent length",
            ));
        }
        Ok(())
    }
}

/// Ragged legal actions for a whole batch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedActions {
    /// Packed-buffer protocol version.
    pub schema_version: u32,
    /// Number of environment slots.
    pub batch_size: usize,
    /// Ragged action ranges. Slot `i` owns `[offsets[i], offsets[i + 1])`.
    pub offsets: Vec<u64>,
    /// Owning slot for each packed action.
    pub owner: Vec<u32>,
    /// Slot generation corresponding to every range.
    pub generation: Vec<u64>,
    /// Slot revision corresponding to every range.
    pub revision: Vec<u64>,
    /// Packed action fields.
    pub actions: PackedActionData,
}

impl PackedActions {
    pub(crate) fn from_rows<'a>(
        rows: impl IntoIterator<Item = (SlotVersion, &'a [GameAction])>,
    ) -> Result<Self, BatchError> {
        let rows = rows.into_iter().collect::<Vec<_>>();
        let total = rows.iter().try_fold(0_usize, |accumulator, (_, actions)| {
            accumulator
                .checked_add(actions.len())
                .ok_or(BatchError::BufferOverflow {
                    field: "legal action count",
                })
        })?;
        let mut result = Self {
            schema_version: BATCH_SCHEMA_VERSION,
            batch_size: rows.len(),
            offsets: Vec::with_capacity(rows.len() + 1),
            owner: Vec::with_capacity(total),
            generation: Vec::with_capacity(rows.len()),
            revision: Vec::with_capacity(rows.len()),
            actions: PackedActionData::with_capacity(total),
        };
        result.offsets.push(0);
        for (slot, (version, actions)) in rows.into_iter().enumerate() {
            let owner = u32::try_from(slot).map_err(|_| BatchError::BufferOverflow {
                field: "legal action owner",
            })?;
            result.generation.push(version.generation);
            result.revision.push(version.revision);
            for &action in actions {
                result.owner.push(owner);
                result.actions.push(action);
            }
            result.offsets.push(
                u64::try_from(result.actions.len()).map_err(|_| BatchError::BufferOverflow {
                    field: "legal action offsets",
                })?,
            );
        }
        result.validate()?;
        Ok(result)
    }

    #[must_use]
    pub fn action_count(&self) -> usize {
        self.actions.len()
    }

    #[must_use]
    pub fn local_count(&self, slot: usize) -> Option<usize> {
        let start = usize::try_from(*self.offsets.get(slot)?).ok()?;
        let end = usize::try_from(*self.offsets.get(slot + 1)?).ok()?;
        end.checked_sub(start)
    }

    #[must_use]
    pub fn global_index(&self, slot: usize, local_index: usize) -> Option<usize> {
        let start = usize::try_from(*self.offsets.get(slot)?).ok()?;
        let end = usize::try_from(*self.offsets.get(slot + 1)?).ok()?;
        let global = start.checked_add(local_index)?;
        (global < end).then_some(global)
    }

    pub fn validate(&self) -> Result<(), BatchError> {
        if self.offsets.len() != self.batch_size + 1
            || self.generation.len() != self.batch_size
            || self.revision.len() != self.batch_size
        {
            return Err(BatchError::InternalInvariant(
                "packed legal-action batch columns have inconsistent lengths",
            ));
        }
        if self.offsets.first().copied() != Some(0)
            || self.offsets.windows(2).any(|pair| pair[0] > pair[1])
            || self.offsets.last().copied()
                != Some(u64::try_from(self.action_count()).map_err(|_| {
                    BatchError::BufferOverflow {
                        field: "legal action final offset",
                    }
                })?)
            || self.owner.len() != self.action_count()
        {
            return Err(BatchError::InternalInvariant(
                "packed legal-action offsets are invalid",
            ));
        }
        for slot in 0..self.batch_size {
            let start = usize::try_from(self.offsets[slot]).map_err(|_| {
                BatchError::BufferOverflow {
                    field: "legal action start offset",
                }
            })?;
            let end = usize::try_from(self.offsets[slot + 1]).map_err(|_| {
                BatchError::BufferOverflow {
                    field: "legal action end offset",
                }
            })?;
            let expected_owner = u32::try_from(slot).map_err(|_| BatchError::BufferOverflow {
                field: "legal action owner",
            })?;
            if self.owner[start..end]
                .iter()
                .any(|&owner| owner != expected_owner)
            {
                return Err(BatchError::InternalInvariant(
                    "packed legal-action owner differs from its ragged range",
                ));
            }
        }
        self.actions.validate(self.action_count())
    }
}
