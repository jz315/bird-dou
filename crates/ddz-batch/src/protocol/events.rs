//! Ragged public or authoritative event streams.

use ddz_core::{GameEvent, GameEventKind, PlayerEvent, SystemEvent};

use crate::{BatchError, BATCH_SCHEMA_VERSION};

use super::actions::PackedActionData;
use super::codes::{
    EVENT_PLAYER, EVENT_SYSTEM, NO_EVENT_CODE, NO_SEAT, SYSTEM_BOTTOM_REVEALED,
    SYSTEM_CARD_PLAY_STARTED, SYSTEM_DEAL_ROUND, SYSTEM_LANDLORD_RESOLVED, SYSTEM_REDEAL,
};

/// Ragged event streams, one range per environment slot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedEvents {
    /// Packed-buffer protocol version.
    pub schema_version: u32,
    /// Number of environment slots.
    pub batch_size: usize,
    /// Slot ranges. Slot `i` owns `[offsets[i], offsets[i + 1])`.
    pub offsets: Vec<u64>,
    /// Event sequence within the complete match history.
    pub sequence: Vec<u32>,
    /// Deal-attempt index.
    pub attempt: Vec<u32>,
    /// [`EVENT_PLAYER`] or [`EVENT_SYSTEM`].
    pub kind: Vec<u8>,
    /// Player actor, or [`NO_SEAT`] for system events.
    pub actor: Vec<i8>,
    /// Player action fields. System rows contain sentinels.
    pub actions: PackedActionData,
    /// System event code, or [`NO_EVENT_CODE`] for player events.
    pub system_kind: Vec<u8>,
    /// First normalized system payload.
    pub system_arg0: Vec<i64>,
    /// Second normalized system payload.
    pub system_arg1: Vec<i64>,
}

impl PackedEvents {
    #[must_use]
    pub fn empty(batch_size: usize) -> Self {
        Self {
            schema_version: BATCH_SCHEMA_VERSION,
            batch_size,
            offsets: vec![0; batch_size + 1],
            sequence: Vec::new(),
            attempt: Vec::new(),
            kind: Vec::new(),
            actor: Vec::new(),
            actions: PackedActionData::with_capacity(0),
            system_kind: Vec::new(),
            system_arg0: Vec::new(),
            system_arg1: Vec::new(),
        }
    }

    pub(crate) fn from_rows<'a>(
        rows: impl IntoIterator<Item = &'a [GameEvent]>,
    ) -> Result<Self, BatchError> {
        let rows = rows.into_iter().collect::<Vec<_>>();
        let total = rows.iter().try_fold(0_usize, |accumulator, row| {
            accumulator
                .checked_add(row.len())
                .ok_or(BatchError::BufferOverflow {
                    field: "event count",
                })
        })?;
        let mut result = Self {
            schema_version: BATCH_SCHEMA_VERSION,
            batch_size: rows.len(),
            offsets: Vec::with_capacity(rows.len() + 1),
            sequence: Vec::with_capacity(total),
            attempt: Vec::with_capacity(total),
            kind: Vec::with_capacity(total),
            actor: Vec::with_capacity(total),
            actions: PackedActionData::with_capacity(total),
            system_kind: Vec::with_capacity(total),
            system_arg0: Vec::with_capacity(total),
            system_arg1: Vec::with_capacity(total),
        };
        result.offsets.push(0);
        for row in rows {
            for event in row {
                result.push(*event);
            }
            result.offsets.push(
                u64::try_from(result.sequence.len()).map_err(|_| {
                    BatchError::BufferOverflow {
                        field: "event offsets",
                    }
                })?,
            );
        }
        result.validate()?;
        Ok(result)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }

    fn push(&mut self, event: GameEvent) {
        self.sequence.push(event.sequence);
        self.attempt.push(event.attempt);
        match event.kind {
            GameEventKind::Player(PlayerEvent { actor, action }) => {
                self.kind.push(EVENT_PLAYER);
                self.actor.push(super::codes::seat(actor));
                self.actions.push(action);
                self.system_kind.push(NO_EVENT_CODE);
                self.system_arg0.push(0);
                self.system_arg1.push(0);
            }
            GameEventKind::System(system) => {
                self.kind.push(EVENT_SYSTEM);
                self.actor.push(NO_SEAT);
                self.actions.push_none();
                let (kind, arg0, arg1) = system_payload(system);
                self.system_kind.push(kind);
                self.system_arg0.push(arg0);
                self.system_arg1.push(arg1);
            }
        }
    }

    pub fn validate(&self) -> Result<(), BatchError> {
        let expected = self.len();
        if self.offsets.len() != self.batch_size + 1
            || self.attempt.len() != expected
            || self.kind.len() != expected
            || self.actor.len() != expected
            || self.system_kind.len() != expected
            || self.system_arg0.len() != expected
            || self.system_arg1.len() != expected
        {
            return Err(BatchError::InternalInvariant(
                "packed event columns have inconsistent lengths",
            ));
        }
        if self.offsets.first().copied() != Some(0)
            || self.offsets.windows(2).any(|pair| pair[0] > pair[1])
            || self.offsets.last().copied()
                != Some(u64::try_from(expected).map_err(|_| BatchError::BufferOverflow {
                    field: "event final offset",
                })?)
        {
            return Err(BatchError::InternalInvariant(
                "packed event offsets are invalid",
            ));
        }
        self.actions.validate(expected)
    }
}

fn system_payload(event: SystemEvent) -> (u8, i64, i64) {
    match event {
        SystemEvent::DealRound { round } => (SYSTEM_DEAL_ROUND, i64::from(round), 0),
        SystemEvent::Redeal {
            from_attempt,
            to_attempt,
        } => (SYSTEM_REDEAL, i64::from(from_attempt), i64::from(to_attempt)),
        SystemEvent::LandlordResolved { landlord } => (SYSTEM_LANDLORD_RESOLVED, i64::from(landlord.value()), 0),
        SystemEvent::BottomRevealed => (SYSTEM_BOTTOM_REVEALED, 0, 0),
        SystemEvent::CardPlayStarted => (SYSTEM_CARD_PLAY_STARTED, 0, 0),
    }
}
