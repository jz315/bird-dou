use ddz_core::{GameEvent, Observation, Seat};

use crate::protocol::observation::ObservationRow;
use crate::protocol::{PackedEvents, PackedObservation};
use crate::BatchError;

use super::{BatchEnv, Slot};

impl BatchEnv {
    /// Pack one observation for every active current player.
    ///
    /// Terminal rows contain public state metadata but set `status.valid = 0` because no arbitrary
    /// player perspective is silently chosen.
    pub fn observations_current(&self) -> Result<PackedObservation, BatchError> {
        self.require_initialized()?;
        pack_current(&self.slots)
    }

    /// Pack one explicit observer perspective per slot, including terminal slots.
    pub fn observations_for(&self, observers: &[Seat]) -> Result<PackedObservation, BatchError> {
        self.require_initialized()?;
        if observers.len() != self.slots.len() {
            return Err(BatchError::BatchSizeMismatch {
                field: "observers",
                expected: self.slots.len(),
                actual: observers.len(),
            });
        }
        let observations = self
            .slots
            .iter()
            .zip(observers.iter().copied())
            .enumerate()
            .map(|(slot, (value, observer))| {
                value
                    .game
                    .observe_without_history(observer)
                    .map_err(|source| BatchError::Observe { slot, source })
            })
            .collect::<Result<Vec<_>, _>>()?;
        pack_with_observations(&self.slots, &observations)
    }

    /// Pack the complete information-safe public history for each current player.
    ///
    /// Use this for reset/restore resynchronization. In the normal hot path, keep history outside
    /// Rust and append public deltas derived from player observations rather than repacking all
    /// previous events every step.
    pub fn public_history_packed(&self) -> Result<PackedEvents, BatchError> {
        self.require_initialized()?;
        let histories = self
            .slots
            .iter()
            .enumerate()
            .map(|(slot, value)| match value.game.state().current_player {
                Some(observer) => value
                    .game
                    .observe(observer)
                    .map(|observation| observation.history)
                    .map_err(|source| BatchError::Observe { slot, source }),
                None => Ok(value.game.state().history.clone()),
            })
            .collect::<Result<Vec<Vec<GameEvent>>, _>>()?;
        PackedEvents::from_rows(histories.iter().map(Vec::as_slice))
    }

    /// Pack authoritative engine history, including temporarily hidden in-progress double choices.
    /// This is for replay/debugging, not model input.
    pub fn authoritative_history_packed(&self) -> Result<PackedEvents, BatchError> {
        self.require_initialized()?;
        PackedEvents::from_rows(
            self.slots
                .iter()
                .map(|slot| slot.game.state().history.as_slice()),
        )
    }
}

pub(crate) fn pack_current(slots: &[Slot]) -> Result<PackedObservation, BatchError> {
    let observations = slots
        .iter()
        .enumerate()
        .map(|(slot, value)| match value.game.state().current_player {
            Some(observer) => value
                .game
                .observe_without_history(observer)
                .map(Some)
                .map_err(|source| BatchError::Observe { slot, source }),
            None => Ok(None),
        })
        .collect::<Result<Vec<Option<Observation>>, _>>()?;
    let rows = slots
        .iter()
        .zip(observations.iter())
        .map(|(slot, observation)| ObservationRow {
            state: slot.game.state(),
            version: slot.version,
            observation: observation.as_ref(),
        })
        .collect::<Vec<_>>();
    PackedObservation::from_rows(&rows)
}

fn pack_with_observations(
    slots: &[Slot],
    observations: &[Observation],
) -> Result<PackedObservation, BatchError> {
    let rows = slots
        .iter()
        .zip(observations.iter())
        .map(|(slot, observation)| ObservationRow {
            state: slot.game.state(),
            version: slot.version,
            observation: Some(observation),
        })
        .collect::<Vec<_>>();
    PackedObservation::from_rows(&rows)
}
