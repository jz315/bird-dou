use ddz_core::Seat;

use super::{Game, GameError};

pub(crate) fn finish(game: &mut Game, winner: Seat) -> Result<(), GameError> {
    crate::settle_game(&mut game.state, winner, &game.rules).map_err(GameError::Settlement)
}
