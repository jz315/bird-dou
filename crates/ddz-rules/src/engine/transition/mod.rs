mod calling;
mod card_play;
mod doubling;
mod reveal;
mod robbing;

use ddz_core::{GameAction, Seat};

use super::{Game, GameError};

pub(crate) fn apply(game: &mut Game, actor: Seat, action: GameAction) -> Result<(), GameError> {
    match action {
        GameAction::Reveal(action) => reveal::apply(game, actor, action),
        GameAction::Call(action) => calling::apply(game, actor, action),
        GameAction::Rob(action) => robbing::apply(game, actor, action),
        GameAction::Double(action) => doubling::apply(game, actor, action),
        GameAction::Play(movement) => card_play::apply(game, actor, movement),
    }
}
