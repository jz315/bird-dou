use ddz_core::{DoubleAction, DoublingState, GameAction, Phase, Seat};

use super::super::{history, Game, GameError};

pub(crate) fn apply(
    game: &mut Game,
    actor: Seat,
    action: DoubleAction,
) -> Result<(), GameError> {
    if game.state.phase != Phase::Doubling {
        return Err(GameError::WrongActionForPhase {
            phase: game.state.phase,
            action: GameAction::Double(action),
        });
    }
    let DoublingState::InProgress(mut round) = game.state.doubling.clone() else {
        return Err(GameError::InvalidInternalState(
            "doubling phase does not contain an in-progress round",
        ));
    };
    history::push_player(game, actor, GameAction::Double(action))?;
    if action == DoubleAction::Double {
        round.doubled.insert(actor);
    }
    round.cursor = round
        .cursor
        .checked_add(1)
        .ok_or(GameError::InvalidInternalState(
            "doubling cursor overflowed",
        ))?;
    game.state.doubling = DoublingState::InProgress(round);
    game.state.current_player = None;
    Ok(())
}
