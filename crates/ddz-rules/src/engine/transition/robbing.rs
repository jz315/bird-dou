use ddz_core::{GameAction, LandlordSelectionState, Phase, RobAction, Seat};

use super::super::{history, Game, GameError};

pub(crate) fn apply(game: &mut Game, actor: Seat, action: RobAction) -> Result<(), GameError> {
    if game.state.phase != Phase::Robbing {
        return Err(GameError::WrongActionForPhase {
            phase: game.state.phase,
            action: GameAction::Rob(action),
        });
    }
    let LandlordSelectionState::Robbing(mut robbing) = game.state.landlord_selection.clone() else {
        return Err(GameError::InvalidInternalState(
            "robbing phase does not contain RobbingState",
        ));
    };
    history::push_player(game, actor, GameAction::Rob(action))?;
    if action == RobAction::RobLandlord {
        robbing.candidate = actor;
        robbing.successful_robs = robbing
            .successful_robs
            .checked_add(1)
            .ok_or(GameError::StakeExponentOverflow)?;
        game.state.stake.rob_exponent = robbing.successful_robs;
    }
    robbing.cursor = robbing
        .cursor
        .checked_add(1)
        .ok_or(GameError::InvalidInternalState(
            "robbing cursor overflowed",
        ))?;
    game.state.landlord_selection = LandlordSelectionState::Robbing(robbing);
    game.state.current_player = None;
    Ok(())
}
