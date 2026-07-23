use ddz_core::{
    CallAction, GameAction, LandlordSelectionState, Phase, RobbingState, Seat, SeatOrder, SeatSet,
};

use super::super::{history, landlord, Game, GameError};

pub(crate) fn apply(game: &mut Game, actor: Seat, action: CallAction) -> Result<(), GameError> {
    if game.state.phase != Phase::Calling {
        return Err(GameError::WrongActionForPhase {
            phase: game.state.phase,
            action: GameAction::Call(action),
        });
    }
    let LandlordSelectionState::Calling(mut calling) = game.state.landlord_selection.clone() else {
        return Err(GameError::InvalidInternalState(
            "calling phase does not contain CallingState",
        ));
    };

    history::push_player(game, actor, GameAction::Call(action))?;
    calling.acted.insert(actor);
    match action {
        CallAction::Pass => {
            calling.declined.insert(actor);
            if calling.acted == SeatSet::all() {
                if game
                    .rules
                    .calling
                    .first_revealer_becomes_landlord_on_all_pass
                {
                    if let Some(first_revealer) = game.state.reveal.first_revealer {
                        return landlord::resolve(game, first_revealer, first_revealer, 0);
                    }
                }
                if game.rules.calling.redeal_on_all_pass {
                    return landlord::redeal(game);
                }
                return Err(GameError::InvalidInternalState(
                    "all players passed but this profile defines no fallback",
                ));
            }
            let next = history::next_cyclic_unacted(
                calling.first_player,
                calling.acted,
                SeatSet::all(),
            )
            .ok_or(GameError::InvalidInternalState(
                "calling has unacted seats but no next player",
            ))?;
            calling.current_player = next;
            game.state.current_player = Some(next);
            game.state.landlord_selection = LandlordSelectionState::Calling(calling);
            Ok(())
        }
        CallAction::CallLandlord => begin_robbing_or_resolve(game, actor, calling.declined),
    }
}

fn begin_robbing_or_resolve(
    game: &mut Game,
    caller: Seat,
    declined: SeatSet,
) -> Result<(), GameError> {
    if !game.rules.robbing.enabled {
        return landlord::resolve(game, caller, caller, 0);
    }
    let eligible = SeatSet::all().difference(declined);
    let order = SeatOrder::new((1_u8..=3).filter_map(|offset| {
        let seat = caller.offset(offset);
        if !eligible.contains(seat) {
            return None;
        }
        if seat == caller && !game.rules.robbing.caller_can_reclaim {
            return None;
        }
        Some(seat)
    }))
    .map_err(GameError::SeatOrder)?;
    if order.is_empty() {
        return landlord::resolve(game, caller, caller, 0);
    }
    game.state.landlord_selection = LandlordSelectionState::Robbing(RobbingState {
        caller,
        candidate: caller,
        order,
        cursor: 0,
        successful_robs: 0,
    });
    game.state.phase = Phase::Robbing;
    game.state.current_player = None;
    Ok(())
}
