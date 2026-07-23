use ddz_core::{
    CardPlayState, DealState, DoublingState, LandlordSelectionState, Phase, RankCounts,
    ResolvedLandlord, RevealState, Seat, SeatMap, StakeState, SystemEvent,
};

use super::{history, Game, GameError};
use crate::{deal_plan_for_attempt, first_player_for_attempt};

pub(crate) fn resolve(
    game: &mut Game,
    landlord: Seat,
    caller: Seat,
    successful_robs: u8,
) -> Result<(), GameError> {
    let bottom = game.state.deal.plan.bottom_counts();
    game.state.hands[landlord] = game.state.hands[landlord]
        .checked_add(bottom)
        .map_err(GameError::RankCounts)?;
    game.state.landlord_selection = LandlordSelectionState::Resolved(ResolvedLandlord {
        landlord,
        caller,
        successful_robs,
    });
    game.state.stake.rob_exponent = successful_robs;
    game.state.phase = Phase::BottomReveal;
    game.state.current_player = None;
    history::push_system(game, SystemEvent::LandlordResolved { landlord })?;
    Ok(())
}

pub(crate) fn redeal(game: &mut Game) -> Result<(), GameError> {
    let from_attempt = game.state.deal.attempt;
    let to_attempt = from_attempt
        .checked_add(1)
        .ok_or(GameError::AttemptOverflow)?;
    history::push_system(
        game,
        SystemEvent::Redeal {
            from_attempt,
            to_attempt,
        },
    )?;

    let first_player = first_player_for_attempt(game.match_seed, to_attempt);
    game.state.phase = Phase::PreDeal;
    game.state.current_player = Some(first_player);
    game.state.deal = DealState::new(
        to_attempt,
        deal_plan_for_attempt(game.match_seed, to_attempt).map_err(GameError::Deal)?,
    );
    game.state.hands = SeatMap::new([RankCounts::empty(); 3]);
    game.state.reveal = RevealState::hidden();
    game.state.landlord_selection = LandlordSelectionState::NotStarted { first_player };
    game.state.doubling = if game.rules.doubling.enabled {
        DoublingState::NotStarted
    } else {
        DoublingState::Disabled
    };
    game.state.stake = StakeState::new(game.rules.settlement.base_unit);
    game.state.card_play = CardPlayState::empty();
    game.state.outcome = None;
    Ok(())
}
