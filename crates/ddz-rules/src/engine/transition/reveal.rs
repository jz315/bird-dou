use ddz_core::{
    GameAction, LandlordSelectionState, Phase, RevealAction, RevealInfo, RevealTiming, Seat,
    SeatSet,
};

use super::super::{history, Game, GameError};

pub(crate) fn apply(
    game: &mut Game,
    actor: Seat,
    action: RevealAction,
) -> Result<(), GameError> {
    let (timing, factor) = match game.state.phase {
        Phase::PreDeal => (RevealTiming::BeforeDeal, game.rules.reveal.before_deal_factor),
        Phase::Dealing => {
            let cards_received = game.state.deal.cards_received(actor);
            let factor = game
                .rules
                .reveal
                .factor_during_deal(cards_received)
                .unwrap_or(1);
            (RevealTiming::DuringDeal { cards_received }, factor)
        }
        Phase::PostBottomReveal => (
            RevealTiming::AfterBottom,
            game.rules.reveal.after_bottom_factor,
        ),
        phase => {
            return Err(GameError::WrongActionForPhase {
                phase,
                action: GameAction::Reveal(action),
            });
        }
    };

    let sequence = history::push_player(game, actor, GameAction::Reveal(action))?;
    if action == RevealAction::Reveal {
        if game.state.reveal.is_revealed(actor) {
            return Err(GameError::InvalidInternalState(
                "a revealed player attempted to reveal again",
            ));
        }
        if factor < 2 {
            return Err(GameError::InvalidInternalState(
                "reveal action was accepted at a disabled reveal timing",
            ));
        }
        game.state.reveal.by_seat[actor] = Some(RevealInfo {
            timing,
            factor,
            sequence,
        });
        if game.state.reveal.first_revealer.is_none() {
            game.state.reveal.first_revealer = Some(actor);
        }
        game.state.reveal.maximum_factor = game.state.reveal.maximum_factor.max(factor);
        game.state.stake.reveal_factor = game.state.reveal.maximum_factor;
    }

    match game.state.phase {
        Phase::PreDeal => {
            let first = initial_first_player(game)?;
            game.state.current_player = history::next_cyclic_unacted(
                first,
                history::predeal_acted(&game.state),
                SeatSet::all(),
            );
        }
        Phase::Dealing => {
            let first = initial_first_player(game)?;
            let eligible = SeatSet::all().difference(history::revealed_seats(&game.state));
            game.state.current_player = history::next_cyclic_unacted(
                first,
                history::dealing_acted(&game.state),
                eligible,
            );
        }
        Phase::PostBottomReveal => game.state.current_player = None,
        _ => unreachable!("phase checked above"),
    }
    Ok(())
}

fn initial_first_player(game: &Game) -> Result<Seat, GameError> {
    match &game.state.landlord_selection {
        LandlordSelectionState::NotStarted { first_player } => Ok(*first_player),
        _ => Err(GameError::InvalidInternalState(
            "reveal-before-calling requires NotStarted landlord state",
        )),
    }
}
