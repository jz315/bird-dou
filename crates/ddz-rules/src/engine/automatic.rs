use ddz_core::{
    CallingState, DoublingRound, DoublingState, LandlordSelectionState, Phase, Seat, SeatOrder,
    SeatSet, SystemEvent, DEAL_ROUNDS,
};

use super::{history, landlord, Game, GameError};

const AUTOMATIC_TRANSITION_LIMIT: usize = 128;

pub(crate) fn advance(game: &mut Game) -> Result<(), GameError> {
    for _ in 0..AUTOMATIC_TRANSITION_LIMIT {
        if game.state.is_terminal() {
            return Ok(());
        }
        match game.state.phase {
            Phase::PreDeal => {
                if game.rules.reveal.before_deal_enabled && game.state.current_player.is_some() {
                    return Ok(());
                }
                game.state.current_player = None;
                game.state.phase = Phase::Dealing;
            }
            Phase::Dealing => {
                if game.state.current_player.is_some() {
                    return Ok(());
                }
                if game.state.deal.rounds_dealt == DEAL_ROUNDS {
                    begin_calling(game)?;
                    continue;
                }
                deal_one_round(game)?;
                let next = next_dealing_reveal_player(game);
                game.state.current_player = next;
                if next.is_some() {
                    return Ok(());
                }
            }
            Phase::Calling => {
                if game.state.current_player.is_some() {
                    return Ok(());
                }
                return Err(GameError::InvalidInternalState(
                    "calling phase has no current player",
                ));
            }
            Phase::Robbing => {
                normalize_robbing(game)?;
                if game.state.phase != Phase::Robbing {
                    continue;
                }
                if game.state.current_player.is_some() {
                    return Ok(());
                }
            }
            Phase::BottomReveal => {
                history::push_system(game, SystemEvent::BottomRevealed)?;
                game.state.phase = Phase::PostBottomReveal;
                let landlord = game
                    .state
                    .landlord()
                    .ok_or(GameError::InvalidInternalState(
                        "bottom reveal requires a resolved landlord",
                    ))?;
                game.state.current_player = (game.rules.reveal.after_bottom_enabled
                    && !game.state.reveal.is_revealed(landlord))
                .then_some(landlord);
            }
            Phase::PostBottomReveal => {
                if game.state.current_player.is_some() {
                    return Ok(());
                }
                begin_doubling_or_card_play(game)?;
            }
            Phase::Doubling => {
                normalize_doubling(game)?;
                if game.state.phase != Phase::Doubling {
                    continue;
                }
                if game.state.current_player.is_some() {
                    return Ok(());
                }
            }
            Phase::CardPlay | Phase::Terminal => return Ok(()),
        }
    }
    Err(GameError::AutomaticTransitionLimit)
}

fn deal_one_round(game: &mut Game) -> Result<(), GameError> {
    let round_index = game.state.deal.rounds_dealt;
    for seat in Seat::ALL {
        let card = game
            .state
            .deal
            .plan
            .card_for(seat, round_index)
            .ok_or(GameError::InvalidInternalState(
                "deal plan did not contain the requested round",
            ))?;
        game.state.hands[seat]
            .add_card(card)
            .map_err(GameError::RankCounts)?;
    }
    game.state.deal.rounds_dealt = game
        .state
        .deal
        .rounds_dealt
        .checked_add(1)
        .ok_or(GameError::InvalidInternalState(
            "deal-round counter overflowed",
        ))?;
    history::push_system(
        game,
        SystemEvent::DealRound {
            round: game.state.deal.rounds_dealt,
        },
    )?;
    Ok(())
}

fn next_dealing_reveal_player(game: &Game) -> Option<Seat> {
    if game
        .rules
        .reveal
        .factor_during_deal(game.state.deal.rounds_dealt)
        .is_none()
    {
        return None;
    }
    let first = initial_first_player(game);
    let acted = history::dealing_acted(&game.state);
    let eligible = SeatSet::all().difference(history::revealed_seats(&game.state));
    history::next_cyclic_unacted(first, acted, eligible)
}

fn begin_calling(game: &mut Game) -> Result<(), GameError> {
    if !game.rules.calling.enabled {
        return Err(GameError::InvalidInternalState(
            "complete deal reached calling with calling disabled",
        ));
    }
    let first_player = game
        .state
        .reveal
        .first_revealer
        .unwrap_or_else(|| initial_first_player(game));
    game.state.landlord_selection = LandlordSelectionState::Calling(CallingState {
        first_player,
        current_player: first_player,
        acted: SeatSet::empty(),
        declined: SeatSet::empty(),
    });
    game.state.phase = Phase::Calling;
    game.state.current_player = Some(first_player);
    Ok(())
}

fn normalize_robbing(game: &mut Game) -> Result<(), GameError> {
    let LandlordSelectionState::Robbing(mut robbing) = game.state.landlord_selection.clone() else {
        return Err(GameError::InvalidInternalState(
            "robbing phase does not contain a robbing state",
        ));
    };
    loop {
        let Some(seat) = robbing.current_player() else {
            landlord::resolve(
                game,
                robbing.candidate,
                robbing.caller,
                robbing.successful_robs,
            )?;
            return Ok(());
        };
        if seat == robbing.candidate {
            robbing.cursor = robbing
                .cursor
                .checked_add(1)
                .ok_or(GameError::InvalidInternalState(
                    "robbing cursor overflowed",
                ))?;
            continue;
        }
        game.state.current_player = Some(seat);
        game.state.landlord_selection = LandlordSelectionState::Robbing(robbing);
        return Ok(());
    }
}

fn begin_doubling_or_card_play(game: &mut Game) -> Result<(), GameError> {
    if !game.rules.doubling.enabled {
        game.state.doubling = DoublingState::Disabled;
        return begin_card_play(game);
    }
    let landlord = game
        .state
        .landlord()
        .ok_or(GameError::InvalidInternalState(
            "doubling requires a resolved landlord",
        ))?;
    let order = SeatOrder::new(
        (0_u8..3)
            .map(|offset| landlord.offset(offset))
            .filter(|seat| {
                game.economy
                    .double_eligible(*seat, landlord, game.rules.doubling)
            }),
    )
    .map_err(GameError::SeatOrder)?;
    if order.is_empty() {
        game.state.doubling = DoublingState::Resolved {
            eligible: SeatSet::empty(),
            doubled: SeatSet::empty(),
        };
        begin_card_play(game)
    } else {
        let current_player = order.get(0);
        game.state.doubling = DoublingState::InProgress(DoublingRound {
            order,
            cursor: 0,
            doubled: SeatSet::empty(),
        });
        game.state.phase = Phase::Doubling;
        game.state.current_player = current_player;
        Ok(())
    }
}

fn normalize_doubling(game: &mut Game) -> Result<(), GameError> {
    let DoublingState::InProgress(round) = game.state.doubling.clone() else {
        return Err(GameError::InvalidInternalState(
            "doubling phase does not contain an in-progress round",
        ));
    };
    if let Some(current) = round.current_player() {
        game.state.current_player = Some(current);
        return Ok(());
    }
    game.state.doubling = DoublingState::Resolved {
        eligible: round.eligible(),
        doubled: round.doubled,
    };
    begin_card_play(game)
}

fn begin_card_play(game: &mut Game) -> Result<(), GameError> {
    let landlord = game
        .state
        .landlord()
        .ok_or(GameError::InvalidInternalState(
            "card play requires a resolved landlord",
        ))?;
    game.state.phase = Phase::CardPlay;
    game.state.current_player = Some(landlord);
    history::push_system(game, SystemEvent::CardPlayStarted)?;
    Ok(())
}

fn initial_first_player(game: &Game) -> Seat {
    match &game.state.landlord_selection {
        LandlordSelectionState::NotStarted { first_player } => *first_player,
        LandlordSelectionState::Calling(state) => state.first_player,
        _ => crate::first_player_for_attempt(game.match_seed, game.state.deal.attempt),
    }
}
