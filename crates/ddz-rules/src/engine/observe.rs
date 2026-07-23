use ddz_core::{
    GameAction, GameEventKind, Observation, Phase, PlayerEvent, RankCounts, Seat, SeatMap,
};

use super::{Game, GameError};

pub(crate) fn observation(game: &Game, observer: Seat) -> Result<Observation, GameError> {
    observation_impl(game, observer, true)
}

pub(crate) fn observation_without_history(
    game: &Game,
    observer: Seat,
) -> Result<Observation, GameError> {
    observation_impl(game, observer, false)
}

fn observation_impl(
    game: &Game,
    observer: Seat,
    include_history: bool,
) -> Result<Observation, GameError> {
    let mut revealed_hands = SeatMap::new([None; 3]);
    for seat in Seat::ALL {
        if seat != observer && game.state.reveal.is_revealed(seat) {
            revealed_hands[seat] = Some(game.state.hands[seat]);
        }
    }

    let observation = Observation {
        phase: game.state.phase,
        observer,
        role: game.state.role_of(observer),
        current_player: game.state.current_player,
        landlord: game.state.landlord(),
        own_hand: game.state.hands[observer],
        revealed_hands,
        unknown_pool: unknown_pool(game, observer, revealed_hands)?,
        cards_left: game.state.cards_left(),
        public_bottom_cards: public_bottom_cards(game),
        reveal: game.state.reveal.clone(),
        landlord_selection: game.state.landlord_selection.clone(),
        doubling: Observation::public_doubling_from_private(&game.state.doubling),
        stake: game.state.stake,
        card_play: game.state.card_play.clone(),
        history: if include_history {
            public_history(game)
        } else {
            Vec::new()
        },
        outcome: game.state.outcome.clone(),
    };
    observation.validate().map_err(GameError::Observation)?;
    Ok(observation)
}

fn unknown_pool(
    game: &Game,
    observer: Seat,
    revealed_hands: SeatMap<Option<RankCounts>>,
) -> Result<RankCounts, GameError> {
    let mut unknown = RankCounts::empty();
    for seat in Seat::ALL {
        if seat != observer && revealed_hands[seat].is_none() {
            unknown = unknown
                .checked_add(game.state.hands[seat])
                .map_err(GameError::RankCounts)?;
        }
    }
    if game.state.landlord().is_none() {
        unknown = unknown
            .checked_add(game.state.deal.plan.bottom_counts())
            .map_err(GameError::RankCounts)?;
    }
    if matches!(game.state.phase, Phase::PreDeal | Phase::Dealing) {
        let first_undealt = usize::from(game.state.deal.rounds_dealt) * 3;
        for index in first_undealt..51 {
            let card = game
                .state
                .deal
                .plan
                .deck()
                .card(index)
                .ok_or(GameError::InvalidInternalState(
                    "validated deal plan is missing an undealt card",
                ))?;
            unknown.add_card(card).map_err(GameError::RankCounts)?;
        }
    }
    Ok(unknown)
}

fn public_bottom_cards(game: &Game) -> Option<RankCounts> {
    (game.rules.bottom_cards_public && game.state.landlord().is_some())
        .then(|| game.state.deal.plan.bottom_counts())
}

fn public_history(game: &Game) -> Vec<ddz_core::GameEvent> {
    game.state
        .history
        .iter()
        .filter(|event| {
            if game.state.phase != Phase::Doubling || event.attempt != game.state.deal.attempt {
                return true;
            }
            !matches!(
                event.kind,
                GameEventKind::Player(PlayerEvent {
                    action: GameAction::Double(_),
                    ..
                })
            )
        })
        .copied()
        .collect()
}
