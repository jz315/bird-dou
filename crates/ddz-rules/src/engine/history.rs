use ddz_core::{
    GameAction, GameEvent, GameEventKind, GameState, PlayerEvent, Seat, SeatSet, SystemEvent,
};

use super::{Game, GameError};

pub(crate) fn next_sequence(state: &GameState) -> Result<u32, GameError> {
    u32::try_from(state.history.len()).map_err(|_| GameError::HistoryTooLong)
}

pub(crate) fn push_player(
    game: &mut Game,
    actor: Seat,
    action: GameAction,
) -> Result<u32, GameError> {
    let sequence = next_sequence(&game.state)?;
    game.state.history.push(GameEvent {
        sequence,
        attempt: game.state.deal.attempt,
        kind: GameEventKind::Player(PlayerEvent { actor, action }),
    });
    Ok(sequence)
}

pub(crate) fn push_system(game: &mut Game, event: SystemEvent) -> Result<u32, GameError> {
    let sequence = next_sequence(&game.state)?;
    game.state.history.push(GameEvent {
        sequence,
        attempt: game.state.deal.attempt,
        kind: GameEventKind::System(event),
    });
    Ok(sequence)
}

pub(crate) fn predeal_acted(state: &GameState) -> SeatSet {
    let mut acted = SeatSet::empty();
    for event in state
        .history
        .iter()
        .filter(|event| event.attempt == state.deal.attempt)
    {
        match event.kind {
            GameEventKind::System(SystemEvent::DealRound { .. }) => break,
            GameEventKind::Player(PlayerEvent {
                actor,
                action: GameAction::Reveal(_),
            }) => {
                acted.insert(actor);
            }
            GameEventKind::Player(_) | GameEventKind::System(_) => {}
        }
    }
    acted
}

pub(crate) fn dealing_acted(state: &GameState) -> SeatSet {
    let mut acted = SeatSet::empty();
    let Some(round_start) = state.history.iter().rposition(|event| {
        event.attempt == state.deal.attempt
            && matches!(
                event.kind,
                GameEventKind::System(SystemEvent::DealRound { round })
                    if round == state.deal.rounds_dealt
            )
    }) else {
        return acted;
    };

    for event in &state.history[round_start + 1..] {
        if event.attempt != state.deal.attempt {
            continue;
        }
        if let GameEventKind::Player(PlayerEvent {
            actor,
            action: GameAction::Reveal(_),
        }) = event.kind
        {
            acted.insert(actor);
        }
    }
    acted
}

pub(crate) fn next_cyclic_unacted(
    first: Seat,
    acted: SeatSet,
    eligible: SeatSet,
) -> Option<Seat> {
    (0_u8..3)
        .map(|offset| first.offset(offset))
        .find(|seat| eligible.contains(*seat) && !acted.contains(*seat))
}

pub(crate) fn revealed_seats(state: &GameState) -> SeatSet {
    let mut revealed = SeatSet::empty();
    for seat in Seat::ALL {
        if state.reveal.is_revealed(seat) {
            revealed.insert(seat);
        }
    }
    revealed
}
