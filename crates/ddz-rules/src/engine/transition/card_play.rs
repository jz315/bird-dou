use ddz_core::{GameAction, Move, MoveKind, Phase, Seat};

use super::super::{history, settle, Game, GameError};

pub(crate) fn apply(game: &mut Game, actor: Seat, movement: Move) -> Result<(), GameError> {
    if game.state.phase != Phase::CardPlay {
        return Err(GameError::WrongActionForPhase {
            phase: game.state.phase,
            action: GameAction::Play(movement),
        });
    }
    history::push_player(game, actor, GameAction::Play(movement))?;
    if movement.is_pass() {
        return apply_pass(game, actor);
    }

    game.state.hands[actor] = game.state.hands[actor]
        .checked_sub(movement.cards())
        .map_err(GameError::RankCounts)?;
    game.state.card_play.played_cards[actor] = game.state.card_play.played_cards[actor]
        .checked_add(movement.cards())
        .map_err(GameError::RankCounts)?;
    game.state.card_play.non_pass_plays[actor] = game.state.card_play.non_pass_plays[actor]
        .checked_add(1)
        .ok_or(GameError::InvalidInternalState(
            "non-pass play counter overflowed",
        ))?;
    if matches!(movement.kind(), MoveKind::Bomb | MoveKind::Rocket) {
        game.state.card_play.bomb_count = game
            .state
            .card_play
            .bomb_count
            .checked_add(1)
            .ok_or(GameError::InvalidInternalState(
                "bomb counter overflowed",
            ))?;
        game.state.stake.bomb_exponent = game
            .state
            .stake
            .bomb_exponent
            .checked_add(1)
            .ok_or(GameError::StakeExponentOverflow)?;
    }
    game.state.card_play.last_non_pass = Some(movement);
    game.state.card_play.last_non_pass_player = Some(actor);
    game.state.card_play.consecutive_passes = 0;
    if game.state.hands[actor].is_empty() {
        settle::finish(game, actor)
    } else {
        game.state.current_player = Some(actor.next());
        Ok(())
    }
}

fn apply_pass(game: &mut Game, actor: Seat) -> Result<(), GameError> {
    let target_player = game
        .state
        .card_play
        .last_non_pass_player
        .ok_or(GameError::InvalidInternalState(
            "pass was accepted without an active target",
        ))?;
    match game.state.card_play.consecutive_passes {
        0 => {
            game.state.card_play.consecutive_passes = 1;
            game.state.current_player = Some(actor.next());
        }
        1 => {
            game.state.card_play.consecutive_passes = 0;
            game.state.card_play.last_non_pass = None;
            game.state.card_play.last_non_pass_player = None;
            game.state.current_player = Some(target_player);
        }
        _ => {
            return Err(GameError::InvalidInternalState(
                "card-play state retained two passes without clearing the trick",
            ));
        }
    }
    Ok(())
}
