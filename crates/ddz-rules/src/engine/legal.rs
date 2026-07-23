use ddz_core::{CallAction, DoubleAction, GameAction, Phase, RevealAction, RobAction};

use super::{Game, GameError};
use crate::{generate_follow_moves, generate_lead_moves};

pub(crate) fn legal_actions(game: &Game) -> Result<Vec<GameAction>, GameError> {
    if game.state.is_terminal() {
        return Ok(Vec::new());
    }
    let actor = game
        .state
        .current_player
        .ok_or(GameError::NoCurrentPlayer {
            phase: game.state.phase,
        })?;

    let mut actions = match game.state.phase {
        Phase::PreDeal => reveal_actions(
            game.rules.reveal.before_deal_enabled,
            !game.state.reveal.is_revealed(actor),
        ),
        Phase::Dealing => reveal_actions(
            game.rules
                .reveal
                .factor_during_deal(game.state.deal.cards_received(actor))
                .is_some(),
            !game.state.reveal.is_revealed(actor),
        ),
        Phase::Calling => vec![
            GameAction::Call(CallAction::Pass),
            GameAction::Call(CallAction::CallLandlord),
        ],
        Phase::Robbing => vec![
            GameAction::Rob(RobAction::Pass),
            GameAction::Rob(RobAction::RobLandlord),
        ],
        Phase::PostBottomReveal => reveal_actions(
            game.rules.reveal.after_bottom_enabled,
            !game.state.reveal.is_revealed(actor),
        ),
        Phase::Doubling => vec![
            GameAction::Double(DoubleAction::Decline),
            GameAction::Double(DoubleAction::Double),
        ],
        Phase::CardPlay => card_play_actions(game, actor)?,
        Phase::BottomReveal | Phase::Terminal => {
            return Err(GameError::AutomaticPhase {
                phase: game.state.phase,
            });
        }
    };
    actions.sort_unstable();
    actions.dedup();
    Ok(actions)
}

fn reveal_actions(reveal_enabled: bool, not_yet_revealed: bool) -> Vec<GameAction> {
    let mut actions = vec![GameAction::Reveal(RevealAction::Continue)];
    if reveal_enabled && not_yet_revealed {
        actions.push(GameAction::Reveal(RevealAction::Reveal));
    }
    actions
}

fn card_play_actions(game: &Game, actor: ddz_core::Seat) -> Result<Vec<GameAction>, GameError> {
    let hand = game.state.hands[actor];
    let movements = match game.state.card_play.last_non_pass {
        Some(target) => generate_follow_moves(hand, target, &game.rules),
        None => generate_lead_moves(hand, &game.rules),
    }
    .map_err(GameError::GenerateMoves)?;
    Ok(movements.into_iter().map(GameAction::Play).collect())
}
