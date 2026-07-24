use guandan_rules::{generate_legal_moves, Action, GameError, Round, Seat};

use crate::model::{kind_code, ActionView};

pub(crate) fn action_views(round: &Round) -> Result<Vec<ActionView>, GameError> {
    if round.outcome().is_some() {
        return Ok(Vec::new());
    }
    let mut actions = Vec::new();
    if round.target_move().is_some() {
        actions.push(ActionView {
            index: 0,
            kind: "pass",
            cards: Vec::new(),
            total_cards: 0,
        });
    }
    let moves = generate_legal_moves(
        round.hand(round.current_player()),
        round.target_move(),
        round.level(),
    )
    .map_err(GameError::InvalidMove)?;
    for movement in moves {
        actions.push(ActionView {
            index: actions.len(),
            kind: kind_code(*movement.kind()),
            cards: movement.cards().to_vec(),
            total_cards: movement.len(),
        });
    }
    Ok(actions)
}

pub(crate) fn apply_action_index(
    round: &mut Round,
    actor: Seat,
    index: usize,
) -> Result<guandan_rules::StepResult, GameError> {
    let selected = action_views(round)?
        .into_iter()
        .find(|action| action.index == index)
        .ok_or(GameError::DoesNotBeat)?;
    let action = if selected.kind == "pass" {
        Action::Pass
    } else {
        Action::Play(selected.cards)
    };
    round.step(actor, action)
}
