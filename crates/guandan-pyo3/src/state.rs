use guandan_rules::{Round, Seat};

use crate::legal::action_views;
use crate::model::{EventView, ResultView, StateView, TargetView};

pub(crate) fn build_state(
    round: &Round,
    human_seat: Seat,
    observer: Seat,
    history: &[EventView],
) -> Result<StateView, guandan_rules::GameError> {
    let outcome = round.outcome();
    let human_turn = outcome.is_none() && round.current_player() == observer;
    let legal_actions = if human_turn {
        action_views(round)?
    } else {
        Vec::new()
    };
    let result = outcome.map(|value| ResultView {
        finish_order: *value.finish_order(),
        winning_team: value.winning_team(),
        level_advance: value.level_advance(),
        human_won: value.winning_team() == human_seat.team(),
    });
    Ok(StateView {
        schema_version: 1,
        phase: if outcome.is_some() {
            "terminal"
        } else {
            "card_play"
        },
        human_seat,
        human_turn,
        current_player: outcome.is_none().then_some(round.current_player()),
        level: round.level(),
        hand: round.hand(observer).cards().collect(),
        cards_left: std::array::from_fn(|index| round.hand(Seat::ALL[index]).len()),
        target: round
            .target_move()
            .zip(round.target_player())
            .map(|(movement, actor)| TargetView::from_move(actor, movement)),
        legal_actions,
        recent_actions: history.iter().rev().take(16).cloned().rev().collect(),
        finish_order: round.finish_order().to_vec(),
        result,
    })
}
