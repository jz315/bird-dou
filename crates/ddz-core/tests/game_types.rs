use ddz_core::{BidAction, DoubleAction, GameAction, GameEvent, Move, Phase, StepResult};

#[test]
fn game_actions_use_phase_first_stable_order() {
    let actions = [
        GameAction::Bid(BidAction::Pass),
        GameAction::Bid(BidAction::Score(1)),
        GameAction::Double(DoubleAction::Decline),
        GameAction::Double(DoubleAction::Double),
        GameAction::Play(Move::pass()),
    ];

    assert!(actions.windows(2).all(|pair| pair[0] < pair[1]));
}

#[test]
fn phase_actions_events_and_results_serialize_without_data_loss() {
    let event = GameEvent {
        sequence: 7,
        actor: 2,
        action: GameAction::Play(Move::pass()),
    };
    let result = StepResult {
        event,
        next_player: Some(0),
        terminal: false,
        raw_payoff: [0; 3],
        objective_payoff: [0; 3],
    };

    let yaml = serde_yaml_ng::to_string(&(Phase::CardPlay, result.clone())).unwrap();
    let decoded: (Phase, StepResult) = serde_yaml_ng::from_str(&yaml).unwrap();
    assert_eq!(decoded, (Phase::CardPlay, result));
    assert!(yaml.contains("card_play"));
    assert!(yaml.contains("play"));
}
