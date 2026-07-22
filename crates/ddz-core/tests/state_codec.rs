use ddz_core::{
    deserialize_game_state, serialize_game_state, BidState, GameState, Phase, SpringState,
    StateCodecError, EMPTY_RANK_COUNTS, GAME_STATE_SCHEMA_VERSION,
};

fn sample_state() -> GameState {
    GameState {
        rule_config_id: 1,
        phase: Phase::CardPlay,
        current_player: 0,
        landlord: Some(0),
        hands: [EMPTY_RANK_COUNTS; 3],
        bottom_cards: EMPTY_RANK_COUNTS,
        played_cards: [EMPTY_RANK_COUNTS; 3],
        cards_left: [0; 3],
        last_non_pass: None,
        last_non_pass_player: None,
        consecutive_passes: 0,
        bid_state: BidState::DisabledPostBid,
        multiplier_exp: 0,
        bomb_count: 0,
        spring_state: SpringState::default(),
        history: Vec::new(),
        terminal: false,
        raw_payoff: [0; 3],
    }
}

#[test]
fn state_envelope_round_trips_deterministically() {
    let state = sample_state();
    let first = serialize_game_state(&state).unwrap();
    let second = serialize_game_state(&state).unwrap();

    assert_eq!(first, second);
    assert_eq!(deserialize_game_state(&first).unwrap(), state);
    assert!(String::from_utf8(first).unwrap().starts_with(&format!(
        "{{\"schema_version\":{GAME_STATE_SCHEMA_VERSION},\"state\":"
    )));
}

#[test]
fn malformed_unknown_and_future_envelopes_are_rejected() {
    assert!(matches!(
        deserialize_game_state(b"not json"),
        Err(StateCodecError::Json(_))
    ));
    assert!(matches!(
        deserialize_game_state(b"{\"schema_version\":1,\"state\":{},\"extra\":0}"),
        Err(StateCodecError::Json(_))
    ));

    let mut value: serde_json::Value =
        serde_json::from_slice(&serialize_game_state(&sample_state()).unwrap()).unwrap();
    value["schema_version"] = serde_json::json!(GAME_STATE_SCHEMA_VERSION + 1);
    let future = serde_json::to_vec(&value).unwrap();
    assert!(matches!(
        deserialize_game_state(&future),
        Err(StateCodecError::UnsupportedSchemaVersion {
            expected: GAME_STATE_SCHEMA_VERSION,
            actual,
        }) if actual == GAME_STATE_SCHEMA_VERSION + 1
    ));
}
