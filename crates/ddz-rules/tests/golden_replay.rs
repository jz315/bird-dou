use ddz_core::{cards_to_rank_counts, GameAction, RankCounts};
use ddz_rules::{detect_move_with_rules, PostBidGame, RuleConfig};
use serde::Deserialize;

const DOUZERO_POST_BID_YAML: &str = include_str!("../../../configs/rules/douzero_post_bid.yaml");
const GOLDEN_REPLAY_JSON: &str =
    include_str!("../../../tests/golden_replays/post_bid_five_bombs.json");

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GoldenReplay {
    schema_version: u32,
    name: String,
    landlord: u8,
    hands_card_ids: [Vec<u8>; 3],
    bottom_card_ids: Vec<u8>,
    actions: Vec<RankCounts>,
    expected: GoldenExpected,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GoldenExpected {
    winner: u8,
    bomb_count: u8,
    raw_payoff: [i32; 3],
    objective_payoff: [i32; 3],
    history_len: usize,
    serialized_state_fnv1a64: u64,
}

#[test]
fn checked_in_replay_has_a_stable_outcome_and_state_encoding() {
    let fixture: GoldenReplay = serde_json::from_str(GOLDEN_REPLAY_JSON).unwrap();
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.name, "post_bid_landlord_five_bombs");
    let rules = RuleConfig::from_yaml_str(DOUZERO_POST_BID_YAML).unwrap();
    let hands = fixture
        .hands_card_ids
        .each_ref()
        .map(|cards| cards_to_rank_counts(cards).unwrap());
    let bottom_cards = cards_to_rank_counts(&fixture.bottom_card_ids).unwrap();
    let mut game = PostBidGame::new(hands, bottom_cards, fixture.landlord, rules).unwrap();
    let mut terminal_result = None;

    for cards in fixture.actions {
        let played = detect_move_with_rules(cards, &rules).unwrap();
        let result = game.step(&GameAction::Play(played)).unwrap();
        if result.terminal {
            terminal_result = Some(result);
        }
    }

    let result = terminal_result.expect("golden replay must terminate");
    assert!(game.is_terminal());
    assert_eq!(game.state().current_player, fixture.expected.winner);
    assert_eq!(game.state().bomb_count, fixture.expected.bomb_count);
    assert_eq!(game.state().raw_payoff, fixture.expected.raw_payoff);
    assert_eq!(result.objective_payoff, fixture.expected.objective_payoff);
    assert_eq!(game.state().history.len(), fixture.expected.history_len);

    let bytes = game.serialize_state().unwrap();
    assert_eq!(fnv1a64(&bytes), fixture.expected.serialized_state_fnv1a64);
    assert_eq!(PostBidGame::deserialize_state(&bytes, rules).unwrap(), game);
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}
