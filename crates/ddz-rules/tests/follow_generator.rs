use std::collections::BTreeSet;

use ddz_core::{cards_to_rank_counts, CardError, Move, MoveKind, RankCounts, EMPTY_RANK_COUNTS};
use ddz_rules::{
    detect_move, detect_move_with_rules, generate_follow_moves, GenerateMovesError, RuleConfig,
};
use proptest::prelude::*;

const DOUZERO_POST_BID_YAML: &str = include_str!("../../../configs/rules/douzero_post_bid.yaml");
const CANONICAL_FULL_YAML: &str = include_str!("../../../configs/rules/canonical_full.yaml");

fn counts(entries: &[(u8, u8)]) -> RankCounts {
    let mut result = EMPTY_RANK_COUNTS;
    for &(rank_id, count) in entries {
        result[usize::from(rank_id)] = count;
    }
    result
}

fn canonical_rules() -> RuleConfig {
    RuleConfig::from_yaml_str(CANONICAL_FULL_YAML).expect("checked-in profile must be valid")
}

fn douzero_rules() -> RuleConfig {
    RuleConfig::from_yaml_str(DOUZERO_POST_BID_YAML).expect("checked-in profile must be valid")
}

fn detected(cards: RankCounts, rules: &RuleConfig) -> Move {
    detect_move_with_rules(cards, rules).expect("test target must be legal")
}

#[test]
fn ordinary_target_gets_higher_same_kind_bombs_rocket_and_pass() {
    let rules = canonical_rules();
    let target = detected(counts(&[(2, 1)]), &rules);
    let hand = counts(&[(0, 1), (3, 1), (5, 1), (6, 4), (13, 1), (14, 1)]);
    let expected = vec![
        Move::pass(),
        detected(counts(&[(3, 1)]), &rules),
        detected(counts(&[(5, 1)]), &rules),
        detected(counts(&[(6, 1)]), &rules),
        detected(counts(&[(13, 1)]), &rules),
        detected(counts(&[(14, 1)]), &rules),
        detected(counts(&[(6, 4)]), &rules),
        detected(counts(&[(13, 1), (14, 1)]), &rules),
    ];

    assert_eq!(
        generate_follow_moves(&hand, &target, &rules).unwrap(),
        expected
    );
}

#[test]
fn chain_responses_require_the_same_kind_and_length() {
    let rules = canonical_rules();
    let target = detected(counts(&[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]), &rules);
    let hand = counts(&[(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]);
    let responses = generate_follow_moves(&hand, &target, &rules).unwrap();

    assert_eq!(responses.len(), 3);
    assert_eq!(responses[0], Move::pass());
    assert_eq!(responses[1].main_rank(), 1);
    assert_eq!(responses[2].main_rank(), 2);
    assert!(responses[1..].iter().all(|response| {
        response.kind() == MoveKind::Straight && response.chain_len() == target.chain_len()
    }));
}

#[test]
fn every_comparable_non_bomb_kind_can_beat_a_lower_target() {
    let rules = canonical_rules();
    let samples = [
        (counts(&[(0, 2)]), counts(&[(1, 2)])),
        (counts(&[(0, 3)]), counts(&[(1, 3)])),
        (counts(&[(0, 3), (7, 1)]), counts(&[(1, 3), (7, 1)])),
        (counts(&[(0, 3), (7, 2)]), counts(&[(1, 3), (7, 2)])),
        (
            counts(&[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]),
            counts(&[(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]),
        ),
        (
            counts(&[(0, 2), (1, 2), (2, 2)]),
            counts(&[(1, 2), (2, 2), (3, 2)]),
        ),
        (counts(&[(0, 3), (1, 3)]), counts(&[(1, 3), (2, 3)])),
        (
            counts(&[(0, 3), (1, 3), (7, 1), (8, 1)]),
            counts(&[(1, 3), (2, 3), (7, 1), (8, 1)]),
        ),
        (
            counts(&[(0, 3), (1, 3), (7, 2), (8, 2)]),
            counts(&[(1, 3), (2, 3), (7, 2), (8, 2)]),
        ),
        (
            counts(&[(0, 4), (7, 1), (8, 1)]),
            counts(&[(1, 4), (7, 1), (8, 1)]),
        ),
        (
            counts(&[(0, 4), (7, 2), (8, 2)]),
            counts(&[(1, 4), (7, 2), (8, 2)]),
        ),
    ];

    for (target_cards, response_cards) in samples {
        let target = detected(target_cards, &rules);
        let expected = detected(response_cards, &rules);
        let responses = generate_follow_moves(&response_cards, &target, &rules).unwrap();
        assert!(
            responses.contains(&expected),
            "missing {:?} response to {:?}",
            expected.kind(),
            target.kind()
        );
    }
}

#[test]
fn bomb_and_rocket_targets_apply_the_override_hierarchy() {
    let rules = canonical_rules();
    let hand = counts(&[(1, 4), (6, 4), (12, 4), (13, 1), (14, 1)]);
    let target = detected(counts(&[(5, 4)]), &rules);
    let responses = generate_follow_moves(&hand, &target, &rules).unwrap();

    assert_eq!(
        responses,
        vec![
            Move::pass(),
            detected(counts(&[(6, 4)]), &rules),
            detected(counts(&[(12, 4)]), &rules),
            detected(counts(&[(13, 1), (14, 1)]), &rules),
        ]
    );

    let rocket = detected(counts(&[(13, 1), (14, 1)]), &rules);
    assert_eq!(
        generate_follow_moves(&hand, &rocket, &rules).unwrap(),
        vec![Move::pass()]
    );
}

#[test]
fn pass_is_the_only_response_when_nothing_can_beat_the_target() {
    let rules = canonical_rules();
    let target = detected(counts(&[(12, 4)]), &rules);
    let hand = counts(&[(0, 4), (1, 3), (2, 2)]);

    assert_eq!(
        generate_follow_moves(&hand, &target, &rules).unwrap(),
        vec![Move::pass()]
    );
}

#[test]
fn pass_illegal_and_noncanonical_targets_are_rejected() {
    let canonical = canonical_rules();
    let hand = counts(&[(0, 1)]);
    assert!(matches!(
        generate_follow_moves(&hand, &Move::pass(), &canonical),
        Err(GenerateMovesError::TargetIsPass)
    ));

    let profile_illegal = detect_move(counts(&[(0, 3), (1, 3), (3, 2)])).unwrap();
    assert!(matches!(
        generate_follow_moves(&hand, &profile_illegal, &canonical),
        Err(GenerateMovesError::Target(_))
    ));

    let douzero = douzero_rules();
    let ambiguous_cards = counts(&[(0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (6, 1)]);
    let noncanonical = Move::try_new(MoveKind::AirplaneWithSingles, ambiguous_cards, 0, 4).unwrap();
    assert!(matches!(
        generate_follow_moves(&hand, &noncanonical, &douzero),
        Err(GenerateMovesError::NonCanonicalTarget { .. })
    ));
}

#[test]
fn invalid_hands_and_rule_configurations_are_rejected_before_filtering() {
    let rules = canonical_rules();
    let target = detected(counts(&[(0, 1)]), &rules);
    assert!(matches!(
        generate_follow_moves(&counts(&[(0, 5)]), &target, &rules),
        Err(GenerateMovesError::Cards(CardError::TooManyCardsForRank {
            rank_id: 0,
            ..
        }))
    ));

    let mut invalid_rules = rules;
    invalid_rules.schema_version = 2;
    assert!(matches!(
        generate_follow_moves(&counts(&[(1, 1)]), &target, &invalid_rules),
        Err(GenerateMovesError::RuleConfig(_))
    ));
}

fn target_samples(rules: &RuleConfig) -> Vec<Move> {
    [
        counts(&[(2, 1)]),
        counts(&[(2, 2)]),
        counts(&[(2, 3)]),
        counts(&[(2, 3), (7, 1)]),
        counts(&[(2, 3), (7, 2)]),
        counts(&[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]),
        counts(&[(0, 2), (1, 2), (2, 2)]),
        counts(&[(0, 3), (1, 3)]),
        counts(&[(0, 3), (1, 3), (7, 1), (8, 1)]),
        counts(&[(0, 3), (1, 3), (7, 2), (8, 2)]),
        counts(&[(2, 4), (7, 1), (8, 1)]),
        counts(&[(2, 4), (7, 2), (8, 2)]),
        counts(&[(5, 4)]),
        counts(&[(13, 1), (14, 1)]),
    ]
    .into_iter()
    .map(|cards| detected(cards, rules))
    .collect()
}

fn oracle_beats(candidate: &Move, target: &Move) -> bool {
    match (candidate.kind(), target.kind()) {
        (_, MoveKind::Rocket) => false,
        (MoveKind::Bomb, MoveKind::Bomb) => candidate.main_rank() > target.main_rank(),
        (MoveKind::Rocket | MoveKind::Bomb, _) => true,
        (_, MoveKind::Bomb) => false,
        _ => {
            candidate.kind() == target.kind()
                && candidate.chain_len() == target.chain_len()
                && candidate.main_rank() > target.main_rank()
        }
    }
}

fn brute_force_responses(hand: &RankCounts, target: &Move, rules: &RuleConfig) -> BTreeSet<Move> {
    fn visit(
        hand: &RankCounts,
        target: &Move,
        rules: &RuleConfig,
        rank: usize,
        candidate: &mut RankCounts,
        responses: &mut BTreeSet<Move>,
    ) {
        if rank == hand.len() {
            if let Ok(detected) = detect_move_with_rules(*candidate, rules) {
                if oracle_beats(&detected, target) {
                    responses.insert(detected);
                }
            }
            return;
        }

        for count in 0..=hand[rank] {
            candidate[rank] = count;
            visit(hand, target, rules, rank + 1, candidate, responses);
        }
        candidate[rank] = 0;
    }

    let mut responses = BTreeSet::from([Move::pass()]);
    let mut candidate = EMPTY_RANK_COUNTS;
    visit(hand, target, rules, 0, &mut candidate, &mut responses);
    responses
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn follow_generation_matches_small_hand_subset_oracle(
        cards in proptest::collection::btree_set(0_u8..54, 0..=8),
        target_index in any::<usize>(),
    ) {
        let hand = cards_to_rank_counts(&cards.into_iter().collect::<Vec<_>>()).unwrap();
        let rules = canonical_rules();
        let samples = target_samples(&rules);
        let target = samples[target_index % samples.len()];
        let expected = brute_force_responses(&hand, &target, &rules);
        let generated = generate_follow_moves(&hand, &target, &rules).unwrap();

        prop_assert_eq!(generated, expected.into_iter().collect::<Vec<_>>());
    }
}
