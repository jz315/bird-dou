use ddz_core::{CardError, Move, MoveKind, RankCounts, EMPTY_RANK_COUNTS};
use ddz_rules::{detect_move, detect_move_with_rules, DetectMoveError, RuleConfig, RuleProfile};
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

fn assert_detected(
    cards: RankCounts,
    expected_kind: MoveKind,
    expected_main_rank: u8,
    expected_chain_len: u8,
) -> Move {
    let detected = detect_move(cards).expect("cards must form the expected move");

    assert_eq!(detected.kind(), expected_kind);
    assert_eq!(detected.main_rank(), expected_main_rank);
    assert_eq!(detected.chain_len(), expected_chain_len);
    assert_eq!(detected.cards(), &cards);
    assert_eq!(detected.total_cards(), cards.iter().copied().sum::<u8>());
    detected
}

#[test]
fn detects_pass_and_every_fixed_size_kind() {
    assert_detected(EMPTY_RANK_COUNTS, MoveKind::Pass, 15, 0);
    assert_detected(counts(&[(0, 1)]), MoveKind::Single, 0, 1);
    assert_detected(counts(&[(1, 2)]), MoveKind::Pair, 1, 1);
    assert_detected(counts(&[(2, 3)]), MoveKind::Triple, 2, 1);
    assert_detected(counts(&[(2, 3), (5, 1)]), MoveKind::TripleWithSingle, 2, 1);
    assert_detected(counts(&[(2, 3), (5, 2)]), MoveKind::TripleWithPair, 2, 1);
    assert_detected(counts(&[(12, 4)]), MoveKind::Bomb, 12, 1);
    assert_detected(counts(&[(13, 1), (14, 1)]), MoveKind::Rocket, 14, 1);
}

#[test]
fn detects_all_chain_kinds_at_their_lower_bound() {
    assert_detected(
        counts(&[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]),
        MoveKind::Straight,
        0,
        5,
    );
    assert_detected(
        counts(&[(0, 2), (1, 2), (2, 2)]),
        MoveKind::PairStraight,
        0,
        3,
    );
    assert_detected(counts(&[(0, 3), (1, 3)]), MoveKind::TripleStraight, 0, 2);
    assert_detected(
        counts(&[(0, 3), (1, 3), (3, 1), (4, 1)]),
        MoveKind::AirplaneWithSingles,
        0,
        2,
    );
    assert_detected(
        counts(&[(0, 3), (1, 3), (3, 2), (4, 2)]),
        MoveKind::AirplaneWithPairs,
        0,
        2,
    );
}

#[test]
fn detects_both_four_with_two_forms() {
    assert_detected(
        counts(&[(0, 4), (3, 1), (4, 1)]),
        MoveKind::FourWithTwoSingles,
        0,
        1,
    );
    assert_detected(
        counts(&[(0, 4), (3, 2), (4, 2)]),
        MoveKind::FourWithTwoPairs,
        0,
        1,
    );
}

#[test]
fn detects_both_legal_straight_edges() {
    assert_detected(
        counts(&[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]),
        MoveKind::Straight,
        0,
        5,
    );
    assert_detected(
        counts(&[(7, 1), (8, 1), (9, 1), (10, 1), (11, 1)]),
        MoveKind::Straight,
        7,
        5,
    );
    assert_detected(
        counts(&[
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 1),
            (7, 1),
            (8, 1),
            (9, 1),
            (10, 1),
            (11, 1),
        ]),
        MoveKind::Straight,
        0,
        12,
    );
}

#[test]
fn rejects_short_gapped_or_high_chains() {
    for invalid in [
        counts(&[(0, 1), (1, 1), (2, 1), (3, 1)]),
        counts(&[(0, 1), (1, 1), (3, 1), (4, 1), (5, 1)]),
        counts(&[(8, 1), (9, 1), (10, 1), (11, 1), (12, 1)]),
        counts(&[(9, 2), (10, 2), (11, 2), (12, 2)]),
        counts(&[(10, 3), (11, 3), (12, 3)]),
    ] {
        assert!(matches!(
            detect_move(invalid),
            Err(DetectMoveError::Unrecognized { .. })
        ));
    }
}

#[test]
fn rejects_malformed_attachments() {
    for invalid in [
        counts(&[(0, 3), (1, 1), (2, 1)]),
        counts(&[(0, 4), (1, 1), (2, 1), (3, 2)]),
        counts(&[(0, 3), (1, 3), (2, 1), (3, 1), (4, 2)]),
    ] {
        assert!(matches!(
            detect_move(invalid),
            Err(DetectMoveError::Unrecognized { .. })
        ));
    }
}

#[test]
fn rejects_impossible_physical_counts_before_classification() {
    assert!(matches!(
        detect_move(counts(&[(0, 5)])),
        Err(DetectMoveError::Cards(CardError::TooManyCardsForRank {
            rank_id: 0,
            count: 5,
            maximum: 4,
        }))
    ));
    assert!(matches!(
        detect_move(counts(&[(13, 2)])),
        Err(DetectMoveError::Cards(CardError::TooManyCardsForRank {
            rank_id: 13,
            count: 2,
            maximum: 1,
        }))
    ));
}

#[test]
fn profile_specific_single_wing_multiplicity_is_enforced() {
    let douzero = RuleConfig::from_yaml_str(DOUZERO_POST_BID_YAML).expect("profile is valid");
    let canonical = RuleConfig::from_yaml_str(CANONICAL_FULL_YAML).expect("profile is valid");
    let shared_airplane_wing = counts(&[(0, 3), (1, 3), (3, 2)]);
    let distinct_airplane_wings = counts(&[(0, 3), (1, 3), (3, 1), (4, 1)]);
    let shared_four_wing = counts(&[(0, 4), (3, 2)]);

    assert!(detect_move_with_rules(shared_airplane_wing, &douzero).is_ok());
    assert!(detect_move_with_rules(distinct_airplane_wings, &douzero).is_ok());
    assert!(detect_move_with_rules(distinct_airplane_wings, &canonical).is_ok());
    assert!(matches!(
        detect_move_with_rules(shared_airplane_wing, &canonical),
        Err(DetectMoveError::AttachmentRanksMustBeDistinct {
            kind: MoveKind::AirplaneWithSingles,
            rank_id: 3,
        })
    ));
    assert!(detect_move_with_rules(shared_four_wing, &douzero).is_ok());
    assert!(detect_move_with_rules(shared_four_wing, &canonical).is_ok());
}

#[test]
fn repeated_pair_wing_ranks_are_rejected_by_checked_in_profiles() {
    let douzero = RuleConfig::from_yaml_str(DOUZERO_POST_BID_YAML).expect("profile is valid");
    let canonical = RuleConfig::from_yaml_str(CANONICAL_FULL_YAML).expect("profile is valid");
    let two_quads = counts(&[(0, 4), (1, 4)]);
    let repeated_airplane_pairs = counts(&[(0, 3), (1, 3), (3, 4)]);
    let detected = assert_detected(two_quads, MoveKind::FourWithTwoPairs, 1, 1);
    assert_detected(repeated_airplane_pairs, MoveKind::AirplaneWithPairs, 0, 2);

    assert_eq!(detected.cards(), &two_quads);
    for rules in [&douzero, &canonical] {
        assert!(matches!(
            detect_move_with_rules(two_quads, rules),
            Err(DetectMoveError::AttachmentRanksMustBeDistinct {
                kind: MoveKind::FourWithTwoPairs,
                rank_id: 0,
            })
        ));
        assert!(matches!(
            detect_move_with_rules(repeated_airplane_pairs, rules),
            Err(DetectMoveError::AttachmentRanksMustBeDistinct {
                kind: MoveKind::AirplaneWithPairs,
                rank_id: 3,
            })
        ));
    }
}

#[test]
fn disabled_four_with_two_form_is_rejected_without_affecting_detection() {
    let mut rules = RuleConfig::from_yaml_str(CANONICAL_FULL_YAML).expect("profile is valid");
    rules.four_with_two.two_singles_enabled = false;
    let cards = counts(&[(0, 4), (3, 1), (4, 1)]);

    assert_eq!(
        detect_move(cards).expect("shape remains detectable").kind(),
        MoveKind::FourWithTwoSingles
    );
    assert!(matches!(
        detect_move_with_rules(cards, &rules),
        Err(DetectMoveError::MoveDisabled {
            kind: MoveKind::FourWithTwoSingles,
            profile: RuleProfile::CanonicalFull,
        })
    ));
}

#[test]
fn ambiguous_airplane_bodies_choose_the_highest_main_rank() {
    let cards = counts(&[(0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (6, 1)]);

    assert_detected(cards, MoveKind::AirplaneWithSingles, 1, 4);
}

#[test]
fn invalid_rule_configuration_is_rejected_before_detection() {
    let mut rules = RuleConfig::from_yaml_str(CANONICAL_FULL_YAML).expect("profile is valid");
    rules.schema_version = 2;

    assert!(matches!(
        detect_move_with_rules(counts(&[(0, 1)]), &rules),
        Err(DetectMoveError::RuleConfig(_))
    ));
}

fn valid_rank_counts() -> impl Strategy<Value = RankCounts> {
    (proptest::collection::vec(0_u8..=4, 13), 0_u8..=1, 0_u8..=1).prop_map(
        |(ordinary, small_joker, big_joker)| {
            let mut cards = EMPTY_RANK_COUNTS;
            cards[..13].copy_from_slice(&ordinary);
            cards[13] = small_joker;
            cards[14] = big_joker;
            cards
        },
    )
}

proptest! {
    #[test]
    fn detection_is_deterministic_and_preserves_successful_inputs(cards in valid_rank_counts()) {
        let first = detect_move(cards);
        let second = detect_move(cards);

        match (first, second) {
            (Ok(first_move), Ok(second_move)) => {
                prop_assert_eq!(first_move, second_move);
                prop_assert_eq!(*first_move.cards(), cards);
                prop_assert_eq!(first_move.total_cards(), cards.iter().copied().sum::<u8>());
                let rebuilt = Move::try_new(
                    first_move.kind(),
                    *first_move.cards(),
                    first_move.main_rank(),
                    first_move.chain_len(),
                );
                prop_assert_eq!(rebuilt, Ok(first_move));
            }
            (Err(_), Err(_)) => {}
            _ => prop_assert!(false, "same input produced inconsistent detection results"),
        }
    }
}
