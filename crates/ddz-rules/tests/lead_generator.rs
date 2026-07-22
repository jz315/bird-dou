use std::collections::BTreeSet;

use ddz_core::{cards_to_rank_counts, CardError, Move, MoveKind, RankCounts, EMPTY_RANK_COUNTS};
use ddz_rules::{detect_move_with_rules, generate_lead_moves, GenerateMovesError, RuleConfig};
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

#[test]
fn exact_small_hand_matches_the_complete_oracle() {
    let rules = canonical_rules();
    let hand = counts(&[(0, 3), (1, 2), (2, 1)]);
    let mut expected = [
        counts(&[(0, 1)]),
        counts(&[(1, 1)]),
        counts(&[(2, 1)]),
        counts(&[(0, 2)]),
        counts(&[(1, 2)]),
        counts(&[(0, 3)]),
        counts(&[(0, 3), (1, 1)]),
        counts(&[(0, 3), (2, 1)]),
        counts(&[(0, 3), (1, 2)]),
    ]
    .into_iter()
    .map(|cards| detect_move_with_rules(cards, &rules).expect("oracle move must be legal"))
    .collect::<Vec<_>>();
    expected.sort_unstable();

    assert_eq!(generate_lead_moves(&hand, &rules).unwrap(), expected);
}

#[test]
fn every_non_pass_move_kind_is_generated() {
    let rules = canonical_rules();
    let samples = [
        (MoveKind::Single, counts(&[(0, 1)])),
        (MoveKind::Pair, counts(&[(0, 2)])),
        (MoveKind::Triple, counts(&[(0, 3)])),
        (MoveKind::TripleWithSingle, counts(&[(0, 3), (3, 1)])),
        (MoveKind::TripleWithPair, counts(&[(0, 3), (3, 2)])),
        (
            MoveKind::Straight,
            counts(&[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]),
        ),
        (MoveKind::PairStraight, counts(&[(0, 2), (1, 2), (2, 2)])),
        (MoveKind::TripleStraight, counts(&[(0, 3), (1, 3)])),
        (
            MoveKind::AirplaneWithSingles,
            counts(&[(0, 3), (1, 3), (3, 1), (4, 1)]),
        ),
        (
            MoveKind::AirplaneWithPairs,
            counts(&[(0, 3), (1, 3), (3, 2), (4, 2)]),
        ),
        (
            MoveKind::FourWithTwoSingles,
            counts(&[(0, 4), (3, 1), (4, 1)]),
        ),
        (
            MoveKind::FourWithTwoPairs,
            counts(&[(0, 4), (3, 2), (4, 2)]),
        ),
        (MoveKind::Bomb, counts(&[(0, 4)])),
        (MoveKind::Rocket, counts(&[(13, 1), (14, 1)])),
    ];

    for (kind, hand) in samples {
        let expected = detect_move_with_rules(hand, &rules).expect("sample must be legal");
        assert_eq!(expected.kind(), kind);
        let generated = generate_lead_moves(&hand, &rules).unwrap();
        assert!(
            generated.contains(&expected),
            "missing {kind:?} from {hand:?}"
        );
        assert!(generated
            .iter()
            .all(|candidate| candidate.kind() != MoveKind::Pass));
    }
}

#[test]
fn attachment_multiplicity_and_disabled_forms_follow_the_profile() {
    let douzero = douzero_rules();
    let canonical = canonical_rules();
    let shared_airplane_hand = counts(&[(0, 3), (1, 3), (3, 2)]);
    let shared_airplane = detect_move_with_rules(shared_airplane_hand, &douzero).unwrap();

    assert!(generate_lead_moves(&shared_airplane_hand, &douzero)
        .unwrap()
        .contains(&shared_airplane));
    assert!(!generate_lead_moves(&shared_airplane_hand, &canonical)
        .unwrap()
        .iter()
        .any(|candidate| candidate.cards() == &shared_airplane_hand));

    let repeated_pair_hand = counts(&[(0, 3), (1, 3), (3, 4)]);
    for rules in [&douzero, &canonical] {
        assert!(!generate_lead_moves(&repeated_pair_hand, rules)
            .unwrap()
            .iter()
            .any(|candidate| {
                candidate.kind() == MoveKind::AirplaneWithPairs
                    && candidate.cards() == &repeated_pair_hand
            }));
    }

    let four_with_two = counts(&[(0, 4), (3, 1), (4, 1)]);
    let mut singles_disabled = canonical;
    singles_disabled.four_with_two.two_singles_enabled = false;
    assert!(!generate_lead_moves(&four_with_two, &singles_disabled)
        .unwrap()
        .iter()
        .any(|candidate| candidate.kind() == MoveKind::FourWithTwoSingles));
}

#[test]
fn generated_moves_are_sorted_unique_legal_and_bounded_by_the_hand() {
    let rules = douzero_rules();
    let hand = counts(&[
        (0, 4),
        (1, 4),
        (2, 3),
        (3, 3),
        (4, 2),
        (5, 1),
        (13, 1),
        (14, 1),
    ]);
    let generated = generate_lead_moves(&hand, &rules).unwrap();
    let set = generated.iter().copied().collect::<BTreeSet<_>>();

    assert_eq!(generated, set.iter().copied().collect::<Vec<_>>());
    assert!(generated
        .iter()
        .all(|candidate| candidate.kind() != MoveKind::Pass));
    for candidate in generated {
        assert!(candidate
            .cards()
            .iter()
            .zip(hand)
            .all(|(&required, available)| required <= available));
        assert_eq!(
            detect_move_with_rules(*candidate.cards(), &rules).unwrap(),
            candidate
        );
    }
}

#[test]
fn invalid_hands_and_rule_configurations_are_rejected() {
    let rules = canonical_rules();
    assert!(matches!(
        generate_lead_moves(&counts(&[(0, 5)]), &rules),
        Err(GenerateMovesError::Cards(CardError::TooManyCardsForRank {
            rank_id: 0,
            ..
        }))
    ));

    let mut invalid_rules = rules;
    invalid_rules.schema_version = 2;
    assert!(matches!(
        generate_lead_moves(&counts(&[(0, 1)]), &invalid_rules),
        Err(GenerateMovesError::RuleConfig(_))
    ));
}

fn brute_force_lead_moves(hand: &RankCounts, rules: &RuleConfig) -> BTreeSet<Move> {
    fn visit(
        hand: &RankCounts,
        rules: &RuleConfig,
        rank: usize,
        candidate: &mut RankCounts,
        moves: &mut BTreeSet<Move>,
    ) {
        if rank == hand.len() {
            if let Ok(detected) = detect_move_with_rules(*candidate, rules) {
                if detected.kind() != MoveKind::Pass {
                    moves.insert(detected);
                }
            }
            return;
        }

        for count in 0..=hand[rank] {
            candidate[rank] = count;
            visit(hand, rules, rank + 1, candidate, moves);
        }
        candidate[rank] = 0;
    }

    let mut moves = BTreeSet::new();
    let mut candidate = EMPTY_RANK_COUNTS;
    visit(hand, rules, 0, &mut candidate, &mut moves);
    moves
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn template_generation_matches_small_hand_subset_oracle(
        cards in proptest::collection::btree_set(0_u8..54, 0..=8),
        use_douzero in any::<bool>(),
    ) {
        let hand = cards_to_rank_counts(&cards.into_iter().collect::<Vec<_>>()).unwrap();
        let rules = if use_douzero { douzero_rules() } else { canonical_rules() };
        let expected = brute_force_lead_moves(&hand, &rules);
        let generated = generate_lead_moves(&hand, &rules).unwrap();

        prop_assert_eq!(generated, expected.into_iter().collect::<Vec<_>>());
    }

    #[test]
    fn generation_is_deterministic_sorted_and_bounded_for_twenty_card_hands(
        cards in proptest::collection::btree_set(0_u8..54, 0..=20),
    ) {
        let hand = cards_to_rank_counts(&cards.into_iter().collect::<Vec<_>>()).unwrap();
        let rules = douzero_rules();
        let first = generate_lead_moves(&hand, &rules).unwrap();
        let second = generate_lead_moves(&hand, &rules).unwrap();

        prop_assert_eq!(&first, &second);
        prop_assert!(first.windows(2).all(|pair| pair[0] < pair[1]));
        let all_moves_are_bounded = first.iter().all(|candidate| {
            candidate.kind() != MoveKind::Pass
                && candidate
                    .cards()
                    .iter()
                    .zip(hand)
                    .all(|(&required, available)| required <= available)
        });
        prop_assert!(all_moves_are_bounded);
    }
}
