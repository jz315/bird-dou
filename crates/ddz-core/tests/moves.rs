use ddz_core::{
    CardError, Move, MoveError, MoveKind, RankCounts, BIG_JOKER_RANK, EMPTY_RANK_COUNTS,
    PASS_MAIN_RANK,
};

const ALL_KINDS: [MoveKind; 15] = [
    MoveKind::Pass,
    MoveKind::Single,
    MoveKind::Pair,
    MoveKind::Triple,
    MoveKind::TripleWithSingle,
    MoveKind::TripleWithPair,
    MoveKind::Straight,
    MoveKind::PairStraight,
    MoveKind::TripleStraight,
    MoveKind::AirplaneWithSingles,
    MoveKind::AirplaneWithPairs,
    MoveKind::FourWithTwoSingles,
    MoveKind::FourWithTwoPairs,
    MoveKind::Bomb,
    MoveKind::Rocket,
];

fn counts(entries: &[(u8, u8)]) -> RankCounts {
    let mut result = EMPTY_RANK_COUNTS;
    for &(rank_id, count) in entries {
        result[usize::from(rank_id)] = count;
    }
    result
}

fn move_of(kind: MoveKind, cards: RankCounts, main_rank: u8, chain_len: u8) -> Move {
    Move::try_new(kind, cards, main_rank, chain_len).expect("test move must be canonical")
}

fn one_move_of_every_kind() -> Vec<Move> {
    vec![
        Move::pass(),
        move_of(MoveKind::Single, counts(&[(0, 1)]), 0, 1),
        move_of(MoveKind::Pair, counts(&[(0, 2)]), 0, 1),
        move_of(MoveKind::Triple, counts(&[(0, 3)]), 0, 1),
        move_of(MoveKind::TripleWithSingle, counts(&[(0, 3), (1, 1)]), 0, 1),
        move_of(MoveKind::TripleWithPair, counts(&[(0, 3), (1, 2)]), 0, 1),
        move_of(
            MoveKind::Straight,
            counts(&[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]),
            0,
            5,
        ),
        move_of(
            MoveKind::PairStraight,
            counts(&[(0, 2), (1, 2), (2, 2)]),
            0,
            3,
        ),
        move_of(MoveKind::TripleStraight, counts(&[(0, 3), (1, 3)]), 0, 2),
        move_of(
            MoveKind::AirplaneWithSingles,
            counts(&[(0, 3), (1, 3), (2, 2)]),
            0,
            2,
        ),
        move_of(
            MoveKind::AirplaneWithPairs,
            counts(&[(0, 3), (1, 3), (2, 2), (3, 2)]),
            0,
            2,
        ),
        move_of(
            MoveKind::FourWithTwoSingles,
            counts(&[(0, 4), (1, 2)]),
            0,
            1,
        ),
        move_of(
            MoveKind::FourWithTwoPairs,
            counts(&[(0, 4), (1, 2), (2, 2)]),
            0,
            1,
        ),
        move_of(MoveKind::Bomb, counts(&[(0, 4)]), 0, 1),
        move_of(MoveKind::Rocket, counts(&[(13, 1), (14, 1)]), 14, 1),
    ]
}

#[test]
fn move_kind_numeric_tags_are_stable_and_complete() {
    for (expected_tag, kind) in (0_u8..).zip(ALL_KINDS) {
        assert_eq!(u8::from(kind), expected_tag);
        assert_eq!(MoveKind::try_from(expected_tag), Ok(kind));
    }
    assert_eq!(
        MoveKind::try_from(15),
        Err(MoveError::UnknownKindTag { tag: 15 })
    );
}

#[test]
fn pass_has_one_canonical_representation() {
    let pass = Move::pass();

    assert_eq!(pass.kind(), MoveKind::Pass);
    assert_eq!(pass.cards(), &EMPTY_RANK_COUNTS);
    assert_eq!(pass.main_rank(), PASS_MAIN_RANK);
    assert_eq!(pass.chain_len(), 0);
    assert_eq!(pass.total_cards(), 0);
    assert_eq!(
        Move::try_new(MoveKind::Pass, EMPTY_RANK_COUNTS, PASS_MAIN_RANK, 0),
        Ok(pass)
    );
    assert_eq!(
        Move::try_new(MoveKind::Pass, EMPTY_RANK_COUNTS, 0, 0),
        Err(MoveError::NonCanonicalPass)
    );
}

#[test]
fn every_kind_constructs_and_serializes_round_trip() {
    let moves = one_move_of_every_kind();

    assert_eq!(moves.len(), ALL_KINDS.len());
    for (expected_kind, original) in ALL_KINDS.into_iter().zip(moves) {
        let yaml = serde_yaml_ng::to_string(&original).expect("move must serialize");
        let decoded: Move = serde_yaml_ng::from_str(&yaml).expect("move must deserialize");

        assert_eq!(original.kind(), expected_kind);
        assert_eq!(decoded, original);
    }
}

#[test]
fn serialization_uses_named_kinds_and_rejects_redundant_total_drift() {
    let straight = move_of(
        MoveKind::Straight,
        counts(&[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]),
        0,
        5,
    );
    let yaml = serde_yaml_ng::to_string(&straight).expect("move must serialize");
    assert!(yaml.contains("kind: straight"));

    let drifted = yaml.replace("total_cards: 5", "total_cards: 6");
    let error = serde_yaml_ng::from_str::<Move>(&drifted).expect_err("drift must be rejected");
    assert!(error.to_string().contains("rank counts contain 5"));
}

#[test]
fn stable_order_uses_kind_then_shape_then_main_rank_then_cards() {
    let single_three = move_of(MoveKind::Single, counts(&[(0, 1)]), 0, 1);
    let single_four = move_of(MoveKind::Single, counts(&[(1, 1)]), 1, 1);
    let straight_five = move_of(
        MoveKind::Straight,
        counts(&[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]),
        0,
        5,
    );
    let straight_six = move_of(
        MoveKind::Straight,
        counts(&[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]),
        0,
        6,
    );
    let bomb = move_of(MoveKind::Bomb, counts(&[(0, 4)]), 0, 1);
    let expected = vec![
        Move::pass(),
        single_three,
        single_four,
        straight_five,
        straight_six,
        bomb,
    ];
    let mut shuffled = vec![
        bomb,
        straight_six,
        single_four,
        Move::pass(),
        straight_five,
        single_three,
    ];

    shuffled.sort();

    assert_eq!(shuffled, expected);
}

#[test]
fn impossible_counts_and_shape_metadata_are_rejected() {
    assert_eq!(
        Move::try_new(MoveKind::Single, counts(&[(0, 5)]), 0, 1),
        Err(MoveError::Cards(CardError::TooManyCardsForRank {
            rank_id: 0,
            count: 5,
            maximum: 4,
        }))
    );
    assert!(matches!(
        Move::try_new(MoveKind::Pair, counts(&[(0, 2)]), 1, 1),
        Err(MoveError::BodyCountMismatch {
            kind: MoveKind::Pair,
            rank_id: 1,
            ..
        })
    ));
    assert!(matches!(
        Move::try_new(MoveKind::TripleWithSingle, counts(&[(0, 3)]), 0, 1),
        Err(MoveError::InvalidTotalCards {
            kind: MoveKind::TripleWithSingle,
            actual: 3,
            expected: 4,
        })
    ));
    assert_eq!(
        Move::try_new(
            MoveKind::TripleWithPair,
            counts(&[(0, 3), (1, 1), (2, 1)]),
            0,
            1,
        ),
        Err(MoveError::InvalidPairAttachment {
            kind: MoveKind::TripleWithPair,
            rank_id: 1,
            count: 1,
        })
    );
}

#[test]
fn chains_cannot_be_too_short_or_include_two() {
    assert!(matches!(
        Move::try_new(
            MoveKind::Straight,
            counts(&[(0, 1), (1, 1), (2, 1), (3, 1)]),
            0,
            4,
        ),
        Err(MoveError::InvalidChainLength {
            kind: MoveKind::Straight,
            chain_len: 4,
            ..
        })
    ));
    assert_eq!(
        Move::try_new(
            MoveKind::Straight,
            counts(&[(8, 1), (9, 1), (10, 1), (11, 1), (12, 1)]),
            8,
            5,
        ),
        Err(MoveError::InvalidChainRange {
            main_rank: 8,
            chain_len: 5,
        })
    );
}

#[test]
fn rocket_requires_the_jokers_and_big_joker_main_rank() {
    assert_eq!(
        Move::try_new(MoveKind::Rocket, counts(&[(13, 1), (14, 1)]), 13, 1,),
        Err(MoveError::UnexpectedMainRank {
            kind: MoveKind::Rocket,
            actual: 13,
            expected: BIG_JOKER_RANK,
        })
    );
    assert_eq!(
        Move::try_new(MoveKind::Rocket, counts(&[(0, 2)]), BIG_JOKER_RANK, 1),
        Err(MoveError::InvalidRocket)
    );
}
