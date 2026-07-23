use ddz_core::{MoveKind, Rank, RankCounts};
use ddz_rules::{
    detect_move_with_rules, generate_follow_moves, generate_lead_moves, move_beats, RewardMode,
    RuleConfig,
};

fn counts(entries: &[(Rank, u8)]) -> RankCounts {
    let mut value = RankCounts::empty();
    for &(rank, count) in entries {
        value.set(rank, count).expect("physical count");
    }
    value
}

#[test]
fn detects_standard_shapes() {
    let rules = RuleConfig::douzero_post_bid(1, RewardMode::WinPercentage);
    let straight = counts(&[
        (Rank::Three, 1),
        (Rank::Four, 1),
        (Rank::Five, 1),
        (Rank::Six, 1),
        (Rank::Seven, 1),
    ]);
    let movement = detect_move_with_rules(straight, &rules).expect("straight");
    assert_eq!(movement.kind(), MoveKind::Straight);
    assert_eq!(movement.main_rank(), Rank::Three.value());
    assert_eq!(movement.chain_len(), 5);

    let airplane = counts(&[
        (Rank::Three, 3),
        (Rank::Four, 3),
        (Rank::Five, 2),
    ]);
    assert_eq!(
        detect_move_with_rules(airplane, &rules)
            .expect("airplane")
            .kind(),
        MoveKind::AirplaneWithSingles
    );
}

#[test]
fn follow_generation_is_shape_directed() {
    let rules = RuleConfig::douzero_post_bid(2, RewardMode::WinPercentage);
    let hand = counts(&[
        (Rank::Three, 2),
        (Rank::Four, 2),
        (Rank::Five, 4),
        (Rank::SmallJoker, 1),
        (Rank::BigJoker, 1),
    ]);
    let target = detect_move_with_rules(counts(&[(Rank::Three, 2)]), &rules).unwrap();
    let responses = generate_follow_moves(hand, target, &rules).unwrap();
    assert!(responses.iter().any(|movement| movement.is_pass()));
    assert!(responses.iter().any(|movement| movement.kind() == MoveKind::Bomb));
    assert!(responses.iter().any(|movement| movement.kind() == MoveKind::Rocket));
    assert!(responses
        .iter()
        .filter(|movement| !movement.is_pass())
        .all(|movement| move_beats(*movement, target)));
    assert!(responses.iter().all(|movement| {
        movement.is_pass()
            || matches!(movement.kind(), MoveKind::Pair | MoveKind::Bomb | MoveKind::Rocket)
    }));
}

#[test]
fn every_generated_lead_is_canonical_and_contained() {
    let rules = RuleConfig::douzero_post_bid(3, RewardMode::WinPercentage);
    let hand = counts(&[
        (Rank::Three, 3),
        (Rank::Four, 3),
        (Rank::Five, 2),
        (Rank::Six, 2),
        (Rank::Seven, 1),
        (Rank::Eight, 1),
        (Rank::Nine, 1),
        (Rank::Ten, 1),
        (Rank::SmallJoker, 1),
        (Rank::BigJoker, 1),
    ]);
    for movement in generate_lead_moves(hand, &rules).expect("lead moves") {
        assert!(!movement.is_pass());
        assert!(hand.contains(movement.cards()));
        assert_eq!(
            detect_move_with_rules(movement.cards(), &rules).unwrap(),
            movement
        );
    }
}

#[test]
fn generated_moves_are_canonical_across_many_physical_deals() {
    use ddz_core::Seat;
    use ddz_rules::Game;

    let rules = RuleConfig::douzero_post_bid(4, RewardMode::WinPercentage);
    for seed in 0_u64..64 {
        let game = Game::new_post_bid(rules.clone(), seed, Seat::ZERO).expect("physical deal");
        for seat in Seat::ALL {
            let hand = game.state().hands[seat];
            for movement in generate_lead_moves(hand, &rules).expect("generated moves") {
                assert!(hand.contains(movement.cards()));
                assert_eq!(
                    detect_move_with_rules(movement.cards(), &rules).expect("canonical move"),
                    movement
                );
            }
        }
    }
}
