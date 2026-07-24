use guandan_rules::{generate_legal_moves, Action, Rank, Round, Seat};

#[test]
fn every_generated_move_is_owned_and_legal() {
    for seed in 0..16 {
        let round = Round::new(seed, Rank::Seven, Seat::ZERO).unwrap();
        for seat in Seat::ALL {
            let hand = round.hand(seat);
            let moves = generate_legal_moves(hand, None, Rank::Seven).unwrap();
            assert!(!moves.is_empty());
            assert!(moves
                .iter()
                .all(|movement| hand.contains_all(movement.cards())));
        }
    }
}

#[test]
fn generated_follow_moves_can_be_applied_transactionally() {
    let mut round = Round::new(2026, Rank::Ten, Seat::ZERO).unwrap();
    let lead = generate_legal_moves(round.hand(Seat::ZERO), None, Rank::Ten)
        .unwrap()
        .into_iter()
        .find(|movement| movement.len() == 1)
        .unwrap();
    round
        .step(Seat::ZERO, Action::Play(lead.cards().to_vec()))
        .unwrap();

    let target = round.target_move().unwrap();
    let replies = generate_legal_moves(round.hand(Seat::ONE), Some(target), Rank::Ten).unwrap();
    for reply in replies {
        let mut branch = round.clone();
        branch
            .step(Seat::ONE, Action::Play(reply.cards().to_vec()))
            .unwrap();
    }
}
