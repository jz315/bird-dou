use crate::game::{Action, MatchProgress, Round, RoundOutcome};
use crate::{Card, Hand, Rank, Seat, Suit, Team};

fn card(copy: u8, suit: Suit, rank: Rank) -> Card {
    Card::standard(copy, suit, rank).expect("test card is valid")
}

fn hand(cards: &[Card]) -> Hand {
    Hand::from_cards(cards.iter().copied()).expect("test hand is valid")
}

#[test]
fn deal_is_deterministic_and_complete() {
    let first = Round::new(42, Rank::Two, Seat::ZERO).unwrap();
    let second = Round::new(42, Rank::Two, Seat::ZERO).unwrap();

    assert_eq!(first, second);
    assert!(Seat::ALL
        .into_iter()
        .all(|seat| first.hand(seat).len() == 27));
}

#[test]
fn failed_step_is_transactional() {
    let mut round = Round::new(9, Rank::Two, Seat::ZERO).unwrap();
    let before = round.clone();
    let foreign_card = round.hand(Seat::ONE).cards().next().unwrap();

    assert!(round
        .step(Seat::ZERO, Action::Play(vec![foreign_card]))
        .is_err());
    assert_eq!(round, before);
}

#[test]
fn finished_players_partner_receives_the_lead() {
    let lead = card(0, Suit::Clubs, Rank::Three);
    let hands = [
        hand(&[lead]),
        hand(&[card(0, Suit::Clubs, Rank::Four)]),
        hand(&[card(0, Suit::Clubs, Rank::Five)]),
        hand(&[card(0, Suit::Clubs, Rank::Six)]),
    ];
    let mut round = Round::from_hands(Rank::Two, hands, Seat::ZERO).unwrap();

    round.step(Seat::ZERO, Action::Play(vec![lead])).unwrap();
    round.step(Seat::ONE, Action::Pass).unwrap();
    round.step(Seat::TWO, Action::Pass).unwrap();
    let result = round.step(Seat::THREE, Action::Pass).unwrap();

    assert!(result.trick_ended);
    assert_eq!(result.next_player, Some(Seat::TWO));
    assert!(round.target_move().is_none());
}

#[test]
fn finishing_positions_control_promotion() {
    let double_up =
        RoundOutcome::from_finish_order([Seat::ZERO, Seat::TWO, Seat::ONE, Seat::THREE]).unwrap();
    let head_and_third =
        RoundOutcome::from_finish_order([Seat::ZERO, Seat::ONE, Seat::TWO, Seat::THREE]).unwrap();
    let head_and_tail =
        RoundOutcome::from_finish_order([Seat::ZERO, Seat::ONE, Seat::THREE, Seat::TWO]).unwrap();

    assert_eq!(double_up.level_advance(), 3);
    assert_eq!(head_and_third.level_advance(), 2);
    assert_eq!(head_and_tail.level_advance(), 1);
}

#[test]
fn a_team_must_win_at_ace_to_finish_the_match() {
    let outcome =
        RoundOutcome::from_finish_order([Seat::ONE, Seat::THREE, Seat::ZERO, Seat::TWO]).unwrap();
    let mut progress = MatchProgress::new(Rank::King).unwrap();

    assert_eq!(progress.record_round(&outcome).unwrap(), Rank::Ace);
    assert_eq!(progress.winner(), None);
    assert_eq!(progress.record_round(&outcome).unwrap(), Rank::Ace);
    assert_eq!(progress.winner(), Some(Team::One));
}

#[test]
fn third_finisher_completes_the_round_and_settles_positions() {
    let zero = card(0, Suit::Clubs, Rank::Three);
    let one = card(0, Suit::Clubs, Rank::Four);
    let two = card(0, Suit::Clubs, Rank::Five);
    let hands = [
        hand(&[zero]),
        hand(&[one]),
        hand(&[two]),
        hand(&[card(0, Suit::Clubs, Rank::Six)]),
    ];
    let mut round = Round::from_hands(Rank::Two, hands, Seat::ZERO).unwrap();

    round.step(Seat::ZERO, Action::Play(vec![zero])).unwrap();
    round.step(Seat::ONE, Action::Play(vec![one])).unwrap();
    let result = round.step(Seat::TWO, Action::Play(vec![two])).unwrap();

    let outcome = result.round_outcome.unwrap();
    assert_eq!(
        outcome.finish_order(),
        &[Seat::ZERO, Seat::ONE, Seat::TWO, Seat::THREE]
    );
    assert_eq!(outcome.level_advance(), 2);
    assert_eq!(result.next_player, None);
}
