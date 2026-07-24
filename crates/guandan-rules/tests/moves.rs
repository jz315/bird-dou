use guandan_rules::{beats, detect_move, Card, MoveKind, Rank, Suit};

fn card(copy: u8, suit: Suit, rank: Rank) -> Card {
    Card::standard(copy, suit, rank).unwrap()
}

fn suited_run(suit: Suit, ranks: &[Rank]) -> Vec<Card> {
    ranks
        .iter()
        .enumerate()
        .map(|(index, rank)| {
            card(
                u8::try_from(index % 2).expect("copy alternates between zero and one"),
                suit,
                *rank,
            )
        })
        .collect()
}

#[test]
fn fixed_length_sequences_follow_the_pdf() {
    let straight = suited_run(
        Suit::Clubs,
        &[Rank::Three, Rank::Four, Rank::Five, Rank::Six, Rank::Seven],
    );
    let too_long = suited_run(
        Suit::Clubs,
        &[
            Rank::Three,
            Rank::Four,
            Rank::Five,
            Rank::Six,
            Rank::Seven,
            Rank::Eight,
        ],
    );
    let pair_straight = vec![
        card(0, Suit::Clubs, Rank::Queen),
        card(1, Suit::Diamonds, Rank::Queen),
        card(0, Suit::Clubs, Rank::King),
        card(1, Suit::Diamonds, Rank::King),
        card(0, Suit::Clubs, Rank::Ace),
        card(1, Suit::Diamonds, Rank::Ace),
    ];

    assert!(matches!(
        detect_move(&straight, Rank::Two).unwrap().kind(),
        MoveKind::StraightFlush { .. }
    ));
    assert!(detect_move(&too_long, Rank::Two).is_err());
    assert!(matches!(
        detect_move(&pair_straight, Rank::Two).unwrap().kind(),
        MoveKind::PairStraight {
            high: Rank::Ace,
            ..
        }
    ));
}

#[test]
fn two_triples_are_a_steel_plate_but_cannot_take_attachments() {
    let steel_plate = vec![
        card(0, Suit::Clubs, Rank::Three),
        card(0, Suit::Diamonds, Rank::Three),
        card(1, Suit::Spades, Rank::Three),
        card(0, Suit::Clubs, Rank::Four),
        card(0, Suit::Diamonds, Rank::Four),
        card(1, Suit::Spades, Rank::Four),
    ];
    let mut with_attachment = steel_plate.clone();
    with_attachment.push(card(0, Suit::Clubs, Rank::Nine));

    assert!(matches!(
        detect_move(&steel_plate, Rank::Two).unwrap().kind(),
        MoveKind::TripleStraight { .. }
    ));
    assert!(detect_move(&with_attachment, Rank::Two).is_err());
}

#[test]
fn wildcards_fill_standard_cards_and_make_ten_card_bombs() {
    let mut bomb = Vec::new();
    for copy in 0..2 {
        for suit in Suit::ALL {
            bomb.push(card(copy, suit, Rank::Nine));
        }
    }
    bomb.push(card(0, Suit::Hearts, Rank::Five));
    bomb.push(card(1, Suit::Hearts, Rank::Five));

    assert!(matches!(
        detect_move(&bomb, Rank::Five).unwrap().kind(),
        MoveKind::Bomb {
            rank: Rank::Nine,
            size: 10
        }
    ));
}

#[test]
fn straight_flush_sits_between_five_and_six_card_bombs() {
    let straight_flush = detect_move(
        &suited_run(
            Suit::Spades,
            &[Rank::Six, Rank::Seven, Rank::Eight, Rank::Nine, Rank::Ten],
        ),
        Rank::Two,
    )
    .unwrap();
    let five_bomb = detect_move(
        &[
            card(0, Suit::Clubs, Rank::Jack),
            card(0, Suit::Diamonds, Rank::Jack),
            card(0, Suit::Hearts, Rank::Jack),
            card(0, Suit::Spades, Rank::Jack),
            card(1, Suit::Clubs, Rank::Jack),
        ],
        Rank::Two,
    )
    .unwrap();
    let six_bomb = detect_move(
        &[
            card(0, Suit::Clubs, Rank::Three),
            card(0, Suit::Diamonds, Rank::Three),
            card(0, Suit::Hearts, Rank::Three),
            card(0, Suit::Spades, Rank::Three),
            card(1, Suit::Clubs, Rank::Three),
            card(1, Suit::Diamonds, Rank::Three),
        ],
        Rank::Two,
    )
    .unwrap();

    assert!(beats(&straight_flush, &five_bomb, Rank::Two));
    assert!(beats(&six_bomb, &straight_flush, Rank::Two));
}

#[test]
fn four_jokers_are_the_largest_move() {
    let jokers = [Rank::SmallJoker, Rank::BigJoker]
        .into_iter()
        .flat_map(|rank| [Card::joker(0, rank).unwrap(), Card::joker(1, rank).unwrap()])
        .collect::<Vec<_>>();
    let joker_bomb = detect_move(&jokers, Rank::Ten).unwrap();
    let ten_bomb = (0..2)
        .flat_map(|copy| {
            Suit::ALL
                .into_iter()
                .map(move |suit| card(copy, suit, Rank::Ace))
        })
        .chain([
            card(0, Suit::Hearts, Rank::Ten),
            card(1, Suit::Hearts, Rank::Ten),
        ])
        .collect::<Vec<_>>();
    let ten_bomb = detect_move(&ten_bomb, Rank::Ten).unwrap();

    assert!(matches!(joker_bomb.kind(), MoveKind::FourJokers));
    assert!(beats(&joker_bomb, &ten_bomb, Rank::Ten));
}

#[test]
fn wildcard_cannot_complete_a_joker_pair() {
    let cards = [
        Card::joker(0, Rank::BigJoker).unwrap(),
        card(0, Suit::Hearts, Rank::Ten),
    ];

    assert!(detect_move(&cards, Rank::Ten).is_err());
}

#[test]
fn ordinary_moves_only_beat_the_same_shape() {
    let pair = detect_move(
        &[
            card(0, Suit::Clubs, Rank::Ace),
            card(1, Suit::Diamonds, Rank::Ace),
        ],
        Rank::Ten,
    )
    .unwrap();
    let triple = detect_move(
        &[
            card(0, Suit::Clubs, Rank::Three),
            card(1, Suit::Diamonds, Rank::Three),
            card(0, Suit::Spades, Rank::Three),
        ],
        Rank::Ten,
    )
    .unwrap();
    let level_pair = detect_move(
        &[
            card(0, Suit::Clubs, Rank::Ten),
            card(1, Suit::Diamonds, Rank::Ten),
        ],
        Rank::Ten,
    )
    .unwrap();

    assert!(!beats(&triple, &pair, Rank::Ten));
    assert!(beats(&level_pair, &pair, Rank::Ten));
}
