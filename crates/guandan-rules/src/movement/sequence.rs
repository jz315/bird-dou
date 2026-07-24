use crate::Rank;

const CYCLIC_RANKS: [Rank; 14] = [
    Rank::Ace,
    Rank::Two,
    Rank::Three,
    Rank::Four,
    Rank::Five,
    Rank::Six,
    Rank::Seven,
    Rank::Eight,
    Rank::Nine,
    Rank::Ten,
    Rank::Jack,
    Rank::Queen,
    Rank::King,
    Rank::Ace,
];

pub(crate) fn windows(width: usize) -> impl DoubleEndedIterator<Item = (u8, &'static [Rank])> {
    let count = CYCLIC_RANKS.len() + 1 - width;
    (0..count).map(move |start| {
        (
            u8::try_from(start).expect("sequence index fits in u8"),
            &CYCLIC_RANKS[start..start + width],
        )
    })
}

pub(crate) fn high_rank(sequence: &[Rank]) -> Rank {
    *sequence.last().expect("a sequence is never empty")
}
