use crate::{all_cards, Hand, CARDS_PER_PLAYER, PLAYER_COUNT};

pub(crate) fn deal(seed: u64) -> [Hand; PLAYER_COUNT] {
    let mut cards = all_cards();
    let mut random = SplitMix64::new(seed);
    for upper in (1..cards.len()).rev() {
        let selected = random.bounded(upper + 1);
        cards.swap(upper, selected);
    }

    std::array::from_fn(|seat| {
        let start = seat * CARDS_PER_PLAYER;
        let end = start + CARDS_PER_PLAYER;
        Hand::from_cards(cards[start..end].iter().copied())
            .expect("a shuffled physical deck contains no duplicates")
    })
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn bounded(&mut self, upper: usize) -> usize {
        (self.next() % upper as u64) as usize
    }
}
