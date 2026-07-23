//! Deterministic physical dealing.

use std::error::Error;
use std::fmt::{Display, Formatter};

use ddz_core::{CardId, CardIdError, DealPlan, DeckOrder, DeckOrderError, Seat, CARD_COUNT};

pub const SHUFFLE_ALGORITHM: &str = "splitmix64_fisher_yates_v1";
pub const ATTEMPT_SEED_ALGORITHM: &str = "splitmix64_attempt_seed_v1";
pub const FIRST_PLAYER_ALGORITHM: &str = "splitmix64_mod3_v1";

#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const GOLDEN_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(Self::GOLDEN_GAMMA);
        mix64(self.state)
    }
}

#[must_use]
const fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[must_use]
pub const fn derive_attempt_seed(match_seed: u64, attempt: u32) -> u64 {
    mix64(match_seed ^ (attempt as u64).wrapping_mul(SplitMix64::GOLDEN_GAMMA))
}

#[must_use]
pub fn first_player_for_attempt(match_seed: u64, attempt: u32) -> Seat {
    let value = mix64(derive_attempt_seed(match_seed, attempt)) % 3;
    Seat::new(u8::try_from(value).expect("value modulo three fits in u8"))
        .expect("value modulo three is a valid seat")
}

pub fn shuffled_deck(seed: u64) -> Result<DeckOrder, DealError> {
    let mut cards = (0_u8..u8::try_from(CARD_COUNT).expect("54 fits in u8"))
        .map(CardId::new)
        .collect::<Result<Vec<_>, _>>()
        .map_err(DealError::Card)?;
    let mut generator = SplitMix64::new(seed);
    for upper in (1..cards.len()).rev() {
        let modulus = u64::try_from(upper + 1).expect("deck length fits in u64");
        let target = usize::try_from(generator.next() % modulus)
            .expect("shuffle target fits in usize");
        cards.swap(upper, target);
    }
    DeckOrder::try_from(cards).map_err(DealError::Deck)
}

pub fn deal_plan_for_attempt(match_seed: u64, attempt: u32) -> Result<DealPlan, DealError> {
    shuffled_deck(derive_attempt_seed(match_seed, attempt)).map(DealPlan::new)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DealError {
    Card(CardIdError),
    Deck(DeckOrderError),
}

impl Display for DealError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Card(error) => Display::fmt(error, formatter),
            Self::Deck(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for DealError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Card(error) => Some(error),
            Self::Deck(error) => Some(error),
        }
    }
}
