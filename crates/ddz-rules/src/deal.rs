//! Deterministic physical-card dealing shared by single and batched wrappers.

use std::error::Error;
use std::fmt::{Display, Formatter};

use ddz_core::{cards_to_rank_counts, CardError, CARD_COUNT};

use crate::{GameInitError, PostBidGame, RuleConfig, RuleProfile};

const PLAYER_COUNT: usize = 3;
const DEALT_CARD_COUNT: usize = 51;

/// Fixed landlord seat of the post-bid reset primitive.
pub const POST_BID_LANDLORD: u8 = 0;
/// Stable identifier of the seeded physical-card shuffle contract.
pub const SHUFFLE_ALGORITHM: &str = "splitmix64_fisher_yates_v1";

/// Shuffle and deal one complete post-bid game deterministically.
///
/// The first 51 shuffled physical cards are dealt round-robin. The final three
/// become public bottom cards and are added to seat 0's hand.
///
/// # Errors
///
/// Returns [`SeededDealError`] if physical-card conversion or authoritative
/// game initialization rejects the generated deal or supplied rules.
pub fn deal_post_bid(seed: u64, rules: RuleConfig) -> Result<PostBidGame, SeededDealError> {
    let deck = shuffled_deck(seed);

    let mut physical_hands = [
        Vec::with_capacity(20),
        Vec::with_capacity(17),
        Vec::with_capacity(17),
    ];
    for (index, card) in deck[..DEALT_CARD_COUNT].iter().copied().enumerate() {
        physical_hands[index % PLAYER_COUNT].push(card);
    }
    let bottom_cards = cards_to_rank_counts(&deck[DEALT_CARD_COUNT..])?;
    physical_hands[usize::from(POST_BID_LANDLORD)].extend_from_slice(&deck[DEALT_CARD_COUNT..]);
    let hands = [
        cards_to_rank_counts(&physical_hands[0])?,
        cards_to_rank_counts(&physical_hands[1])?,
        cards_to_rank_counts(&physical_hands[2])?,
    ];

    PostBidGame::new(hands, bottom_cards, POST_BID_LANDLORD, rules).map_err(SeededDealError::Game)
}

/// Shuffle and deal one complete canonical game before bidding.
///
/// Each seat receives 17 cards, the final three stay in a hidden bottom-card
/// container, and `seed % 3` selects the first bidder reproducibly.
///
/// # Errors
///
/// Returns [`SeededDealError`] if physical-card conversion or complete-game
/// initialization rejects the generated deal or supplied rules.
pub fn deal_complete(seed: u64, rules: RuleConfig) -> Result<PostBidGame, SeededDealError> {
    let deck = shuffled_deck(seed);
    let mut physical_hands = [
        Vec::with_capacity(17),
        Vec::with_capacity(17),
        Vec::with_capacity(17),
    ];
    for (index, card) in deck[..DEALT_CARD_COUNT].iter().copied().enumerate() {
        physical_hands[index % PLAYER_COUNT].push(card);
    }
    let bottom_cards = cards_to_rank_counts(&deck[DEALT_CARD_COUNT..])?;
    let hands = [
        cards_to_rank_counts(&physical_hands[0])?,
        cards_to_rank_counts(&physical_hands[1])?,
        cards_to_rank_counts(&physical_hands[2])?,
    ];
    let first_bidder = u8::try_from(seed % 3).unwrap_or_default();
    PostBidGame::new_complete(hands, bottom_cards, first_bidder, rules)
        .map_err(SeededDealError::Game)
}

/// Dispatch deterministic dealing by the validated named rule profile.
///
/// # Errors
///
/// Returns [`SeededDealError`] from the selected profile's deal constructor.
pub fn deal_game(seed: u64, rules: RuleConfig) -> Result<PostBidGame, SeededDealError> {
    match rules.profile {
        RuleProfile::DouzeroPostBid => deal_post_bid(seed, rules),
        RuleProfile::CanonicalFull => deal_complete(seed, rules),
    }
}

fn shuffled_deck(seed: u64) -> [u8; CARD_COUNT] {
    let mut deck = [0_u8; CARD_COUNT];
    for (card_id, card) in (0_u8..).zip(deck.iter_mut()) {
        *card = card_id;
    }
    shuffle(&mut deck, seed);
    deck
}

fn shuffle(deck: &mut [u8; CARD_COUNT], seed: u64) {
    let mut random = SplitMix64::new(seed);
    for upper_index in (1..CARD_COUNT).rev() {
        let swap_index = random.sample_below(upper_index + 1);
        deck.swap(upper_index, swap_index);
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    const fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn sample_below(&mut self, upper: usize) -> usize {
        let upper = u64::try_from(upper).expect("deck bounds fit in u64");
        let rejection_threshold = upper.wrapping_neg() % upper;
        loop {
            let value = self.next();
            if value >= rejection_threshold {
                return usize::try_from(value % upper).expect("sample is below deck length");
            }
        }
    }
}

/// Failure to construct a seeded post-bid game.
#[derive(Debug)]
pub enum SeededDealError {
    /// A generated physical-card collection failed canonical conversion.
    Cards(CardError),
    /// The authoritative game rejected the deal or rule profile.
    Game(GameInitError),
}

impl Display for SeededDealError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cards(error) => Display::fmt(error, formatter),
            Self::Game(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for SeededDealError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Cards(error) => Some(error),
            Self::Game(error) => Some(error),
        }
    }
}

impl From<CardError> for SeededDealError {
    fn from(error: CardError) -> Self {
        Self::Cards(error)
    }
}
