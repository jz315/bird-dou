mod error;
mod hand;
mod physical;
mod rank;
mod seat;

pub use error::{CardError, HandError, SeatError};
pub use hand::Hand;
pub use physical::{all_cards, Card};
pub use rank::{Rank, Suit};
pub use seat::{Seat, Team};

pub const PLAYER_COUNT: usize = 4;
pub const CARD_COUNT: usize = 108;
pub const CARDS_PER_PLAYER: usize = 27;
pub(crate) const CARD_FACE_COUNT: u8 = 54;
