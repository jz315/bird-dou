//! Typed reset and restore inputs.

use ddz_core::{GameState, Seat};
use ddz_rules::EconomyContext;

/// Constructor input for one new batch slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResetSpec {
    /// Complete Huanle match.
    Huanle {
        /// Deterministic match seed shared by all redeal attempts in this match.
        match_seed: u64,
        /// Per-seat balance context used only for double eligibility.
        economy: EconomyContext,
    },
    /// Exact DouZero-compatible post-bid game.
    PostBid {
        /// Deterministic deal seed.
        match_seed: u64,
        /// Fixed landlord.
        landlord: Seat,
    },
}

impl ResetSpec {
    /// Convenience constructor using unlimited balances.
    #[must_use]
    pub const fn huanle(match_seed: u64) -> Self {
        Self::Huanle {
            match_seed,
            economy: EconomyContext::unlimited(),
        }
    }

    /// Convenience constructor for a post-bid game.
    #[must_use]
    pub const fn post_bid(match_seed: u64, landlord: Seat) -> Self {
        Self::PostBid {
            match_seed,
            landlord,
        }
    }

    #[must_use]
    pub const fn match_seed(self) -> u64 {
        match self {
            Self::Huanle { match_seed, .. } | Self::PostBid { match_seed, .. } => match_seed,
        }
    }
}

/// Partial reset of one existing slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SlotReset {
    /// Existing slot index.
    pub slot: usize,
    /// New game constructor input.
    pub spec: ResetSpec,
}

/// Typed restore input after the caller has decoded a state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RestoreSpec {
    /// Match seed used to deterministically replay the state.
    pub match_seed: u64,
    /// Balance context used by Huanle double eligibility.
    pub economy: EconomyContext,
    /// Authoritative state to replay and verify.
    pub state: GameState,
}

/// Encoded restore input for checkpoint and FFI consumers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedRestoreSpec {
    /// Match seed used to deterministically replay the state.
    pub match_seed: u64,
    /// Balance context used by Huanle double eligibility.
    pub economy: EconomyContext,
    /// Versioned `ddz-core` state envelope.
    pub state_bytes: Vec<u8>,
}
