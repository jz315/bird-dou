use ddz_core::{Seat, SeatMap};

use crate::DoublingRules;

/// Per-match balances used only to decide whether a double action is available.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EconomyContext {
    pub balances: SeatMap<u64>,
}

impl EconomyContext {
    #[must_use]
    pub const fn new(balances: SeatMap<u64>) -> Self {
        Self { balances }
    }

    #[must_use]
    pub const fn unlimited() -> Self {
        Self::new(SeatMap::new([u64::MAX; 3]))
    }

    #[must_use]
    pub fn double_eligible(self, seat: Seat, landlord: Seat, rules: DoublingRules) -> bool {
        if !rules.enabled {
            return false;
        }
        let exceeds = |candidate: Seat| {
            self.balances[candidate] > rules.minimum_balance_exclusive
        };
        if seat == landlord {
            Seat::ALL.into_iter().all(exceeds)
        } else {
            exceeds(seat) && exceeds(landlord)
        }
    }
}

impl Default for EconomyContext {
    fn default() -> Self {
        Self::unlimited()
    }
}
