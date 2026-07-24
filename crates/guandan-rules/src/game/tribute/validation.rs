use std::collections::BTreeSet;

use crate::game::tribute::TributeError;
use crate::movement::strength::rank_strength;
use crate::{Card, Hand, Rank, Seat, CARDS_PER_PLAYER, PLAYER_COUNT};

pub(super) fn validate_deal(hands: &[Hand; PLAYER_COUNT]) -> Result<(), TributeError> {
    let mut seen = BTreeSet::new();
    for seat in Seat::ALL {
        let hand = &hands[seat.index()];
        if hand.len() != CARDS_PER_PLAYER {
            return Err(TributeError::InvalidHandSize {
                seat,
                actual: hand.len(),
            });
        }
        for card in hand.cards() {
            if !seen.insert(card) {
                return Err(TributeError::DuplicatePhysicalCard(card));
            }
        }
    }
    Ok(())
}

pub(super) fn validate_offers(
    hands: &[Hand; PLAYER_COUNT],
    givers: &[Seat],
    offers: &[(Seat, Card)],
    level: Rank,
) -> Result<(), TributeError> {
    for (index, (seat, card)) in offers.iter().enumerate() {
        if offers[..index].iter().any(|(existing, _)| existing == seat) {
            return Err(TributeError::DuplicateSeat(*seat));
        }
        if !givers.contains(seat) {
            return Err(TributeError::UnexpectedSeat(*seat));
        }
        let hand = &hands[seat.index()];
        if !hand.contains(*card) {
            return Err(TributeError::CardNotOwned {
                seat: *seat,
                card: *card,
            });
        }
        if card.is_wild(level) {
            return Err(TributeError::WildcardOffer(*card));
        }
        let highest = hand
            .cards()
            .filter(|candidate| !candidate.is_wild(level))
            .map(|candidate| rank_strength(candidate.rank(), level))
            .max()
            .expect("a dealt hand has non-wild cards");
        if rank_strength(card.rank(), level) != highest {
            return Err(TributeError::OfferIsNotHighest {
                seat: *seat,
                card: *card,
            });
        }
    }
    Ok(())
}
