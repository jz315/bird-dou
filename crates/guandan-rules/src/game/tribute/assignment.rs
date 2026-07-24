use serde::{Deserialize, Serialize};

use crate::game::tribute::TributeError;
use crate::{Card, Hand, Rank, Seat, PLAYER_COUNT};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TributeTransfer {
    pub from: Seat,
    pub to: Seat,
    pub card: Card,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TributeAssignment {
    pub(crate) hands: [Hand; PLAYER_COUNT],
    pub(crate) transfers: Vec<TributeTransfer>,
    pub(crate) opening_player: Seat,
}

impl TributeAssignment {
    pub fn transfers(&self) -> &[TributeTransfer] {
        &self.transfers
    }

    pub const fn opening_player(&self) -> Seat {
        self.opening_player
    }

    pub fn apply_returns(
        self,
        return_cards: &[(Seat, Card)],
    ) -> Result<TributeResolution, TributeError> {
        if return_cards.len() != self.transfers.len() {
            return Err(TributeError::WrongReturnCount {
                expected: self.transfers.len(),
                actual: return_cards.len(),
            });
        }
        let mut hands = self.hands;
        for transfer in &self.transfers {
            move_card(&mut hands, transfer.from, transfer.to, transfer.card)?;
        }

        let mut returns = Vec::with_capacity(return_cards.len());
        for (returner, card) in return_cards {
            if returns
                .iter()
                .any(|transfer: &TributeTransfer| transfer.from == *returner)
            {
                return Err(TributeError::DuplicateSeat(*returner));
            }
            let tribute = self
                .transfers
                .iter()
                .find(|transfer| transfer.to == *returner)
                .ok_or(TributeError::UnexpectedSeat(*returner))?;
            if !is_returnable(*card) {
                return Err(TributeError::ReturnAboveTen(*card));
            }
            move_card(&mut hands, *returner, tribute.from, *card)?;
            returns.push(TributeTransfer {
                from: *returner,
                to: tribute.from,
                card: *card,
            });
        }

        Ok(TributeResolution {
            hands,
            opening_player: self.opening_player,
            tributes: self.transfers,
            returns,
        })
    }
}

fn move_card(
    hands: &mut [Hand; PLAYER_COUNT],
    from: Seat,
    to: Seat,
    card: Card,
) -> Result<(), TributeError> {
    if !hands[from.index()].contains(card) {
        return Err(TributeError::CardNotOwned { seat: from, card });
    }
    hands[from.index()].remove_all(&[card]);
    hands[to.index()].insert(card);
    Ok(())
}

fn is_returnable(card: Card) -> bool {
    card.rank()
        .natural_index()
        .is_some_and(|index| index <= Rank::Ten.natural_index().expect("ten is standard"))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TributeResolution {
    hands: [Hand; PLAYER_COUNT],
    opening_player: Seat,
    tributes: Vec<TributeTransfer>,
    returns: Vec<TributeTransfer>,
}

impl TributeResolution {
    pub fn hands(&self) -> &[Hand; PLAYER_COUNT] {
        &self.hands
    }

    pub fn into_hands(self) -> [Hand; PLAYER_COUNT] {
        self.hands
    }

    pub const fn opening_player(&self) -> Seat {
        self.opening_player
    }

    pub fn tributes(&self) -> &[TributeTransfer] {
        &self.tributes
    }

    pub fn returns(&self) -> &[TributeTransfer] {
        &self.returns
    }
}
