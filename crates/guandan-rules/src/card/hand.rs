use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::{Card, HandError};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Hand(BTreeSet<Card>);

impl Hand {
    pub fn from_cards(cards: impl IntoIterator<Item = Card>) -> Result<Self, HandError> {
        let mut values = BTreeSet::new();
        for card in cards {
            if !values.insert(card) {
                return Err(HandError::DuplicateCard(card));
            }
        }
        Ok(Self(values))
    }

    pub const fn empty() -> Self {
        Self(BTreeSet::new())
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn contains(&self, card: Card) -> bool {
        self.0.contains(&card)
    }

    pub fn contains_all(&self, cards: &[Card]) -> bool {
        cards.iter().all(|card| self.contains(*card))
    }

    pub fn cards(&self) -> impl Iterator<Item = Card> + '_ {
        self.0.iter().copied()
    }

    pub(crate) fn remove_all(&mut self, cards: &[Card]) {
        for card in cards {
            self.0.remove(card);
        }
    }

    pub(crate) fn insert(&mut self, card: Card) {
        self.0.insert(card);
    }
}
