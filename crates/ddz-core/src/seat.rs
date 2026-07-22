use std::error::Error;
use std::fmt::{Display, Formatter};
use std::ops::{Index, IndexMut};

use serde::{Deserialize, Serialize};

use crate::PLAYER_COUNT;

/// Valid player seat in the inclusive range `0..=2`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[repr(transparent)]
#[serde(try_from = "u8", into = "u8")]
pub struct Seat(u8);

impl Seat {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);
    pub const TWO: Self = Self(2);
    pub const ALL: [Self; PLAYER_COUNT] = [Self::ZERO, Self::ONE, Self::TWO];

    pub const fn new(value: u8) -> Result<Self, SeatError> {
        if value < PLAYER_COUNT as u8 {
            Ok(Self(value))
        } else {
            Err(SeatError { value })
        }
    }

    pub const fn value(self) -> u8 {
        self.0
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub const fn next(self) -> Self {
        Self((self.0 + 1) % PLAYER_COUNT as u8)
    }

    pub const fn previous(self) -> Self {
        Self((self.0 + PLAYER_COUNT as u8 - 1) % PLAYER_COUNT as u8)
    }

    pub const fn offset(self, distance: u8) -> Self {
        Self((self.0 + distance % PLAYER_COUNT as u8) % PLAYER_COUNT as u8)
    }

    /// Returns `0` for the same seat, `1` for the next seat and `2` for the previous seat.
    pub const fn relative_to(self, origin: Self) -> u8 {
        (self.0 + PLAYER_COUNT as u8 - origin.0) % PLAYER_COUNT as u8
    }
}

impl TryFrom<u8> for Seat {
    type Error = SeatError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<Seat> for u8 {
    fn from(value: Seat) -> Self {
        value.0
    }
}

impl From<Seat> for usize {
    fn from(value: Seat) -> Self {
        value.index()
    }
}

impl Display for Seat {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SeatError {
    pub value: u8,
}

impl Display for SeatError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "seat {} is outside 0..=2", self.value)
    }
}

impl Error for SeatError {}

/// Exactly one value per seat, indexed by the validated [`Seat`] type.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SeatMap<T>([T; PLAYER_COUNT]);

impl<T> SeatMap<T> {
    pub const fn new(values: [T; PLAYER_COUNT]) -> Self {
        Self(values)
    }

    pub const fn as_array(&self) -> &[T; PLAYER_COUNT] {
        &self.0
    }

    pub fn as_mut_array(&mut self) -> &mut [T; PLAYER_COUNT] {
        &mut self.0
    }

    pub fn into_array(self) -> [T; PLAYER_COUNT] {
        self.0
    }

    pub fn get(&self, seat: Seat) -> &T {
        &self.0[seat.index()]
    }

    pub fn get_mut(&mut self, seat: Seat) -> &mut T {
        &mut self.0[seat.index()]
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = (Seat, &T)> {
        Seat::ALL.into_iter().zip(self.0.iter())
    }

    pub fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = (Seat, &mut T)> {
        Seat::ALL.into_iter().zip(self.0.iter_mut())
    }

    pub fn map<U>(self, mut function: impl FnMut(Seat, T) -> U) -> SeatMap<U> {
        let mut values = self.0.into_iter();
        SeatMap::new(std::array::from_fn(|index| {
            let value = values.next().expect("SeatMap always has three entries");
            function(Seat::ALL[index], value)
        }))
    }

    pub fn from_fn(function: impl FnMut(Seat) -> T) -> Self {
        Self(std::array::from_fn({
            let mut function = function;
            move |index| function(Seat::ALL[index])
        }))
    }
}

impl<T: Copy> Copy for SeatMap<T> {}

impl<T: Default> Default for SeatMap<T> {
    fn default() -> Self {
        Self::from_fn(|_| T::default())
    }
}

impl<T> Index<Seat> for SeatMap<T> {
    type Output = T;

    fn index(&self, seat: Seat) -> &Self::Output {
        self.get(seat)
    }
}

impl<T> IndexMut<Seat> for SeatMap<T> {
    fn index_mut(&mut self, seat: Seat) -> &mut Self::Output {
        self.get_mut(seat)
    }
}

impl<T> From<[T; PLAYER_COUNT]> for SeatMap<T> {
    fn from(values: [T; PLAYER_COUNT]) -> Self {
        Self::new(values)
    }
}

impl<T> From<SeatMap<T>> for [T; PLAYER_COUNT] {
    fn from(values: SeatMap<T>) -> Self {
        values.into_array()
    }
}

/// Compact validated set of player seats.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[repr(transparent)]
#[serde(try_from = "u8", into = "u8")]
pub struct SeatSet(u8);

impl SeatSet {
    const VALID_BITS: u8 = (1 << PLAYER_COUNT) - 1;

    pub const fn empty() -> Self {
        Self(0)
    }

    pub const fn all() -> Self {
        Self(Self::VALID_BITS)
    }

    pub const fn singleton(seat: Seat) -> Self {
        Self(1 << seat.value())
    }

    pub const fn from_bits(bits: u8) -> Result<Self, SeatSetError> {
        if bits & !Self::VALID_BITS == 0 {
            Ok(Self(bits))
        } else {
            Err(SeatSetError { bits })
        }
    }

    pub const fn bits(self) -> u8 {
        self.0
    }

    pub const fn contains(self, seat: Seat) -> bool {
        self.0 & (1 << seat.value()) != 0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn len(self) -> u32 {
        self.0.count_ones()
    }

    pub fn insert(&mut self, seat: Seat) -> bool {
        let before = self.0;
        self.0 |= 1 << seat.value();
        self.0 != before
    }

    pub fn remove(&mut self, seat: Seat) -> bool {
        let before = self.0;
        self.0 &= !(1 << seat.value());
        self.0 != before
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    pub const fn is_subset(self, other: Self) -> bool {
        self.0 & !other.0 == 0
    }

    pub fn iter(self) -> impl Iterator<Item = Seat> {
        Seat::ALL
            .into_iter()
            .filter(move |seat| self.contains(*seat))
    }
}

impl TryFrom<u8> for SeatSet {
    type Error = SeatSetError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Self::from_bits(value)
    }
}

impl From<SeatSet> for u8 {
    fn from(value: SeatSet) -> Self {
        value.bits()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SeatSetError {
    pub bits: u8,
}

impl Display for SeatSetError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "seat-set bit mask {:#010b} contains invalid seats", self.bits)
    }
}

impl Error for SeatSetError {}

/// Stable unique turn order containing at most the three player seats.
#[derive(Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "Vec<Seat>", into = "Vec<Seat>")]
pub struct SeatOrder(Vec<Seat>);

impl SeatOrder {
    pub fn new(seats: impl IntoIterator<Item = Seat>) -> Result<Self, SeatOrderError> {
        Self::try_from(seats.into_iter().collect::<Vec<_>>())
    }

    pub fn as_slice(&self) -> &[Seat] {
        &self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<Seat> {
        self.0.get(index).copied()
    }

    pub fn as_set(&self) -> SeatSet {
        let mut set = SeatSet::empty();
        for seat in &self.0 {
            set.insert(*seat);
        }
        set
    }
}

impl TryFrom<Vec<Seat>> for SeatOrder {
    type Error = SeatOrderError;

    fn try_from(value: Vec<Seat>) -> Result<Self, Self::Error> {
        if value.len() > PLAYER_COUNT {
            return Err(SeatOrderError::TooManySeats { actual: value.len() });
        }
        let mut seen = SeatSet::empty();
        for seat in &value {
            if !seen.insert(*seat) {
                return Err(SeatOrderError::DuplicateSeat { seat: *seat });
            }
        }
        Ok(Self(value))
    }
}

impl From<SeatOrder> for Vec<Seat> {
    fn from(value: SeatOrder) -> Self {
        value.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SeatOrderError {
    TooManySeats { actual: usize },
    DuplicateSeat { seat: Seat },
}

impl Display for SeatOrderError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManySeats { actual } => {
                write!(formatter, "seat order has {actual} entries; maximum is 3")
            }
            Self::DuplicateSeat { seat } => write!(formatter, "seat order repeats seat {seat}"),
        }
    }
}

impl Error for SeatOrderError {}
