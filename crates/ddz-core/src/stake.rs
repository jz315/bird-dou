use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpringKind {
    #[default]
    None,
    LandlordSpring,
    FarmerSpring,
}

/// Public common stake factors. Per-seat double choices remain in `DoublingState`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StakeState {
    pub base_unit: u32,
    pub reveal_factor: u32,
    pub rob_exponent: u8,
    pub bomb_exponent: u8,
    pub spring: SpringKind,
}

impl StakeState {
    pub const fn new(base_unit: u32) -> Self {
        Self {
            base_unit,
            reveal_factor: 1,
            rob_exponent: 0,
            bomb_exponent: 0,
            spring: SpringKind::None,
        }
    }

    pub fn common_multiplier(self) -> Result<u64, StakeError> {
        self.validate()?;
        let spring_exponent = if matches!(self.spring, SpringKind::None) {
            0
        } else {
            1
        };
        let power = u32::from(self.rob_exponent)
            + u32::from(self.bomb_exponent)
            + spring_exponent;
        let power_of_two = 1_u64
            .checked_shl(power)
            .ok_or(StakeError::MultiplierOverflow)?;
        u64::from(self.reveal_factor)
            .checked_mul(power_of_two)
            .ok_or(StakeError::MultiplierOverflow)
    }

    pub fn common_stake(self) -> Result<u64, StakeError> {
        self.common_multiplier()?
            .checked_mul(u64::from(self.base_unit))
            .ok_or(StakeError::MultiplierOverflow)
    }

    pub fn validate(self) -> Result<(), StakeError> {
        if self.base_unit == 0 {
            return Err(StakeError::ZeroBaseUnit);
        }
        if self.reveal_factor == 0 {
            return Err(StakeError::ZeroRevealFactor);
        }
        let spring_exponent = if matches!(self.spring, SpringKind::None) {
            0
        } else {
            1
        };
        let _ = u32::from(self.rob_exponent)
            .checked_add(u32::from(self.bomb_exponent))
            .and_then(|value| value.checked_add(spring_exponent))
            .ok_or(StakeError::MultiplierOverflow)?;
        Ok(())
    }
}

impl Default for StakeState {
    fn default() -> Self {
        Self::new(1)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StakeError {
    ZeroBaseUnit,
    ZeroRevealFactor,
    MultiplierOverflow,
}

impl Display for StakeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroBaseUnit => write!(formatter, "base stake unit must be positive"),
            Self::ZeroRevealFactor => write!(formatter, "reveal factor must be positive"),
            Self::MultiplierOverflow => write!(formatter, "stake multiplier overflow"),
        }
    }
}

impl Error for StakeError {}
