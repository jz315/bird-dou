use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::game::GameError;
use crate::{Seat, Team, PLAYER_COUNT};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RoundOutcome {
    finish_order: [Seat; PLAYER_COUNT],
    winning_team: Team,
    level_advance: u8,
}

impl RoundOutcome {
    pub fn from_finish_order(finish_order: [Seat; PLAYER_COUNT]) -> Result<Self, GameError> {
        if finish_order.into_iter().collect::<BTreeSet<_>>().len() != PLAYER_COUNT {
            return Err(GameError::InvalidFinishOrder);
        }
        let winning_team = finish_order[0].team();
        let partner_position = finish_order
            .iter()
            .position(|seat| seat.team() == winning_team && *seat != finish_order[0])
            .expect("the first finisher has one partner");
        let level_advance = match partner_position {
            1 => 3,
            2 => 2,
            3 => 1,
            _ => unreachable!("partner position is between one and three"),
        };
        Ok(Self {
            finish_order,
            winning_team,
            level_advance,
        })
    }

    pub const fn finish_order(&self) -> &[Seat; PLAYER_COUNT] {
        &self.finish_order
    }

    pub const fn winning_team(&self) -> Team {
        self.winning_team
    }

    pub const fn level_advance(&self) -> u8 {
        self.level_advance
    }
}
