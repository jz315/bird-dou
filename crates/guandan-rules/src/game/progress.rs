use serde::{Deserialize, Serialize};

use crate::game::{GameError, RoundOutcome};
use crate::{Rank, Team};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MatchProgress {
    team_levels: [Rank; 2],
    winner: Option<Team>,
}

impl MatchProgress {
    pub fn new(starting_level: Rank) -> Result<Self, GameError> {
        if !starting_level.is_standard() {
            return Err(GameError::InvalidLevel(starting_level));
        }
        Ok(Self {
            team_levels: [starting_level; 2],
            winner: None,
        })
    }

    pub const fn team_level(&self, team: Team) -> Rank {
        self.team_levels[team.index()]
    }

    pub const fn winner(&self) -> Option<Team> {
        self.winner
    }

    pub fn record_round(&mut self, outcome: &RoundOutcome) -> Result<Rank, GameError> {
        if self.winner.is_some() {
            return Err(GameError::MatchComplete);
        }
        let team = outcome.winning_team();
        let current = self.team_level(team);
        if current == Rank::Ace {
            self.winner = Some(team);
            return Ok(current);
        }

        let mut promoted = current;
        for _ in 0..outcome.level_advance() {
            promoted = promoted.next_level().unwrap_or(Rank::Ace);
        }
        self.team_levels[team.index()] = promoted;
        Ok(promoted)
    }
}
