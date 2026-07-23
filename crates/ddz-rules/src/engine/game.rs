use ddz_core::{
    CardPlayState, DealState, DoublingState, GameAction, GameEvent, GameState,
    LandlordSelectionState, Observation, Phase, RankCounts, RevealState, Seat, SeatMap, StakeState,
};

use super::{automatic, invariants, legal, observe, transition, GameError};
use crate::{deal_plan_for_attempt, first_player_for_attempt, EconomyContext, RuleConfig, RuleProfile};

/// One authoritative game or multi-attempt Huanle match.
#[derive(Clone, Debug)]
pub struct Game {
    pub(crate) rules: RuleConfig,
    pub(crate) match_seed: u64,
    pub(crate) economy: EconomyContext,
    pub(crate) state: GameState,
}

impl Game {
    pub fn new_huanle(
        rules: RuleConfig,
        match_seed: u64,
        economy: EconomyContext,
    ) -> Result<Self, GameError> {
        rules.validate().map_err(GameError::RuleConfig)?;
        if rules.profile != RuleProfile::HuanleClassic {
            return Err(GameError::WrongProfile {
                expected: RuleProfile::HuanleClassic,
                actual: rules.profile,
            });
        }
        let first_player = first_player_for_attempt(match_seed, 0);
        let state = GameState {
            rule_config_id: rules.rule_config_id,
            phase: Phase::PreDeal,
            current_player: Some(first_player),
            deal: DealState::new(
                0,
                deal_plan_for_attempt(match_seed, 0).map_err(GameError::Deal)?,
            ),
            hands: SeatMap::new([RankCounts::empty(); 3]),
            reveal: RevealState::hidden(),
            landlord_selection: LandlordSelectionState::NotStarted { first_player },
            doubling: if rules.doubling.enabled {
                DoublingState::NotStarted
            } else {
                DoublingState::Disabled
            },
            stake: StakeState::new(rules.settlement.base_unit),
            card_play: CardPlayState::empty(),
            history: Vec::new(),
            outcome: None,
        };
        let mut game = Self {
            rules,
            match_seed,
            economy,
            state,
        };
        automatic::advance(&mut game)?;
        game.state.validate().map_err(GameError::State)?;
        invariants::validate(&game.state, &game.rules).map_err(GameError::RuleState)?;
        Ok(game)
    }

    pub fn new_post_bid(
        rules: RuleConfig,
        match_seed: u64,
        landlord: Seat,
    ) -> Result<Self, GameError> {
        rules.validate().map_err(GameError::RuleConfig)?;
        if rules.profile != RuleProfile::DouzeroPostBid {
            return Err(GameError::WrongProfile {
                expected: RuleProfile::DouzeroPostBid,
                actual: rules.profile,
            });
        }
        let plan = deal_plan_for_attempt(match_seed, 0).map_err(GameError::Deal)?;
        let mut deal = DealState::new(0, plan);
        deal.rounds_dealt = ddz_core::DEAL_ROUNDS;
        let mut hands = deal.plan.final_hands();
        hands[landlord] = hands[landlord]
            .checked_add(deal.plan.bottom_counts())
            .map_err(GameError::RankCounts)?;
        let state = GameState {
            rule_config_id: rules.rule_config_id,
            phase: Phase::CardPlay,
            current_player: Some(landlord),
            deal,
            hands,
            reveal: RevealState::hidden(),
            landlord_selection: LandlordSelectionState::PostBid { landlord },
            doubling: DoublingState::Disabled,
            stake: StakeState::new(rules.settlement.base_unit),
            card_play: CardPlayState::empty(),
            history: Vec::new(),
            outcome: None,
        };
        state.validate().map_err(GameError::State)?;
        invariants::validate(&state, &rules).map_err(GameError::RuleState)?;
        Ok(Self {
            rules,
            match_seed,
            economy: EconomyContext::unlimited(),
            state,
        })
    }

    #[must_use]
    pub const fn rules(&self) -> &RuleConfig {
        &self.rules
    }

    #[must_use]
    pub const fn state(&self) -> &GameState {
        &self.state
    }

    #[must_use]
    pub const fn match_seed(&self) -> u64 {
        self.match_seed
    }

    #[must_use]
    pub const fn economy(&self) -> EconomyContext {
        self.economy
    }

    pub fn terminal_reward(&self, seat: Seat) -> Result<i64, GameError> {
        crate::terminal_reward(&self.state, seat, &self.rules).map_err(GameError::Settlement)
    }

    #[must_use]
    pub fn into_state(self) -> GameState {
        self.state
    }

    pub fn legal_actions(&self) -> Result<Vec<GameAction>, GameError> {
        legal::legal_actions(self)
    }

    pub fn observe(&self, observer: Seat) -> Result<Observation, GameError> {
        observe::observation(self, observer)
    }

    /// Build the current information-safe observation without cloning public history.
    ///
    /// Batch inference uses this fixed-width view and obtains history through a separate API.
    pub fn observe_without_history(&self, observer: Seat) -> Result<Observation, GameError> {
        observe::observation_without_history(self, observer)
    }

    pub fn step(&mut self, actor: Seat, action: GameAction) -> Result<StepResult, GameError> {
        let snapshot = self.state.clone();
        self.execute_transaction(actor, action, snapshot)
            .map(|(result, _)| result)
    }

    pub fn apply_with_undo(
        &mut self,
        actor: Seat,
        action: GameAction,
    ) -> Result<(StepResult, UndoToken), GameError> {
        let snapshot = self.state.clone();
        let (result, previous) = self.execute_transaction(actor, action, snapshot)?;
        Ok((
            result,
            UndoToken {
                rule_config_id: self.rules.rule_config_id,
                match_seed: self.match_seed,
                expected_history_len: self.state.history.len(),
                previous,
            },
        ))
    }

    fn execute_transaction(
        &mut self,
        actor: Seat,
        action: GameAction,
        snapshot: GameState,
    ) -> Result<(StepResult, GameState), GameError> {
        let history_start = self.state.history.len();
        let phase_before = self.state.phase;
        let result = (|| {
            if self.state.is_terminal() {
                return Err(GameError::Terminal);
            }
            let expected = self
                .state
                .current_player
                .ok_or(GameError::NoCurrentPlayer {
                    phase: self.state.phase,
                })?;
            if actor != expected {
                return Err(GameError::NotCurrentPlayer {
                    expected,
                    actual: actor,
                });
            }
            let legal = self.legal_actions()?;
            if !legal.contains(&action) {
                return Err(GameError::IllegalAction { actor, action });
            }
            transition::apply(self, actor, action)?;
            automatic::advance(self)?;
            self.state.validate().map_err(GameError::State)?;
            invariants::validate(&self.state, &self.rules).map_err(GameError::RuleState)?;
            Ok(StepResult {
                actor,
                action,
                phase_before,
                phase_after: self.state.phase,
                emitted_events: self.state.history[history_start..].to_vec(),
                terminal: self.state.is_terminal(),
            })
        })();
        match result {
            Ok(result) => Ok((result, snapshot)),
            Err(error) => {
                self.state = snapshot;
                Err(error)
            }
        }
    }

    pub fn undo(&mut self, token: UndoToken) -> Result<(), GameError> {
        if token.rule_config_id != self.rules.rule_config_id
            || token.match_seed != self.match_seed
            || token.expected_history_len != self.state.history.len()
        {
            return Err(GameError::UndoTokenMismatch);
        }
        self.state = token.previous;
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StepResult {
    pub actor: Seat,
    pub action: GameAction,
    pub phase_before: Phase,
    pub phase_after: Phase,
    pub emitted_events: Vec<GameEvent>,
    pub terminal: bool,
}

#[derive(Clone, Debug)]
pub struct UndoToken {
    rule_config_id: u32,
    match_seed: u64,
    expected_history_len: usize,
    previous: GameState,
}
