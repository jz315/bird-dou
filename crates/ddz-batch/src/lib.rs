//! Deterministic, transactionally stepped batches of BIRD-Dou environments.

use std::error::Error;
use std::fmt::{Display, Formatter};

use ddz_core::{
    BidAction, DoubleAction, GameAction, Observation, Phase, Role, StateCodecError, StepResult,
    EMPTY_RANK_COUNTS, RANK_COUNT,
};
use ddz_rules::{
    deal_game, GameError, PostBidGame, RuleConfig, RuleConfigError, SeededDealError, UndoError,
    UndoToken,
};

const PLAYER_COUNT: usize = 3;

/// Version of the packed raw-environment buffer protocol.
pub const BATCH_SCHEMA_VERSION: u32 = 1;
/// Sentinel used where no seat is present.
pub const NO_PLAYER: i8 = -1;
/// Sentinel used where no event sequence is present.
pub const NO_EVENT: i64 = -1;
/// Sentinel used where no move kind is present.
pub const NO_MOVE_KIND: u8 = u8::MAX;
/// Sentinel used where no main rank is present.
pub const NO_RANK: u8 = u8::MAX;
/// Sentinel used where no normalized phase action code is present.
pub const NO_ACTION_CODE: u8 = u8::MAX;

/// One owner of multiple authoritative Rust environments.
#[derive(Debug)]
pub struct BatchDdzEnv {
    rules: RuleConfig,
    games: Vec<PostBidGame>,
    initialized: bool,
    legal_cache: Option<Vec<Vec<GameAction>>>,
    last_objective_payoff: Vec<[i32; PLAYER_COUNT]>,
}

impl BatchDdzEnv {
    /// Construct an empty batch for one validated rule configuration.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::RuleConfig`] when `rules` is invalid.
    pub fn new(rules: RuleConfig) -> Result<Self, BatchError> {
        rules.validate().map_err(BatchError::RuleConfig)?;
        Ok(Self {
            rules,
            games: Vec::new(),
            initialized: false,
            legal_cache: None,
            last_objective_payoff: Vec::new(),
        })
    }

    /// Rule configuration shared by every environment in this batch.
    #[must_use]
    pub const fn rules(&self) -> &RuleConfig {
        &self.rules
    }

    /// Number of initialized environments.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.games.len()
    }

    /// Whether [`Self::reset`] has successfully initialized this batch.
    #[must_use]
    pub const fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Whether every initialized environment is terminal.
    #[must_use]
    pub fn all_terminal(&self) -> bool {
        self.initialized && self.games.iter().all(PostBidGame::is_terminal)
    }

    /// Transactionally replace the batch with deterministic seeded deals.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::EmptyBatch`] for no seeds or
    /// [`BatchError::Deal`] if any seeded environment cannot initialize. The
    /// existing batch is unchanged on failure.
    pub fn reset(&mut self, seeds: &[u64]) -> Result<PackedObservation, BatchError> {
        if seeds.is_empty() {
            return Err(BatchError::EmptyBatch);
        }

        let mut games = Vec::with_capacity(seeds.len());
        for (env_index, &seed) in seeds.iter().enumerate() {
            games.push(
                deal_game(seed, self.rules)
                    .map_err(|source| BatchError::Deal { env_index, source })?,
            );
        }

        let previous_games = std::mem::replace(&mut self.games, games);
        let previous_initialized = self.initialized;
        self.initialized = true;
        match self.packed_observation() {
            Ok(observation) => {
                self.legal_cache = None;
                self.last_objective_payoff = vec![[0; PLAYER_COUNT]; seeds.len()];
                Ok(observation)
            }
            Err(error) => {
                self.games = previous_games;
                self.initialized = previous_initialized;
                Err(error)
            }
        }
    }

    /// Pack the current-player observation of every environment.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::Uninitialized`] before reset, a query error for an
    /// invalid authoritative state, or a buffer-size error on integer overflow.
    pub fn packed_observation(&self) -> Result<PackedObservation, BatchError> {
        self.require_initialized()?;
        let history_capacity = self
            .games
            .iter()
            .map(|game| game.state().history.len())
            .sum();
        let mut packed = PackedObservation::with_capacity(self.games.len(), history_capacity);
        for (env_index, game) in self.games.iter().enumerate() {
            let observer = game.state().current_player;
            let observation = game
                .observe(observer)
                .map_err(|source| BatchError::Query { env_index, source })?;
            packed.push(observation, game.is_terminal(), env_index)?;
        }
        Ok(packed)
    }

    /// Generate and pack the stable legal-action ranges for every environment.
    ///
    /// Terminal environments contribute empty ranges. The generated actions
    /// are cached until the next successful reset or step.
    ///
    /// # Errors
    ///
    /// Returns a query, unsupported-action, or buffer-size error.
    pub fn legal_actions_packed(&mut self) -> Result<PackedActions, BatchError> {
        self.ensure_legal_cache()?;
        let cache = self
            .legal_cache
            .as_ref()
            .ok_or(BatchError::InternalInvariant(
                "legal cache was not populated",
            ))?;
        PackedActions::from_cache(cache)
    }

    /// Apply one local legal-action index per environment.
    ///
    /// Active environments require a non-negative local index into their range
    /// from [`Self::legal_actions_packed`]. Already-terminal environments require
    /// `-1`. All indices are validated before mutation, and any later failure
    /// rolls back every earlier environment in the call.
    ///
    /// # Errors
    ///
    /// Returns an input, query, transition, rollback, unsupported-action, or
    /// buffer-size error. No environment advances when an error is returned.
    pub fn step_packed(&mut self, action_indices: &[i64]) -> Result<PackedStepResult, BatchError> {
        self.require_initialized()?;
        if action_indices.len() != self.games.len() {
            return Err(BatchError::BatchSizeMismatch {
                field: "action_indices",
                expected: self.games.len(),
                actual: action_indices.len(),
            });
        }
        self.ensure_legal_cache()?;

        let selected = self.validate_action_indices(action_indices)?;
        let mut applied = Vec::<(usize, UndoToken)>::with_capacity(self.games.len());
        let mut results = vec![None; self.games.len()];
        for (env_index, action) in selected.into_iter().enumerate() {
            let Some(action) = action else {
                continue;
            };
            match self.games[env_index].step_with_undo(&action) {
                Ok((result, undo)) => {
                    applied.push((env_index, undo));
                    results[env_index] = Some(result);
                }
                Err(source) => {
                    self.rollback(&applied)?;
                    return Err(BatchError::Step { env_index, source });
                }
            }
        }

        let mut objective_payoff = self.last_objective_payoff.clone();
        for (env_index, result) in results.iter().enumerate() {
            if let Some(result) = result {
                objective_payoff[env_index] = result.objective_payoff;
            }
        }
        let output = self.build_step_result(&results, &objective_payoff);
        match output {
            Ok(output) => {
                self.last_objective_payoff = objective_payoff;
                self.legal_cache = None;
                Ok(output)
            }
            Err(error) => {
                self.rollback(&applied)?;
                Err(error)
            }
        }
    }

    /// Serialize every authoritative state in stable environment order.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::Uninitialized`] before reset or a codec error
    /// associated with the failing environment.
    pub fn serialize_states(&self) -> Result<Vec<Vec<u8>>, BatchError> {
        self.require_initialized()?;
        self.games
            .iter()
            .enumerate()
            .map(|(env_index, game)| {
                game.serialize_state()
                    .map_err(|source| BatchError::Serialize { env_index, source })
            })
            .collect()
    }

    fn require_initialized(&self) -> Result<(), BatchError> {
        if self.initialized {
            Ok(())
        } else {
            Err(BatchError::Uninitialized)
        }
    }

    fn ensure_legal_cache(&mut self) -> Result<(), BatchError> {
        self.require_initialized()?;
        if self.legal_cache.is_none() {
            let mut cache = Vec::with_capacity(self.games.len());
            for (env_index, game) in self.games.iter().enumerate() {
                cache.push(
                    game.legal_actions()
                        .map_err(|source| BatchError::Query { env_index, source })?,
                );
            }
            self.legal_cache = Some(cache);
        }
        Ok(())
    }

    fn validate_action_indices(
        &self,
        action_indices: &[i64],
    ) -> Result<Vec<Option<GameAction>>, BatchError> {
        let cache = self
            .legal_cache
            .as_ref()
            .ok_or(BatchError::InternalInvariant(
                "legal cache was not populated",
            ))?;
        let mut selected = Vec::with_capacity(self.games.len());
        for (env_index, ((game, actions), &index)) in self
            .games
            .iter()
            .zip(cache.iter())
            .zip(action_indices.iter())
            .enumerate()
        {
            if game.is_terminal() {
                if index != -1 {
                    return Err(BatchError::TerminalActionIndex { env_index, index });
                }
                selected.push(None);
                continue;
            }
            let local_index = usize::try_from(index)
                .map_err(|_| BatchError::MissingActionIndex { env_index, index })?;
            let action =
                actions
                    .get(local_index)
                    .copied()
                    .ok_or(BatchError::ActionIndexOutOfRange {
                        env_index,
                        index,
                        legal_count: actions.len(),
                    })?;
            selected.push(Some(action));
        }
        Ok(selected)
    }

    fn rollback(&mut self, applied: &[(usize, UndoToken)]) -> Result<(), BatchError> {
        for (env_index, undo) in applied.iter().rev() {
            self.games[*env_index]
                .undo(undo)
                .map_err(|source| BatchError::Rollback {
                    env_index: *env_index,
                    source,
                })?;
        }
        Ok(())
    }

    fn build_step_result(
        &self,
        results: &[Option<StepResult>],
        objective_payoff: &[[i32; PLAYER_COUNT]],
    ) -> Result<PackedStepResult, BatchError> {
        let mut packed = PackedStepResult::with_capacity(self.games.len());
        for (env_index, ((game, result), objective)) in self
            .games
            .iter()
            .zip(results.iter())
            .zip(objective_payoff.iter())
            .enumerate()
        {
            packed.push(env_index, game, result.as_ref(), *objective);
        }
        packed.observation = self.packed_observation()?;
        Ok(packed)
    }
}

/// Structure-of-arrays representation of current-player observations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedObservation {
    /// Packed protocol version.
    pub schema_version: u32,
    /// Number of environments represented by every leading batch dimension.
    pub batch_size: usize,
    /// Phase codes: bidding 0, doubling 1, card play 2, terminal 3.
    pub phase: Vec<u8>,
    /// Seat receiving each observation.
    pub observer: Vec<u8>,
    /// Role codes: landlord 0, farmer 1.
    pub role: Vec<u8>,
    /// Flattened `[B, 15]` observer hands.
    pub own_hand: Vec<u8>,
    /// Flattened `[B, 3, 15]` public played-card counts.
    pub public_played: Vec<u8>,
    /// Flattened `[B, 15]` public bottom-card counts.
    pub public_bottom_cards: Vec<u8>,
    /// Flattened `[B, 15]` unknown opponent-card pools.
    pub unknown_pool: Vec<u8>,
    /// Flattened `[B, 3]` current hand sizes.
    pub cards_left: Vec<u8>,
    /// Acting or winning seat for every environment.
    pub current_player: Vec<u8>,
    /// Landlord seat or [`NO_PLAYER`].
    pub landlord: Vec<i8>,
    /// Whether a current target move exists.
    pub last_non_pass_valid: Vec<u8>,
    /// Flattened `[B, 15]` current target cards.
    pub last_non_pass_cards: Vec<u8>,
    /// Current target [`ddz_core::MoveKind`] numeric tag or [`NO_MOVE_KIND`].
    pub last_non_pass_kind: Vec<u8>,
    /// Current target main rank or [`NO_RANK`].
    pub last_non_pass_main_rank: Vec<u8>,
    /// Current target chain length.
    pub last_non_pass_chain_len: Vec<u8>,
    /// Current target total card count.
    pub last_non_pass_total_cards: Vec<u8>,
    /// Consecutive Pass count after the current target.
    pub consecutive_passes: Vec<u8>,
    /// Base-two score multiplier exponent.
    pub multiplier_exp: Vec<u8>,
    /// Number of bombs and rockets played.
    pub bomb_count: Vec<u8>,
    /// Terminal mask stored as compact 0/1 bytes.
    pub terminal: Vec<u8>,
    /// Ragged public-history offsets of length `B + 1`.
    pub history_offsets: Vec<i64>,
    /// History sequence numbers.
    pub history_sequence: Vec<u32>,
    /// History actors.
    pub history_actor: Vec<u8>,
    /// History action phases using [`phase_code`].
    pub history_phase: Vec<u8>,
    /// Phase-local normalized history action codes.
    pub history_action_code: Vec<u8>,
    /// Flattened `[H, 15]` history move cards.
    pub history_cards: Vec<u8>,
    /// History [`ddz_core::MoveKind`] numeric tags.
    pub history_kind: Vec<u8>,
    /// History move main ranks.
    pub history_main_rank: Vec<u8>,
    /// History move chain lengths.
    pub history_chain_len: Vec<u8>,
    /// History move total card counts.
    pub history_total_cards: Vec<u8>,
}

impl PackedObservation {
    fn with_capacity(batch_size: usize, history_capacity: usize) -> Self {
        Self {
            schema_version: BATCH_SCHEMA_VERSION,
            batch_size,
            phase: Vec::with_capacity(batch_size),
            observer: Vec::with_capacity(batch_size),
            role: Vec::with_capacity(batch_size),
            own_hand: Vec::with_capacity(batch_size * RANK_COUNT),
            public_played: Vec::with_capacity(batch_size * PLAYER_COUNT * RANK_COUNT),
            public_bottom_cards: Vec::with_capacity(batch_size * RANK_COUNT),
            unknown_pool: Vec::with_capacity(batch_size * RANK_COUNT),
            cards_left: Vec::with_capacity(batch_size * PLAYER_COUNT),
            current_player: Vec::with_capacity(batch_size),
            landlord: Vec::with_capacity(batch_size),
            last_non_pass_valid: Vec::with_capacity(batch_size),
            last_non_pass_cards: Vec::with_capacity(batch_size * RANK_COUNT),
            last_non_pass_kind: Vec::with_capacity(batch_size),
            last_non_pass_main_rank: Vec::with_capacity(batch_size),
            last_non_pass_chain_len: Vec::with_capacity(batch_size),
            last_non_pass_total_cards: Vec::with_capacity(batch_size),
            consecutive_passes: Vec::with_capacity(batch_size),
            multiplier_exp: Vec::with_capacity(batch_size),
            bomb_count: Vec::with_capacity(batch_size),
            terminal: Vec::with_capacity(batch_size),
            history_offsets: vec![0],
            history_sequence: Vec::with_capacity(history_capacity),
            history_actor: Vec::with_capacity(history_capacity),
            history_phase: Vec::with_capacity(history_capacity),
            history_action_code: Vec::with_capacity(history_capacity),
            history_cards: Vec::with_capacity(history_capacity * RANK_COUNT),
            history_kind: Vec::with_capacity(history_capacity),
            history_main_rank: Vec::with_capacity(history_capacity),
            history_chain_len: Vec::with_capacity(history_capacity),
            history_total_cards: Vec::with_capacity(history_capacity),
        }
    }

    fn push(
        &mut self,
        observation: Observation,
        terminal: bool,
        _env_index: usize,
    ) -> Result<(), BatchError> {
        self.phase.push(phase_code(observation.phase));
        self.observer.push(observation.observer);
        self.role.push(role_code(observation.role));
        self.own_hand.extend_from_slice(&observation.own_hand);
        for played in observation.public_played {
            self.public_played.extend_from_slice(&played);
        }
        self.public_bottom_cards
            .extend_from_slice(&observation.public_bottom_cards);
        self.unknown_pool
            .extend_from_slice(&observation.unknown_pool);
        self.cards_left.extend_from_slice(&observation.cards_left);
        self.current_player.push(observation.current_player);
        self.landlord
            .push(observation.landlord.map_or(NO_PLAYER, |seat| {
                i8::try_from(seat).expect("seat fits in i8")
            }));
        if let Some(target) = observation.last_non_pass {
            self.last_non_pass_valid.push(1);
            self.last_non_pass_cards.extend_from_slice(target.cards());
            self.last_non_pass_kind.push(u8::from(target.kind()));
            self.last_non_pass_main_rank.push(target.main_rank());
            self.last_non_pass_chain_len.push(target.chain_len());
            self.last_non_pass_total_cards.push(target.total_cards());
        } else {
            self.last_non_pass_valid.push(0);
            self.last_non_pass_cards
                .extend_from_slice(&EMPTY_RANK_COUNTS);
            self.last_non_pass_kind.push(NO_MOVE_KIND);
            self.last_non_pass_main_rank.push(NO_RANK);
            self.last_non_pass_chain_len.push(0);
            self.last_non_pass_total_cards.push(0);
        }
        self.consecutive_passes.push(observation.consecutive_passes);
        self.multiplier_exp.push(observation.multiplier_exp);
        self.bomb_count.push(observation.bomb_count);
        self.terminal.push(u8::from(terminal));

        for event in observation.history {
            self.history_sequence.push(event.sequence);
            self.history_actor.push(event.actor);
            self.history_phase.push(game_action_phase(event.action));
            self.history_action_code
                .push(game_action_code(event.action));
            if let GameAction::Play(played_move) = event.action {
                self.history_cards.extend_from_slice(played_move.cards());
                self.history_kind.push(u8::from(played_move.kind()));
                self.history_main_rank.push(played_move.main_rank());
                self.history_chain_len.push(played_move.chain_len());
                self.history_total_cards.push(played_move.total_cards());
            } else {
                self.history_cards.extend_from_slice(&EMPTY_RANK_COUNTS);
                self.history_kind.push(NO_MOVE_KIND);
                self.history_main_rank.push(NO_RANK);
                self.history_chain_len.push(0);
                self.history_total_cards.push(0);
            }
        }
        self.history_offsets
            .push(buffer_index(self.history_kind.len(), "history_offsets")?);
        Ok(())
    }
}

/// Ragged structure-of-arrays representation of all current legal actions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedActions {
    /// Packed protocol version.
    pub schema_version: u32,
    /// Number of represented environments.
    pub batch_size: usize,
    /// Per-environment action offsets of length `B + 1`.
    pub offsets: Vec<i64>,
    /// Environment index for each flat action.
    pub state_index: Vec<i64>,
    /// Phase code for each candidate action.
    pub phase: Vec<u8>,
    /// Phase-local normalized candidate action code.
    pub action_code: Vec<u8>,
    /// Flattened `[M, 15]` rank counts.
    pub cards: Vec<u8>,
    /// [`ddz_core::MoveKind`] numeric tag for each flat action.
    pub kind: Vec<u8>,
    /// Main rank for each flat action.
    pub main_rank: Vec<u8>,
    /// Chain length for each flat action.
    pub chain_len: Vec<u8>,
    /// Total card count for each flat action.
    pub total_cards: Vec<u8>,
}

impl PackedActions {
    fn from_cache(cache: &[Vec<GameAction>]) -> Result<Self, BatchError> {
        let action_capacity = cache.iter().map(Vec::len).sum();
        let mut packed = Self {
            schema_version: BATCH_SCHEMA_VERSION,
            batch_size: cache.len(),
            offsets: vec![0],
            state_index: Vec::with_capacity(action_capacity),
            phase: Vec::with_capacity(action_capacity),
            action_code: Vec::with_capacity(action_capacity),
            cards: Vec::with_capacity(action_capacity * RANK_COUNT),
            kind: Vec::with_capacity(action_capacity),
            main_rank: Vec::with_capacity(action_capacity),
            chain_len: Vec::with_capacity(action_capacity),
            total_cards: Vec::with_capacity(action_capacity),
        };
        for (env_index, actions) in cache.iter().enumerate() {
            for action in actions {
                packed
                    .state_index
                    .push(buffer_index(env_index, "action state_index")?);
                packed.phase.push(game_action_phase(*action));
                packed.action_code.push(game_action_code(*action));
                if let GameAction::Play(played_move) = *action {
                    packed.cards.extend_from_slice(played_move.cards());
                    packed.kind.push(u8::from(played_move.kind()));
                    packed.main_rank.push(played_move.main_rank());
                    packed.chain_len.push(played_move.chain_len());
                    packed.total_cards.push(played_move.total_cards());
                } else {
                    packed.cards.extend_from_slice(&EMPTY_RANK_COUNTS);
                    packed.kind.push(NO_MOVE_KIND);
                    packed.main_rank.push(NO_RANK);
                    packed.chain_len.push(0);
                    packed.total_cards.push(0);
                }
            }
            packed
                .offsets
                .push(buffer_index(packed.kind.len(), "action offsets")?);
        }
        Ok(packed)
    }
}

/// Packed result and next observations for one transactional batch step.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedStepResult {
    /// Packed protocol version.
    pub schema_version: u32,
    /// Number of represented environments.
    pub batch_size: usize,
    /// Whether each environment applied an action in this call.
    pub acted: Vec<u8>,
    /// Event sequence or [`NO_EVENT`] for an already-terminal environment.
    pub event_sequence: Vec<i64>,
    /// Event actor or [`NO_PLAYER`].
    pub event_actor: Vec<i8>,
    /// Flattened `[B, 15]` applied action cards, zero when not acted.
    pub action_cards: Vec<u8>,
    /// Applied action phase or [`NO_ACTION_CODE`] for a no-op.
    pub action_phase: Vec<u8>,
    /// Applied phase-local action code or [`NO_ACTION_CODE`] for a no-op.
    pub action_code: Vec<u8>,
    /// Applied action kind or [`NO_MOVE_KIND`].
    pub action_kind: Vec<u8>,
    /// Applied action main rank or [`NO_RANK`].
    pub action_main_rank: Vec<u8>,
    /// Applied action chain length.
    pub action_chain_len: Vec<u8>,
    /// Applied action total card count.
    pub action_total_cards: Vec<u8>,
    /// Next actor or [`NO_PLAYER`] after terminal transition/no-op.
    pub next_player: Vec<i8>,
    /// Current terminal mask.
    pub terminal: Vec<u8>,
    /// Flattened `[B, 3]` raw zero-sum payoffs.
    pub raw_payoff: Vec<i32>,
    /// Flattened `[B, 3]` selected objective payoffs.
    pub objective_payoff: Vec<i32>,
    /// Current-player observations after every successful transition.
    pub observation: PackedObservation,
}

impl PackedStepResult {
    fn with_capacity(batch_size: usize) -> Self {
        Self {
            schema_version: BATCH_SCHEMA_VERSION,
            batch_size,
            acted: Vec::with_capacity(batch_size),
            event_sequence: Vec::with_capacity(batch_size),
            event_actor: Vec::with_capacity(batch_size),
            action_cards: Vec::with_capacity(batch_size * RANK_COUNT),
            action_phase: Vec::with_capacity(batch_size),
            action_code: Vec::with_capacity(batch_size),
            action_kind: Vec::with_capacity(batch_size),
            action_main_rank: Vec::with_capacity(batch_size),
            action_chain_len: Vec::with_capacity(batch_size),
            action_total_cards: Vec::with_capacity(batch_size),
            next_player: Vec::with_capacity(batch_size),
            terminal: Vec::with_capacity(batch_size),
            raw_payoff: Vec::with_capacity(batch_size * PLAYER_COUNT),
            objective_payoff: Vec::with_capacity(batch_size * PLAYER_COUNT),
            observation: PackedObservation::with_capacity(0, 0),
        }
    }

    fn push(
        &mut self,
        _env_index: usize,
        game: &PostBidGame,
        result: Option<&StepResult>,
        objective_payoff: [i32; PLAYER_COUNT],
    ) {
        if let Some(result) = result {
            self.acted.push(1);
            self.event_sequence.push(i64::from(result.event.sequence));
            self.event_actor
                .push(i8::try_from(result.event.actor).expect("validated actor fits in i8"));
            self.push_action(result.event.action);
            self.next_player
                .push(result.next_player.map_or(NO_PLAYER, |seat| {
                    i8::try_from(seat).expect("validated seat fits in i8")
                }));
            self.raw_payoff.extend_from_slice(&result.raw_payoff);
        } else {
            self.acted.push(0);
            self.event_sequence.push(NO_EVENT);
            self.event_actor.push(NO_PLAYER);
            self.action_cards.extend_from_slice(&EMPTY_RANK_COUNTS);
            self.action_phase.push(NO_ACTION_CODE);
            self.action_code.push(NO_ACTION_CODE);
            self.action_kind.push(NO_MOVE_KIND);
            self.action_main_rank.push(NO_RANK);
            self.action_chain_len.push(0);
            self.action_total_cards.push(0);
            self.next_player.push(NO_PLAYER);
            self.raw_payoff.extend_from_slice(&game.state().raw_payoff);
        }
        self.terminal.push(u8::from(game.is_terminal()));
        self.objective_payoff.extend_from_slice(&objective_payoff);
    }

    fn push_action(&mut self, action: GameAction) {
        self.action_phase.push(game_action_phase(action));
        self.action_code.push(game_action_code(action));
        if let GameAction::Play(played_move) = action {
            self.action_cards.extend_from_slice(played_move.cards());
            self.action_kind.push(u8::from(played_move.kind()));
            self.action_main_rank.push(played_move.main_rank());
            self.action_chain_len.push(played_move.chain_len());
            self.action_total_cards.push(played_move.total_cards());
        } else {
            self.action_cards.extend_from_slice(&EMPTY_RANK_COUNTS);
            self.action_kind.push(NO_MOVE_KIND);
            self.action_main_rank.push(NO_RANK);
            self.action_chain_len.push(0);
            self.action_total_cards.push(0);
        }
    }
}

const fn game_action_phase(action: GameAction) -> u8 {
    match action {
        GameAction::Bid(_) => 0,
        GameAction::Double(_) => 1,
        GameAction::Play(_) => 2,
    }
}

fn game_action_code(action: GameAction) -> u8 {
    match action {
        GameAction::Bid(BidAction::Pass) | GameAction::Double(DoubleAction::Decline) => 0,
        GameAction::Bid(BidAction::Score(score)) => score,
        GameAction::Bid(BidAction::Call) => 4,
        GameAction::Bid(BidAction::Rob) => 5,
        GameAction::Double(DoubleAction::Double) => 1,
        GameAction::Play(played_move) => u8::from(played_move.kind()),
    }
}

const fn phase_code(phase: Phase) -> u8 {
    match phase {
        Phase::Bidding => 0,
        Phase::Doubling => 1,
        Phase::CardPlay => 2,
        Phase::Terminal => 3,
    }
}

const fn role_code(role: Role) -> u8 {
    match role {
        Role::Unassigned => 2,
        Role::Landlord => 0,
        Role::Farmer => 1,
    }
}

fn buffer_index(value: usize, buffer: &'static str) -> Result<i64, BatchError> {
    i64::try_from(value).map_err(|_| BatchError::BufferTooLarge { buffer })
}

/// Failures from deterministic batch construction, queries, and transitions.
#[derive(Debug)]
pub enum BatchError {
    /// Shared rule configuration is invalid.
    RuleConfig(RuleConfigError),
    /// At least one seed is required.
    EmptyBatch,
    /// A state-dependent operation was called before reset.
    Uninitialized,
    /// One seeded deal failed before reset could commit.
    Deal {
        /// Environment position in the requested batch.
        env_index: usize,
        /// Seeded-deal failure.
        source: SeededDealError,
    },
    /// A per-environment input did not match the batch size.
    BatchSizeMismatch {
        /// Input field name.
        field: &'static str,
        /// Required length.
        expected: usize,
        /// Supplied length.
        actual: usize,
    },
    /// An active environment received a negative action index.
    MissingActionIndex {
        /// Environment position.
        env_index: usize,
        /// Rejected index.
        index: i64,
    },
    /// A local action index exceeded the environment's legal range.
    ActionIndexOutOfRange {
        /// Environment position.
        env_index: usize,
        /// Rejected index.
        index: i64,
        /// Number of available actions.
        legal_count: usize,
    },
    /// An already-terminal environment received an index other than `-1`.
    TerminalActionIndex {
        /// Environment position.
        env_index: usize,
        /// Rejected index.
        index: i64,
    },
    /// Authoritative observation or legal-action query failed.
    Query {
        /// Environment position.
        env_index: usize,
        /// Game query failure.
        source: GameError,
    },
    /// A validated cached action unexpectedly failed to apply.
    Step {
        /// Environment position.
        env_index: usize,
        /// Transition failure.
        source: GameError,
    },
    /// Transactional rollback failed, indicating an internal invariant defect.
    Rollback {
        /// Environment position.
        env_index: usize,
        /// Undo failure.
        source: UndoError,
    },
    /// Post-bid packing encountered a non-card-play action.
    UnsupportedAction {
        /// Environment position.
        env_index: usize,
    },
    /// A ragged offset cannot fit the signed 64-bit transport type.
    BufferTooLarge {
        /// Buffer whose offset overflowed.
        buffer: &'static str,
    },
    /// State serialization failed for one environment.
    Serialize {
        /// Environment position.
        env_index: usize,
        /// State codec failure.
        source: StateCodecError,
    },
    /// An internal cache/state relationship was violated.
    InternalInvariant(&'static str),
}

impl Display for BatchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RuleConfig(error) => Display::fmt(error, formatter),
            Self::EmptyBatch => write!(formatter, "batch reset requires at least one seed"),
            Self::Uninitialized => write!(formatter, "batch environment must be reset first"),
            Self::Deal { env_index, source } => {
                write!(formatter, "failed to deal environment {env_index}: {source}")
            }
            Self::BatchSizeMismatch {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "{field} length {actual} does not match batch size {expected}"
            ),
            Self::MissingActionIndex { env_index, index } => write!(
                formatter,
                "active environment {env_index} requires a non-negative action index, got {index}"
            ),
            Self::ActionIndexOutOfRange {
                env_index,
                index,
                legal_count,
            } => write!(
                formatter,
                "action index {index} is outside environment {env_index}'s {legal_count} legal actions"
            ),
            Self::TerminalActionIndex { env_index, index } => write!(
                formatter,
                "terminal environment {env_index} requires action index -1, got {index}"
            ),
            Self::Query { env_index, source } => {
                write!(formatter, "environment {env_index} query failed: {source}")
            }
            Self::Step { env_index, source } => {
                write!(formatter, "environment {env_index} step failed: {source}")
            }
            Self::Rollback { env_index, source } => write!(
                formatter,
                "environment {env_index} transactional rollback failed: {source}"
            ),
            Self::UnsupportedAction { env_index } => write!(
                formatter,
                "environment {env_index} produced a non-card-play action in the post-bid batch"
            ),
            Self::BufferTooLarge { buffer } => {
                write!(formatter, "{buffer} exceeds signed 64-bit packed offsets")
            }
            Self::Serialize { env_index, source } => {
                write!(formatter, "environment {env_index} serialization failed: {source}")
            }
            Self::InternalInvariant(reason) => {
                write!(formatter, "batch internal invariant failed: {reason}")
            }
        }
    }
}

impl Error for BatchError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::RuleConfig(error) => Some(error),
            Self::Deal { source, .. } => Some(source),
            Self::Query { source, .. } | Self::Step { source, .. } => Some(source),
            Self::Rollback { source, .. } => Some(source),
            Self::Serialize { source, .. } => Some(source),
            Self::EmptyBatch
            | Self::Uninitialized
            | Self::BatchSizeMismatch { .. }
            | Self::MissingActionIndex { .. }
            | Self::ActionIndexOutOfRange { .. }
            | Self::TerminalActionIndex { .. }
            | Self::UnsupportedAction { .. }
            | Self::BufferTooLarge { .. }
            | Self::InternalInvariant(_) => None,
        }
    }
}
