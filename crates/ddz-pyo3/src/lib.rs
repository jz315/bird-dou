//! Python bindings for the authoritative BIRD-Dou Rust environment.

use std::fmt::Display;

use ddz_batch::{
    BatchDdzEnv, BatchError, PackedActions, PackedObservation, PackedStepResult,
    BATCH_SCHEMA_VERSION,
};
use ddz_core::{GameAction, RankCounts, RANK_COUNT};
use ddz_rules::{
    deal_game, generate_lead_moves, GameError, PostBidGame, RuleConfig, VersionedRuleConfig,
};
use ddz_search::{minimum_play_groups_many, solve_exact_endgame, ExactSearchConfig};
use numpy::{Element, PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};
use serde::de::DeserializeOwned;
use serde::Serialize;

/// Version of the Python object protocol emitted by this module.
pub const API_SCHEMA_VERSION: u32 = 1;
/// Reproducible shuffle implementation used by Python `PyDdzEnv.reset`.
pub const SHUFFLE_ALGORITHM: &str = ddz_rules::SHUFFLE_ALGORITHM;

/// One deterministic post-bid environment backed entirely by Rust.
#[pyclass(module = "birddou._native")]
pub struct PyDdzEnv {
    game: Option<PostBidGame>,
    seed: Option<u64>,
}

#[pymethods]
impl PyDdzEnv {
    #[new]
    const fn new() -> Self {
        Self {
            game: None,
            seed: None,
        }
    }

    /// Deal and initialize one deterministic game.
    fn reset<'py>(
        &mut self,
        py: Python<'py>,
        seed: u64,
        rule_config: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rules = legacy_rules_from_python(py, rule_config.as_any(), "rule_config")?;

        let game = deal_game(seed, rules).map_err(runtime_error)?;
        let observation = game
            .observe(game.state().current_player)
            .map_err(game_error)?;
        let python_observation = to_python(py, &observation, "observation")?;

        self.game = Some(game);
        self.seed = Some(seed);
        Ok(python_observation)
    }

    /// Initialize a privileged training branch from an explicit complete deal.
    ///
    /// This API is intentionally separate from policy observations: it is used
    /// by Monte Carlo label generation after sampling information-set-consistent
    /// hidden allocations.
    fn reset_complete_deal<'py>(
        &mut self,
        py: Python<'py>,
        hands: &Bound<'py, PyAny>,
        bottom_cards: &Bound<'py, PyAny>,
        first_bidder: u8,
        rule_config: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let hands: [RankCounts; 3] = from_python(py, hands, "complete hands")?;
        let bottom_cards: RankCounts = from_python(py, bottom_cards, "bottom cards")?;
        let rules = legacy_rules_from_python(py, rule_config.as_any(), "rule_config")?;
        let game = PostBidGame::new_complete(hands, bottom_cards, first_bidder, rules)
            .map_err(value_error)?;
        let observation = game
            .observe(game.state().current_player)
            .map_err(game_error)?;
        let python_observation = to_python(py, &observation, "observation")?;
        self.game = Some(game);
        self.seed = None;
        Ok(python_observation)
    }

    /// Restore a serialized authoritative state under an explicitly supplied rule profile.
    fn restore<'py>(
        &mut self,
        py: Python<'py>,
        serialized_state: &Bound<'py, PyBytes>,
        rule_config: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rules = legacy_rules_from_python(py, rule_config.as_any(), "rule_config")?;
        let game = PostBidGame::deserialize_state(serialized_state.as_bytes(), rules)
            .map_err(value_error)?;
        let observation = game
            .observe(game.state().current_player)
            .map_err(game_error)?;
        let python_observation = to_python(py, &observation, "observation")?;

        self.game = Some(game);
        self.seed = None;
        Ok(python_observation)
    }

    /// Restore a root and replace its two hidden hands with a replay-valid sample.
    fn restore_with_hidden_sample<'py>(
        &mut self,
        py: Python<'py>,
        serialized_state: &Bound<'py, PyBytes>,
        rule_config: &Bound<'py, PyDict>,
        observer: u8,
        assignment_a: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rules = legacy_rules_from_python(py, rule_config.as_any(), "rule_config")?;
        let assignment_a: RankCounts = from_python(py, assignment_a, "container-A assignment")?;
        let root = PostBidGame::deserialize_state(serialized_state.as_bytes(), rules)
            .map_err(value_error)?;
        let game = root
            .with_hidden_assignment(observer, assignment_a)
            .map_err(value_error)?;
        let observation = game.observe(observer).map_err(game_error)?;
        let python_observation = to_python(py, &observation, "observation")?;
        self.game = Some(game);
        self.seed = None;
        Ok(python_observation)
    }

    /// Return all canonical legal actions for the current state.
    fn legal_actions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let actions = self.game()?.legal_actions().map_err(game_error)?;
        to_python(py, &actions, "legal actions")
    }

    /// Apply one canonical action returned by [`Self::legal_actions`].
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        action: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let action: GameAction = from_python(py, action, "action")?;
        let result = self.game_mut()?.step(&action).map_err(game_error)?;
        to_python(py, &result, "step result")
    }

    /// Observe the current information set from one seat.
    fn observe<'py>(&self, py: Python<'py>, player: u8) -> PyResult<Bound<'py, PyAny>> {
        let observation = self.game()?.observe(player).map_err(game_error)?;
        to_python(py, &observation, "observation")
    }

    /// Serialize the authoritative state using the versioned E009 envelope.
    fn serialize<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.game()?.serialize_state().map_err(runtime_error)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Seed used by the latest successful reset.
    #[getter]
    const fn seed(&self) -> Option<u64> {
        self.seed
    }

    /// Whether this object currently owns an initialized Rust game.
    #[getter]
    const fn is_initialized(&self) -> bool {
        self.game.is_some()
    }

    /// Current acting seat, or the winning seat after termination.
    #[getter]
    fn current_player(&self) -> PyResult<u8> {
        Ok(self.game()?.state().current_player)
    }

    /// Whether the current game has ended.
    #[getter]
    fn terminal(&self) -> PyResult<bool> {
        Ok(self.game()?.is_terminal())
    }

    fn __repr__(&self) -> String {
        match &self.game {
            Some(game) => format!(
                "PyDdzEnv(seed={:?}, current_player={}, terminal={})",
                self.seed,
                game.state().current_player,
                game.is_terminal()
            ),
            None => "PyDdzEnv(uninitialized)".to_owned(),
        }
    }
}

/// Multiple deterministic Rust environments exposed through packed `NumPy` buffers.
#[pyclass(module = "birddou._native")]
pub struct PyBatchDdzEnv {
    batch: BatchDdzEnv,
}

#[pymethods]
impl PyBatchDdzEnv {
    #[new]
    fn new(py: Python<'_>, rule_config: &Bound<'_, PyDict>) -> PyResult<Self> {
        let rules = legacy_rules_from_python(py, rule_config.as_any(), "rule_config")?;
        let batch = BatchDdzEnv::new(rules).map_err(batch_error)?;
        Ok(Self { batch })
    }

    /// Reset all environments from a contiguous one-dimensional `uint64` seed array.
    #[allow(clippy::needless_pass_by_value)] // Required by PyO3's NumPy extractor.
    fn reset<'py>(
        &mut self,
        py: Python<'py>,
        seeds: PyReadonlyArray1<'py, u64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let seeds = seeds.as_slice().map_err(|error| {
            PyValueError::new_err(format!("seeds must be C-contiguous: {error}"))
        })?;
        let observation = self.batch.reset(seeds).map_err(batch_error)?;
        packed_observation_to_python(py, observation)
    }

    /// Pack current-player observations without advancing any environment.
    fn observe_packed<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let observation = self.batch.packed_observation().map_err(batch_error)?;
        packed_observation_to_python(py, observation)
    }

    /// Return all legal actions as one ragged structure of contiguous arrays.
    fn legal_actions_packed<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let actions = self.batch.legal_actions_packed().map_err(batch_error)?;
        packed_actions_to_python(py, actions)
    }

    /// Apply one local action index per environment and return packed next states.
    #[allow(clippy::needless_pass_by_value)] // Required by PyO3's NumPy extractor.
    fn step_packed<'py>(
        &mut self,
        py: Python<'py>,
        action_indices: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let action_indices = action_indices.as_slice().map_err(|error| {
            PyValueError::new_err(format!("action_indices must be C-contiguous: {error}"))
        })?;
        let result = self
            .batch
            .step_packed(action_indices)
            .map_err(batch_error)?;
        packed_step_result_to_python(py, result)
    }

    /// Number of initialized environments.
    #[getter]
    fn batch_size(&self) -> usize {
        self.batch.batch_size()
    }

    /// Whether reset has successfully initialized the batch.
    #[getter]
    const fn is_initialized(&self) -> bool {
        self.batch.is_initialized()
    }

    /// Whether every initialized environment is terminal.
    #[getter]
    fn all_terminal(&self) -> bool {
        self.batch.all_terminal()
    }

    fn __repr__(&self) -> String {
        if self.batch.is_initialized() {
            format!(
                "PyBatchDdzEnv(batch_size={}, all_terminal={})",
                self.batch.batch_size(),
                self.batch.all_terminal()
            )
        } else {
            "PyBatchDdzEnv(uninitialized)".to_owned()
        }
    }
}

fn packed_observation_to_python(
    py: Python<'_>,
    mut packed: PackedObservation,
) -> PyResult<Bound<'_, PyDict>> {
    let dictionary = PyDict::new(py);
    dictionary.set_item("schema_version", packed.schema_version)?;
    dictionary.set_item("batch_size", packed.batch_size)?;
    set_observation_public_arrays(&dictionary, py, &mut packed)?;
    set_observation_target_arrays(&dictionary, py, &mut packed)?;
    set_observation_history_arrays(&dictionary, py, &mut packed)?;
    Ok(dictionary)
}

fn set_observation_public_arrays(
    dictionary: &Bound<'_, PyDict>,
    py: Python<'_>,
    packed: &mut PackedObservation,
) -> PyResult<()> {
    let batch_size = packed.batch_size;
    set_array1(dictionary, py, "phase", std::mem::take(&mut packed.phase))?;
    set_array1(
        dictionary,
        py,
        "observer",
        std::mem::take(&mut packed.observer),
    )?;
    set_array1(dictionary, py, "role", std::mem::take(&mut packed.role))?;
    set_array2(
        dictionary,
        py,
        "own_hand",
        std::mem::take(&mut packed.own_hand),
        batch_size,
        RANK_COUNT,
    )?;
    set_array3(
        dictionary,
        py,
        "public_played",
        std::mem::take(&mut packed.public_played),
        batch_size,
        3,
        RANK_COUNT,
    )?;
    set_array2(
        dictionary,
        py,
        "public_bottom_cards",
        std::mem::take(&mut packed.public_bottom_cards),
        batch_size,
        RANK_COUNT,
    )?;
    set_array2(
        dictionary,
        py,
        "unknown_pool",
        std::mem::take(&mut packed.unknown_pool),
        batch_size,
        RANK_COUNT,
    )?;
    set_array2(
        dictionary,
        py,
        "cards_left",
        std::mem::take(&mut packed.cards_left),
        batch_size,
        3,
    )?;
    set_array1(
        dictionary,
        py,
        "current_player",
        std::mem::take(&mut packed.current_player),
    )?;
    set_array1(
        dictionary,
        py,
        "landlord",
        std::mem::take(&mut packed.landlord),
    )?;
    Ok(())
}

fn set_observation_target_arrays(
    dictionary: &Bound<'_, PyDict>,
    py: Python<'_>,
    packed: &mut PackedObservation,
) -> PyResult<()> {
    set_array1(
        dictionary,
        py,
        "last_non_pass_valid",
        std::mem::take(&mut packed.last_non_pass_valid),
    )?;
    set_array2(
        dictionary,
        py,
        "last_non_pass_rank_counts",
        std::mem::take(&mut packed.last_non_pass_cards),
        packed.batch_size,
        RANK_COUNT,
    )?;
    set_array1(
        dictionary,
        py,
        "last_non_pass_kind",
        std::mem::take(&mut packed.last_non_pass_kind),
    )?;
    set_array1(
        dictionary,
        py,
        "last_non_pass_main_rank",
        std::mem::take(&mut packed.last_non_pass_main_rank),
    )?;
    set_array1(
        dictionary,
        py,
        "last_non_pass_chain_len",
        std::mem::take(&mut packed.last_non_pass_chain_len),
    )?;
    set_array1(
        dictionary,
        py,
        "last_non_pass_total_cards",
        std::mem::take(&mut packed.last_non_pass_total_cards),
    )?;
    set_array1(
        dictionary,
        py,
        "consecutive_passes",
        std::mem::take(&mut packed.consecutive_passes),
    )?;
    set_array1(
        dictionary,
        py,
        "multiplier_exp",
        std::mem::take(&mut packed.multiplier_exp),
    )?;
    set_array1(
        dictionary,
        py,
        "bomb_count",
        std::mem::take(&mut packed.bomb_count),
    )?;
    set_array1(
        dictionary,
        py,
        "terminal",
        std::mem::take(&mut packed.terminal),
    )?;
    Ok(())
}

fn set_observation_history_arrays(
    dictionary: &Bound<'_, PyDict>,
    py: Python<'_>,
    packed: &mut PackedObservation,
) -> PyResult<()> {
    let history_size = packed.history_kind.len();
    set_array1(
        dictionary,
        py,
        "history_offsets",
        std::mem::take(&mut packed.history_offsets),
    )?;
    set_array1(
        dictionary,
        py,
        "history_sequence",
        std::mem::take(&mut packed.history_sequence),
    )?;
    set_array1(
        dictionary,
        py,
        "history_actor",
        std::mem::take(&mut packed.history_actor),
    )?;
    set_array1(
        dictionary,
        py,
        "history_phase",
        std::mem::take(&mut packed.history_phase),
    )?;
    set_array1(
        dictionary,
        py,
        "history_action_code",
        std::mem::take(&mut packed.history_action_code),
    )?;
    set_array2(
        dictionary,
        py,
        "history_rank_counts",
        std::mem::take(&mut packed.history_cards),
        history_size,
        RANK_COUNT,
    )?;
    set_array1(
        dictionary,
        py,
        "history_kind",
        std::mem::take(&mut packed.history_kind),
    )?;
    set_array1(
        dictionary,
        py,
        "history_main_rank",
        std::mem::take(&mut packed.history_main_rank),
    )?;
    set_array1(
        dictionary,
        py,
        "history_chain_len",
        std::mem::take(&mut packed.history_chain_len),
    )?;
    set_array1(
        dictionary,
        py,
        "history_total_cards",
        std::mem::take(&mut packed.history_total_cards),
    )?;
    Ok(())
}

fn packed_actions_to_python(py: Python<'_>, packed: PackedActions) -> PyResult<Bound<'_, PyDict>> {
    let dictionary = PyDict::new(py);
    let action_size = packed.kind.len();
    dictionary.set_item("schema_version", packed.schema_version)?;
    dictionary.set_item("batch_size", packed.batch_size)?;
    set_array1(&dictionary, py, "offsets", packed.offsets)?;
    set_array1(&dictionary, py, "state_index", packed.state_index)?;
    set_array1(&dictionary, py, "phase", packed.phase)?;
    set_array1(&dictionary, py, "action_code", packed.action_code)?;
    set_array2(
        &dictionary,
        py,
        "rank_counts",
        packed.cards,
        action_size,
        RANK_COUNT,
    )?;
    set_array1(&dictionary, py, "kind", packed.kind)?;
    set_array1(&dictionary, py, "main_rank", packed.main_rank)?;
    set_array1(&dictionary, py, "chain_len", packed.chain_len)?;
    set_array1(&dictionary, py, "total_cards", packed.total_cards)?;
    Ok(dictionary)
}

fn packed_step_result_to_python(
    py: Python<'_>,
    packed: PackedStepResult,
) -> PyResult<Bound<'_, PyDict>> {
    let dictionary = PyDict::new(py);
    let batch_size = packed.batch_size;
    dictionary.set_item("schema_version", packed.schema_version)?;
    dictionary.set_item("batch_size", batch_size)?;
    set_array1(&dictionary, py, "acted", packed.acted)?;
    set_array1(&dictionary, py, "event_sequence", packed.event_sequence)?;
    set_array1(&dictionary, py, "event_actor", packed.event_actor)?;
    set_array2(
        &dictionary,
        py,
        "action_rank_counts",
        packed.action_cards,
        batch_size,
        RANK_COUNT,
    )?;
    set_array1(&dictionary, py, "action_phase", packed.action_phase)?;
    set_array1(&dictionary, py, "action_code", packed.action_code)?;
    set_array1(&dictionary, py, "action_kind", packed.action_kind)?;
    set_array1(&dictionary, py, "action_main_rank", packed.action_main_rank)?;
    set_array1(&dictionary, py, "action_chain_len", packed.action_chain_len)?;
    set_array1(
        &dictionary,
        py,
        "action_total_cards",
        packed.action_total_cards,
    )?;
    set_array1(&dictionary, py, "next_player", packed.next_player)?;
    set_array1(&dictionary, py, "terminal", packed.terminal)?;
    set_array2(
        &dictionary,
        py,
        "raw_payoff",
        packed.raw_payoff,
        batch_size,
        3,
    )?;
    set_array2(
        &dictionary,
        py,
        "objective_payoff",
        packed.objective_payoff,
        batch_size,
        3,
    )?;
    dictionary.set_item(
        "observation",
        packed_observation_to_python(py, packed.observation)?,
    )?;
    Ok(dictionary)
}

fn set_array1<T: Element>(
    dictionary: &Bound<'_, PyDict>,
    py: Python<'_>,
    name: &str,
    values: Vec<T>,
) -> PyResult<()> {
    dictionary.set_item(name, PyArray1::from_vec(py, values))
}

fn set_array2<T: Element>(
    dictionary: &Bound<'_, PyDict>,
    py: Python<'_>,
    name: &str,
    values: Vec<T>,
    rows: usize,
    columns: usize,
) -> PyResult<()> {
    dictionary.set_item(
        name,
        PyArray1::from_vec(py, values).reshape([rows, columns])?,
    )
}

fn set_array3<T: Element>(
    dictionary: &Bound<'_, PyDict>,
    py: Python<'_>,
    name: &str,
    values: Vec<T>,
    first: usize,
    second: usize,
    third: usize,
) -> PyResult<()> {
    dictionary.set_item(
        name,
        PyArray1::from_vec(py, values).reshape([first, second, third])?,
    )
}

impl PyDdzEnv {
    fn game(&self) -> PyResult<&PostBidGame> {
        self.game
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("environment must be reset first"))
    }

    fn game_mut(&mut self) -> PyResult<&mut PostBidGame> {
        self.game
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("environment must be reset first"))
    }
}

fn to_python<'py, T: Serialize>(
    py: Python<'py>,
    value: &T,
    label: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let json = serde_json::to_string(value)
        .map_err(|error| PyRuntimeError::new_err(format!("failed to encode {label}: {error}")))?;
    py.import("json")?.call_method1("loads", (json,))
}

fn from_python<T: DeserializeOwned>(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    label: &str,
) -> PyResult<T> {
    let json = py
        .import("json")?
        .call_method1("dumps", (value,))?
        .extract::<String>()?;
    serde_json::from_str(&json)
        .map_err(|error| PyValueError::new_err(format!("invalid {label}: {error}")))
}

fn legacy_rules_from_python(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    label: &str,
) -> PyResult<RuleConfig> {
    let config: VersionedRuleConfig = from_python(py, value, label)?;
    config.validate().map_err(value_error)?;
    config.into_v1().map_err(value_error)
}

fn value_error(error: impl Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn runtime_error(error: impl Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

fn game_error(error: GameError) -> PyErr {
    match error {
        GameError::Terminal | GameError::WrongPhase { .. } | GameError::StateInvariant(_) => {
            runtime_error(error)
        }
        GameError::InvalidSeat { .. }
        | GameError::IllegalAction { .. }
        | GameError::MoveGeneration(_) => value_error(error),
    }
}

fn batch_error(error: BatchError) -> PyErr {
    match error {
        BatchError::RuleConfig(_)
        | BatchError::EmptyBatch
        | BatchError::Deal { .. }
        | BatchError::BatchSizeMismatch { .. }
        | BatchError::MissingActionIndex { .. }
        | BatchError::ActionIndexOutOfRange { .. }
        | BatchError::TerminalActionIndex { .. } => value_error(error),
        BatchError::Uninitialized
        | BatchError::Query { .. }
        | BatchError::Step { .. }
        | BatchError::Rollback { .. }
        | BatchError::UnsupportedAction { .. }
        | BatchError::BufferTooLarge { .. }
        | BatchError::Serialize { .. }
        | BatchError::InternalInvariant(_) => runtime_error(error),
    }
}

#[pyfunction]
fn parse_rule_config<'py>(py: Python<'py>, yaml_text: &str) -> PyResult<Bound<'py, PyAny>> {
    let rules = RuleConfig::from_yaml_str(yaml_text).map_err(value_error)?;
    to_python(py, &rules, "rule configuration")
}

/// Parse either supported rule schema without making it executable in the legacy engine.
#[pyfunction]
fn parse_versioned_rule_config<'py>(
    py: Python<'py>,
    yaml_text: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let rules = VersionedRuleConfig::from_yaml_str(yaml_text).map_err(value_error)?;
    to_python(py, &rules, "versioned rule configuration")
}

/// Return the authoritative stable SHA-256 identity of a complete rule YAML document.
#[pyfunction]
fn rule_config_hash(yaml_text: &str) -> PyResult<String> {
    VersionedRuleConfig::from_yaml_str(yaml_text)
        .and_then(|rules| rules.rules_hash())
        .map_err(value_error)
}

/// Generate every canonical non-Pass lead action for an arbitrary valid hand.
#[pyfunction]
fn generate_lead_actions<'py>(
    py: Python<'py>,
    rank_counts: &Bound<'py, PyAny>,
    rule_config: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let hand: RankCounts = from_python(py, rank_counts, "rank_counts")?;
    let rules = legacy_rules_from_python(py, rule_config.as_any(), "rule_config")?;
    let moves = generate_lead_moves(&hand, &rules).map_err(value_error)?;
    let actions: Vec<GameAction> = moves.into_iter().map(GameAction::Play).collect();
    to_python(py, &actions, "lead actions")
}

/// Compute exact minimum hand groups for several hands using one shared memo.
#[pyfunction]
fn minimum_play_groups<'py>(
    py: Python<'py>,
    rank_count_batch: &Bound<'py, PyAny>,
    rule_config: &Bound<'py, PyDict>,
    count_cap: u32,
) -> PyResult<Bound<'py, PyAny>> {
    let hands: Vec<RankCounts> = from_python(py, rank_count_batch, "rank_count_batch")?;
    let rules = legacy_rules_from_python(py, rule_config.as_any(), "rule_config")?;
    let summaries = minimum_play_groups_many(&hands, &rules, count_cap).map_err(value_error)?;
    to_python(py, &summaries, "minimum play groups")
}

/// Prove a small perfect-information endgame from one serialized native state.
#[pyfunction]
fn solve_endgame<'py>(
    py: Python<'py>,
    serialized_state: &Bound<'py, PyBytes>,
    rule_config: &Bound<'py, PyDict>,
    max_total_cards: u8,
    max_nodes: u64,
) -> PyResult<Bound<'py, PyAny>> {
    let rules = legacy_rules_from_python(py, rule_config.as_any(), "rule_config")?;
    let mut game =
        PostBidGame::deserialize_state(serialized_state.as_bytes(), rules).map_err(value_error)?;
    let result = solve_exact_endgame(
        &mut game,
        ExactSearchConfig {
            max_total_cards,
            max_nodes,
        },
    )
    .map_err(value_error)?;
    to_python(py, &result, "exact endgame result")
}

/// Native extension module installed as `birddou._native`.
#[pymodule]
#[pyo3(name = "_native")]
fn native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDdzEnv>()?;
    module.add_class::<PyBatchDdzEnv>()?;
    module.add_function(wrap_pyfunction!(parse_rule_config, module)?)?;
    module.add_function(wrap_pyfunction!(parse_versioned_rule_config, module)?)?;
    module.add_function(wrap_pyfunction!(rule_config_hash, module)?)?;
    module.add_function(wrap_pyfunction!(generate_lead_actions, module)?)?;
    module.add_function(wrap_pyfunction!(minimum_play_groups, module)?)?;
    module.add_function(wrap_pyfunction!(solve_endgame, module)?)?;
    module.add("API_SCHEMA_VERSION", API_SCHEMA_VERSION)?;
    module.add("BATCH_SCHEMA_VERSION", BATCH_SCHEMA_VERSION)?;
    module.add("SHUFFLE_ALGORITHM", SHUFFLE_ALGORITHM)?;
    Ok(())
}
