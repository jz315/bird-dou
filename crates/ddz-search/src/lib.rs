//! Hand decomposition primitives used as model features and later search inputs.

use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

use ddz_core::{GameAction, Move, Phase, RankCounts, Seat, EMPTY_RANK_COUNTS};
use ddz_rules::{
    generate_lead_moves, GameError, GenerateMovesError, PostBidGame, RuleConfig, UndoError,
};
use serde::Serialize;

/// Exact minimum number of legal lead groups and capped optimal play orderings.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct DecompositionSummary {
    /// Fewest legal non-Pass moves needed to empty the hand.
    pub min_groups: u8,
    /// Number of optimal ordered decompositions, saturated at the requested cap.
    pub optimal_orderings_capped: u32,
}

/// Hard limits for perfect-information landlord-versus-farmer endgame solving.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactSearchConfig {
    /// Reject roots with more cards to keep exact search an endgame-only tool.
    pub max_total_cards: u8,
    /// Abort instead of returning an unproven result after this many unique nodes.
    pub max_nodes: u64,
}

impl Default for ExactSearchConfig {
    fn default() -> Self {
        Self {
            max_total_cards: 12,
            max_nodes: 1_000_000,
        }
    }
}

/// Proven perfect-information result under optimal landlord/team play.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ExactSolveResult {
    /// Whether the landlord can force a win.
    pub landlord_forced_win: bool,
    /// Terminal distance under optimal play (winner shortens, loser delays).
    pub plies_to_terminal: u16,
    /// Optimal root action, absent only for an already-terminal root.
    pub best_action: Option<GameAction>,
    /// Number of unique searched states.
    pub nodes: u64,
    /// Transposition table hits.
    pub cache_hits: u64,
}

/// Failures that prevent an exact, proven endgame result.
#[derive(Debug)]
pub enum ExactSearchError {
    /// Configuration limits must both be positive.
    InvalidConfig,
    /// Exact solver accepts only resolved card-play/terminal states.
    WrongPhase(Phase),
    /// The root exceeds the configured endgame size.
    TooManyCards { actual: u8, maximum: u8 },
    /// Search exhausted its hard node budget and returned no approximation.
    NodeBudgetExceeded { maximum: u64 },
    /// Authoritative transition failed during search.
    Transition(GameError),
    /// A LIFO undo failed during search.
    Undo(UndoError),
    /// A non-terminal state unexpectedly had no legal actions.
    NoLegalActions,
    /// Terminal payoff did not identify a resolved landlord.
    MissingLandlord,
    /// Terminal distance overflowed its stable result field.
    DistanceOverflow,
}

impl Display for ExactSearchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig => {
                formatter.write_str("exact search max_total_cards and max_nodes must be positive")
            }
            Self::WrongPhase(phase) => {
                write!(formatter, "exact search requires card play, got {phase:?}")
            }
            Self::TooManyCards { actual, maximum } => write!(
                formatter,
                "exact-search root has {actual} cards, above configured maximum {maximum}"
            ),
            Self::NodeBudgetExceeded { maximum } => {
                write!(formatter, "exact search exceeded {maximum} unique nodes")
            }
            Self::Transition(error) => write!(formatter, "exact-search transition failed: {error}"),
            Self::Undo(error) => write!(formatter, "exact-search undo failed: {error}"),
            Self::NoLegalActions => {
                formatter.write_str("non-terminal exact-search node has no action")
            }
            Self::MissingLandlord => formatter.write_str("exact-search state has no landlord"),
            Self::DistanceOverflow => {
                formatter.write_str("exact-search terminal distance overflowed")
            }
        }
    }
}

impl Error for ExactSearchError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Transition(error) => Some(error),
            Self::Undo(error) => Some(error),
            Self::InvalidConfig
            | Self::WrongPhase(_)
            | Self::TooManyCards { .. }
            | Self::NodeBudgetExceeded { .. }
            | Self::NoLegalActions
            | Self::MissingLandlord
            | Self::DistanceOverflow => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ExactSearchKey {
    current_player: Seat,
    landlord: Seat,
    hands: [RankCounts; 3],
    last_non_pass: Option<Move>,
    last_non_pass_player: Option<Seat>,
    consecutive_passes: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExactSearchValue {
    landlord_forced_win: bool,
    plies_to_terminal: u16,
}

struct ExactSearchContext {
    config: ExactSearchConfig,
    nodes: u64,
    cache_hits: u64,
    memo: HashMap<ExactSearchKey, ExactSearchValue>,
}

/// Solve a small perfect-information state with authoritative apply/undo transitions.
///
/// Farmers are treated as one team. The side that can force a win minimizes its
/// distance; a side that is losing under optimal play maximizes the remaining distance.
/// The function either proves the result or returns a hard-budget error.
///
/// # Errors
///
/// Rejects invalid limits, unresolved/non-card-play roots, roots above the card
/// threshold, exhausted node budgets, and any authoritative transition/undo failure.
pub fn solve_exact_endgame(
    game: &mut PostBidGame,
    config: ExactSearchConfig,
) -> Result<ExactSolveResult, ExactSearchError> {
    if config.max_total_cards == 0 || config.max_nodes == 0 {
        return Err(ExactSearchError::InvalidConfig);
    }
    if !matches!(game.state().phase, Phase::CardPlay | Phase::Terminal) {
        return Err(ExactSearchError::WrongPhase(game.state().phase));
    }
    let total_cards = game.state().cards_left.iter().copied().sum::<u8>();
    if total_cards > config.max_total_cards {
        return Err(ExactSearchError::TooManyCards {
            actual: total_cards,
            maximum: config.max_total_cards,
        });
    }
    let mut context = ExactSearchContext {
        config,
        nodes: 0,
        cache_hits: 0,
        memo: HashMap::new(),
    };
    if game.is_terminal() {
        let value = terminal_value(game)?;
        return Ok(ExactSolveResult {
            landlord_forced_win: value.landlord_forced_win,
            plies_to_terminal: 0,
            best_action: None,
            nodes: 1,
            cache_hits: 0,
        });
    }
    let actions = game.legal_actions().map_err(ExactSearchError::Transition)?;
    if actions.is_empty() {
        return Err(ExactSearchError::NoLegalActions);
    }
    let actor_is_landlord = Some(game.state().current_player) == game.state().landlord;
    let mut best: Option<(ExactSearchValue, GameAction)> = None;
    for action in actions {
        let token = game
            .apply_in_place(&action)
            .map_err(ExactSearchError::Transition)?;
        let child = solve_exact_node(game, &mut context);
        game.undo(&token).map_err(ExactSearchError::Undo)?;
        let child = child?;
        let candidate = ExactSearchValue {
            landlord_forced_win: child.landlord_forced_win,
            plies_to_terminal: child
                .plies_to_terminal
                .checked_add(1)
                .ok_or(ExactSearchError::DistanceOverflow)?,
        };
        if best
            .as_ref()
            .is_none_or(|(current, _)| prefer_candidate(*current, candidate, actor_is_landlord))
        {
            best = Some((candidate, action));
        }
    }
    let (value, best_action) = best.ok_or(ExactSearchError::NoLegalActions)?;
    Ok(ExactSolveResult {
        landlord_forced_win: value.landlord_forced_win,
        plies_to_terminal: value.plies_to_terminal,
        best_action: Some(best_action),
        nodes: context.nodes,
        cache_hits: context.cache_hits,
    })
}

fn solve_exact_node(
    game: &mut PostBidGame,
    context: &mut ExactSearchContext,
) -> Result<ExactSearchValue, ExactSearchError> {
    if game.is_terminal() {
        return terminal_value(game);
    }
    let state = game.state();
    let landlord = state.landlord.ok_or(ExactSearchError::MissingLandlord)?;
    let key = ExactSearchKey {
        current_player: state.current_player,
        landlord,
        hands: state.hands,
        last_non_pass: state.last_non_pass,
        last_non_pass_player: state.last_non_pass_player,
        consecutive_passes: state.consecutive_passes,
    };
    if let Some(value) = context.memo.get(&key) {
        context.cache_hits = context.cache_hits.saturating_add(1);
        return Ok(*value);
    }
    if context.nodes >= context.config.max_nodes {
        return Err(ExactSearchError::NodeBudgetExceeded {
            maximum: context.config.max_nodes,
        });
    }
    context.nodes += 1;
    let actor_is_landlord = state.current_player == landlord;
    let actions = game.legal_actions().map_err(ExactSearchError::Transition)?;
    if actions.is_empty() {
        return Err(ExactSearchError::NoLegalActions);
    }
    let mut best: Option<ExactSearchValue> = None;
    for action in actions {
        let token = game
            .apply_in_place(&action)
            .map_err(ExactSearchError::Transition)?;
        let child = solve_exact_node(game, context);
        game.undo(&token).map_err(ExactSearchError::Undo)?;
        let child = child?;
        let candidate = ExactSearchValue {
            landlord_forced_win: child.landlord_forced_win,
            plies_to_terminal: child
                .plies_to_terminal
                .checked_add(1)
                .ok_or(ExactSearchError::DistanceOverflow)?,
        };
        if best.is_none_or(|current| prefer_candidate(current, candidate, actor_is_landlord)) {
            best = Some(candidate);
        }
    }
    let value = best.ok_or(ExactSearchError::NoLegalActions)?;
    context.memo.insert(key, value);
    Ok(value)
}

fn terminal_value(game: &PostBidGame) -> Result<ExactSearchValue, ExactSearchError> {
    let landlord = game
        .state()
        .landlord
        .ok_or(ExactSearchError::MissingLandlord)?;
    Ok(ExactSearchValue {
        landlord_forced_win: game.state().raw_payoff[usize::from(landlord)] > 0,
        plies_to_terminal: 0,
    })
}

fn prefer_candidate(
    current: ExactSearchValue,
    candidate: ExactSearchValue,
    actor_is_landlord: bool,
) -> bool {
    let candidate_desired = candidate.landlord_forced_win == actor_is_landlord;
    let current_desired = current.landlord_forced_win == actor_is_landlord;
    if candidate_desired != current_desired {
        return candidate_desired;
    }
    if candidate_desired {
        candidate.plies_to_terminal < current.plies_to_terminal
    } else {
        candidate.plies_to_terminal > current.plies_to_terminal
    }
}

/// Error returned when a hand cannot be analyzed.
#[derive(Debug)]
pub enum DecompositionError {
    /// A zero saturation cap cannot represent any decomposition.
    ZeroCountCap,
    /// Authoritative move generation rejected a hand or rule profile.
    MoveGeneration(GenerateMovesError),
    /// A non-empty valid hand unexpectedly generated no lead move.
    NoLeadMove(RankCounts),
}

impl Display for DecompositionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroCountCap => formatter.write_str("decomposition count cap must be positive"),
            Self::MoveGeneration(error) => write!(formatter, "move generation failed: {error}"),
            Self::NoLeadMove(hand) => {
                write!(formatter, "non-empty hand has no lead move: {hand:?}")
            }
        }
    }
}

impl Error for DecompositionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::MoveGeneration(error) => Some(error),
            Self::ZeroCountCap | Self::NoLeadMove(_) => None,
        }
    }
}

/// Analyze several related hands with one shared transposition table.
///
/// Counting treats play order as distinct. The count is a bounded feature only;
/// minimum group count is exact and neither value may prune authoritative actions.
///
/// # Errors
///
/// Returns [`DecompositionError::ZeroCountCap`] for a zero cap, propagates
/// authoritative move-generation failures, and reports an internal invariant if
/// a valid non-empty hand has no lead action.
pub fn minimum_play_groups_many(
    hands: &[RankCounts],
    rules: &RuleConfig,
    count_cap: u32,
) -> Result<Vec<DecompositionSummary>, DecompositionError> {
    if count_cap == 0 {
        return Err(DecompositionError::ZeroCountCap);
    }
    let mut memo = HashMap::new();
    hands
        .iter()
        .map(|hand| solve(*hand, rules, count_cap, &mut memo))
        .collect()
}

fn solve(
    hand: RankCounts,
    rules: &RuleConfig,
    count_cap: u32,
    memo: &mut HashMap<RankCounts, DecompositionSummary>,
) -> Result<DecompositionSummary, DecompositionError> {
    if hand == EMPTY_RANK_COUNTS {
        return Ok(DecompositionSummary {
            min_groups: 0,
            optimal_orderings_capped: 1,
        });
    }
    if let Some(summary) = memo.get(&hand) {
        return Ok(*summary);
    }
    let moves = generate_lead_moves(&hand, rules).map_err(DecompositionError::MoveGeneration)?;
    if moves.is_empty() {
        return Err(DecompositionError::NoLeadMove(hand));
    }
    let mut best = u8::MAX;
    let mut orderings = 0_u32;
    for played in moves {
        let mut post = hand;
        for (remaining, used) in post.iter_mut().zip(played.cards()) {
            *remaining -= *used;
        }
        let child = solve(post, rules, count_cap, memo)?;
        let candidate = child.min_groups.saturating_add(1);
        if candidate < best {
            best = candidate;
            orderings = child.optimal_orderings_capped;
        } else if candidate == best {
            orderings = orderings
                .saturating_add(child.optimal_orderings_capped)
                .min(count_cap);
        }
    }
    let summary = DecompositionSummary {
        min_groups: best,
        optimal_orderings_capped: orderings.min(count_cap),
    };
    memo.insert(hand, summary);
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddz_rules::deal_post_bid;

    fn rules() -> RuleConfig {
        RuleConfig::from_yaml_str(include_str!("../../../configs/rules/douzero_post_bid.yaml"))
            .unwrap()
    }

    #[test]
    fn exact_groups_and_orderings_cover_empty_pair_and_unrelated_singles() {
        let mut pair = EMPTY_RANK_COUNTS;
        pair[0] = 2;
        let mut singles = EMPTY_RANK_COUNTS;
        singles[0] = 1;
        singles[2] = 1;
        let summaries =
            minimum_play_groups_many(&[EMPTY_RANK_COUNTS, pair, singles], &rules(), 255).unwrap();

        assert_eq!(summaries[0].min_groups, 0);
        assert_eq!(summaries[0].optimal_orderings_capped, 1);
        assert_eq!(summaries[1].min_groups, 1);
        assert_eq!(summaries[1].optimal_orderings_capped, 1);
        assert_eq!(summaries[2].min_groups, 2);
        assert_eq!(summaries[2].optimal_orderings_capped, 2);
    }

    #[test]
    fn duplicate_queries_share_results_and_counts_saturate() {
        let mut hand = EMPTY_RANK_COUNTS;
        hand[..5].fill(1);
        let summaries = minimum_play_groups_many(&[hand, hand], &rules(), 1).unwrap();

        assert_eq!(summaries[0], summaries[1]);
        assert_eq!(summaries[0].min_groups, 1);
        assert_eq!(summaries[0].optimal_orderings_capped, 1);
        assert!(matches!(
            minimum_play_groups_many(&[hand], &rules(), 0),
            Err(DecompositionError::ZeroCountCap)
        ));
    }

    #[test]
    fn exact_endgame_is_proven_branchable_and_leaves_root_unchanged() {
        let rules = rules();
        let mut selected = None;
        for seed in 1..100 {
            let mut game = deal_post_bid(seed, rules).unwrap();
            while !game.is_terminal() && game.state().cards_left.iter().sum::<u8>() > 7 {
                let action = game
                    .legal_actions()
                    .unwrap()
                    .into_iter()
                    .max_by_key(|action| match action {
                        GameAction::Play(played) => played.total_cards(),
                        GameAction::Bid(_) | GameAction::Double(_) => 0,
                    })
                    .unwrap();
                game.step(&action).unwrap();
            }
            if !game.is_terminal() {
                selected = Some(game);
                break;
            }
        }
        let mut game = selected.expect("one deterministic game must reach a live tiny root");
        let before = game.serialize_state().unwrap();
        let config = ExactSearchConfig {
            max_total_cards: 7,
            max_nodes: 1_000_000,
        };
        let result = solve_exact_endgame(&mut game, config).unwrap();
        assert_eq!(game.serialize_state().unwrap(), before);
        assert!(result.best_action.is_some());
        assert!(result.plies_to_terminal > 0);
        assert!(result.nodes > 0);

        game.step(&result.best_action.unwrap()).unwrap();
        let child = solve_exact_endgame(&mut game, config).unwrap();
        assert_eq!(child.landlord_forced_win, result.landlord_forced_win);
        assert_eq!(child.plies_to_terminal + 1, result.plies_to_terminal);
    }

    #[test]
    fn exact_endgame_rejects_non_endgame_roots() {
        let mut game = deal_post_bid(7, rules()).unwrap();
        assert!(matches!(
            solve_exact_endgame(
                &mut game,
                ExactSearchConfig {
                    max_total_cards: 12,
                    max_nodes: 10,
                }
            ),
            Err(ExactSearchError::TooManyCards {
                actual: 54,
                maximum: 12
            })
        ));
    }
}
