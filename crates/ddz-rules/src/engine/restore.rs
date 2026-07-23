use ddz_core::{GameState, LandlordSelectionState};

use super::{invariants, Game, GameRestoreError};
use crate::{deal_plan_for_attempt, EconomyContext, RuleConfig, RuleProfile};

impl Game {
    pub fn restore(
        rules: RuleConfig,
        match_seed: u64,
        economy: EconomyContext,
        state: GameState,
    ) -> Result<Self, GameRestoreError> {
        rules.validate().map_err(GameRestoreError::RuleConfig)?;
        if state.rule_config_id != rules.rule_config_id {
            return Err(GameRestoreError::RuleConfigIdMismatch {
                expected: rules.rule_config_id,
                actual: state.rule_config_id,
            });
        }
        let expected_plan = deal_plan_for_attempt(match_seed, state.deal.attempt)
            .map_err(GameRestoreError::Deal)?;
        if state.deal.plan != expected_plan {
            return Err(GameRestoreError::DealPlanMismatch {
                attempt: state.deal.attempt,
            });
        }
        state.validate().map_err(GameRestoreError::State)?;
        validate_profile_state(&rules, &state)?;
        invariants::validate(&state, &rules).map_err(GameRestoreError::RuleState)?;
        replay_and_compare(rules, match_seed, economy, state)
    }
}

fn replay_and_compare(
    rules: RuleConfig,
    match_seed: u64,
    economy: EconomyContext,
    expected: GameState,
) -> Result<Game, GameRestoreError> {
    let mut replay = match rules.profile {
        RuleProfile::DouzeroPostBid => {
            let landlord = expected
                .landlord()
                .ok_or(GameRestoreError::ReplayMissingLandlord)?;
            Game::new_post_bid(rules, match_seed, landlord).map_err(GameRestoreError::Replay)?
        }
        RuleProfile::HuanleClassic => {
            Game::new_huanle(rules, match_seed, economy).map_err(GameRestoreError::Replay)?
        }
    };
    let player_events = expected.history.iter().filter_map(|event| match event.kind {
        ddz_core::GameEventKind::Player(player) => Some(player),
        ddz_core::GameEventKind::System(_) => None,
    });
    for player in player_events {
        replay
            .step(player.actor, player.action)
            .map_err(GameRestoreError::Replay)?;
    }
    if replay.state != expected {
        return Err(GameRestoreError::ReplayMismatch);
    }
    Ok(replay)
}

fn validate_profile_state(
    rules: &RuleConfig,
    state: &GameState,
) -> Result<(), GameRestoreError> {
    if state.stake.base_unit != rules.settlement.base_unit {
        return Err(GameRestoreError::BaseUnitMismatch {
            expected: rules.settlement.base_unit,
            actual: state.stake.base_unit,
        });
    }
    let profile_matches = match rules.profile {
        RuleProfile::DouzeroPostBid => matches!(
            &state.landlord_selection,
            LandlordSelectionState::PostBid { .. }
        ),
        RuleProfile::HuanleClassic => !matches!(
            &state.landlord_selection,
            LandlordSelectionState::PostBid { .. }
        ),
    };
    if !profile_matches {
        return Err(GameRestoreError::ProfileStateMismatch {
            profile: rules.profile,
        });
    }
    Ok(())
}
