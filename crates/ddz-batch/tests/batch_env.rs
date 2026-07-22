use ddz_batch::{
    BatchDdzEnv, BatchError, BATCH_SCHEMA_VERSION, NO_EVENT, NO_MOVE_KIND, NO_PLAYER, NO_RANK,
};
use ddz_core::{GameAction, RANK_COUNT};
use ddz_rules::{deal_post_bid, RuleConfig};

const DOUZERO_POST_BID_YAML: &str = include_str!("../../../configs/rules/douzero_post_bid.yaml");

fn rules() -> RuleConfig {
    RuleConfig::from_yaml_str(DOUZERO_POST_BID_YAML).expect("checked-in profile must parse")
}

#[test]
fn reset_is_deterministic_packed_and_transactional() {
    let mut batch = BatchDdzEnv::new(rules()).expect("rules must initialize a batch");
    assert!(!batch.is_initialized());
    assert!(!batch.all_terminal());
    assert!(matches!(
        batch.packed_observation(),
        Err(BatchError::Uninitialized)
    ));

    let observation = batch.reset(&[7, 8, 9]).expect("seeds must reset");
    assert_eq!(observation.schema_version, BATCH_SCHEMA_VERSION);
    assert_eq!(observation.batch_size, 3);
    assert_eq!(observation.phase, vec![2, 2, 2]);
    assert_eq!(observation.observer, vec![0, 0, 0]);
    assert_eq!(observation.role, vec![0, 0, 0]);
    assert_eq!(observation.own_hand.len(), 3 * RANK_COUNT);
    assert_eq!(observation.public_played.len(), 3 * 3 * RANK_COUNT);
    assert_eq!(observation.public_bottom_cards.len(), 3 * RANK_COUNT);
    assert_eq!(observation.unknown_pool.len(), 3 * RANK_COUNT);
    assert_eq!(
        observation.cards_left,
        vec![20, 17, 17, 20, 17, 17, 20, 17, 17]
    );
    assert_eq!(observation.landlord, vec![0, 0, 0]);
    assert_eq!(observation.last_non_pass_valid, vec![0, 0, 0]);
    assert_eq!(observation.last_non_pass_kind, vec![NO_MOVE_KIND; 3]);
    assert_eq!(observation.last_non_pass_main_rank, vec![NO_RANK; 3]);
    assert_eq!(observation.terminal, vec![0, 0, 0]);
    assert_eq!(observation.history_offsets, vec![0, 0, 0, 0]);
    assert!(observation.history_kind.is_empty());
    assert_eq!(
        &observation.own_hand[..RANK_COUNT],
        &[1, 0, 2, 1, 1, 2, 2, 2, 0, 4, 2, 3, 0, 0, 0]
    );

    let states = batch.serialize_states().expect("states must serialize");
    let mut same = BatchDdzEnv::new(rules()).expect("rules must initialize a batch");
    same.reset(&[7, 8, 9]).expect("same seeds must reset");
    assert_eq!(
        states,
        same.serialize_states().expect("states must serialize")
    );

    assert!(matches!(batch.reset(&[]), Err(BatchError::EmptyBatch)));
    assert_eq!(
        states,
        batch.serialize_states().expect("failed reset is atomic")
    );
}

#[test]
fn packed_actions_exactly_match_each_independent_environment() {
    let seeds = [7, 8, 9];
    let mut batch = BatchDdzEnv::new(rules()).expect("rules must initialize a batch");
    batch.reset(&seeds).expect("seeds must reset");
    let packed = batch.legal_actions_packed().expect("actions must generate");

    assert_eq!(packed.schema_version, BATCH_SCHEMA_VERSION);
    assert_eq!(packed.batch_size, seeds.len());
    assert_eq!(packed.offsets.len(), seeds.len() + 1);
    assert_eq!(packed.cards.len(), packed.kind.len() * RANK_COUNT);
    assert_eq!(packed.state_index.len(), packed.kind.len());
    assert_eq!(packed.main_rank.len(), packed.kind.len());
    assert_eq!(packed.chain_len.len(), packed.kind.len());
    assert_eq!(packed.total_cards.len(), packed.kind.len());

    for (env_index, seed) in seeds.into_iter().enumerate() {
        let game = deal_post_bid(seed, rules()).expect("single deal must initialize");
        let expected = game.legal_actions().expect("single actions must generate");
        let start = usize::try_from(packed.offsets[env_index]).expect("offset must fit");
        let end = usize::try_from(packed.offsets[env_index + 1]).expect("offset must fit");
        assert_eq!(end - start, expected.len());
        for (local_index, action) in expected.iter().enumerate() {
            let flat_index = start + local_index;
            let GameAction::Play(played_move) = *action else {
                panic!("post-bid action must be card play");
            };
            assert_eq!(
                packed.state_index[flat_index],
                i64::try_from(env_index).expect("environment index must fit")
            );
            assert_eq!(packed.kind[flat_index], u8::from(played_move.kind()));
            assert_eq!(packed.main_rank[flat_index], played_move.main_rank());
            assert_eq!(packed.chain_len[flat_index], played_move.chain_len());
            assert_eq!(packed.total_cards[flat_index], played_move.total_cards());
            let card_start = flat_index * RANK_COUNT;
            assert_eq!(
                &packed.cards[card_start..card_start + RANK_COUNT],
                played_move.cards()
            );
        }
    }
}

#[test]
fn step_validates_the_whole_batch_before_mutation_and_matches_single_games() {
    let seeds = [7, 8, 9];
    let mut batch = BatchDdzEnv::new(rules()).expect("rules must initialize a batch");
    batch.reset(&seeds).expect("seeds must reset");
    let before = batch.serialize_states().expect("states must serialize");

    assert!(matches!(
        batch.step_packed(&[0, 1]),
        Err(BatchError::BatchSizeMismatch { .. })
    ));
    assert!(matches!(
        batch.step_packed(&[0, 1_000_000, 0]),
        Err(BatchError::ActionIndexOutOfRange { env_index: 1, .. })
    ));
    assert_eq!(
        before,
        batch.serialize_states().expect("invalid step is atomic")
    );

    let indices = [0_i64, 1, 2];
    let output = batch
        .step_packed(&indices)
        .expect("valid indices must step");
    assert_eq!(output.acted, vec![1, 1, 1]);
    assert_eq!(output.event_sequence, vec![0, 0, 0]);
    assert_eq!(output.event_actor, vec![0, 0, 0]);
    assert_eq!(output.next_player, vec![1, 1, 1]);
    assert_eq!(output.terminal, vec![0, 0, 0]);
    assert_eq!(output.raw_payoff, vec![0; 9]);
    assert_eq!(output.objective_payoff, vec![0; 9]);
    assert_eq!(output.observation.history_offsets, vec![0, 1, 2, 3]);
    assert_eq!(output.observation.history_sequence, vec![0, 0, 0]);

    let mut expected_states = Vec::new();
    for (seed, index) in seeds.into_iter().zip(indices) {
        let mut game = deal_post_bid(seed, rules()).expect("single deal must initialize");
        let actions = game.legal_actions().expect("single actions must generate");
        game.step(&actions[usize::try_from(index).expect("index must fit")])
            .expect("selected single action must step");
        expected_states.push(game.serialize_state().expect("single state must serialize"));
    }
    assert_eq!(
        batch
            .serialize_states()
            .expect("batch states must serialize"),
        expected_states
    );
}

#[test]
fn asynchronous_terminal_environments_use_negative_one_noops() {
    let seeds = [1, 2, 3, 4, 5];
    let mut batch = BatchDdzEnv::new(rules()).expect("rules must initialize a batch");
    batch.reset(&seeds).expect("seeds must reset");
    let mut saw_partial_terminal = false;
    let mut last_output = None;

    for _ in 0..500 {
        if batch.all_terminal() {
            break;
        }
        let actions = batch.legal_actions_packed().expect("actions must generate");
        let mut indices = vec![-1_i64; seeds.len()];
        for (env_index, index) in indices.iter_mut().enumerate() {
            let start = usize::try_from(actions.offsets[env_index]).expect("offset must fit");
            let end = usize::try_from(actions.offsets[env_index + 1]).expect("offset must fit");
            if start < end {
                let best = (start..end)
                    .max_by_key(|&flat_index| actions.total_cards[flat_index])
                    .expect("non-empty action range has a maximum");
                *index = i64::try_from(best - start).expect("local index must fit");
            }
        }
        let output = batch
            .step_packed(&indices)
            .expect("selected actions must step");
        saw_partial_terminal |= output.acted.contains(&0) && output.acted.contains(&1);
        last_output = Some(output);
    }

    assert!(
        batch.all_terminal(),
        "all games must terminate within 500 ticks"
    );
    assert!(
        saw_partial_terminal,
        "different deals should finish asynchronously"
    );
    let output = last_output.expect("at least one batch step must run");
    for env_index in 0..seeds.len() {
        let payoff = &output.raw_payoff[env_index * 3..env_index * 3 + 3];
        assert_eq!(payoff.iter().sum::<i32>(), 0);
    }

    let no_op = batch
        .step_packed(&vec![-1; seeds.len()])
        .expect("terminal -1 indices are valid no-ops");
    assert_eq!(no_op.acted, vec![0; seeds.len()]);
    assert_eq!(no_op.event_sequence, vec![NO_EVENT; seeds.len()]);
    assert_eq!(no_op.event_actor, vec![NO_PLAYER; seeds.len()]);
    assert_eq!(no_op.action_kind, vec![NO_MOVE_KIND; seeds.len()]);
    assert_eq!(no_op.action_main_rank, vec![NO_RANK; seeds.len()]);
    assert!(no_op.terminal.iter().all(|&terminal| terminal == 1));

    let mut invalid = vec![-1; seeds.len()];
    invalid[0] = 0;
    assert!(matches!(
        batch.step_packed(&invalid),
        Err(BatchError::TerminalActionIndex { env_index: 0, .. })
    ));
}
