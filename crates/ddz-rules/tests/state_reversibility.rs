use ddz_core::{
    cards_to_rank_counts, serialize_game_state, GameAction, Move, RankCounts, EMPTY_RANK_COUNTS,
};
use ddz_rules::{
    detect_move_with_rules, GameDeserializeError, GameError, GameRestoreError, PostBidGame,
    RuleConfig, UndoError, UndoToken,
};
use proptest::prelude::*;

const DOUZERO_POST_BID_YAML: &str = include_str!("../../../configs/rules/douzero_post_bid.yaml");

fn rules() -> RuleConfig {
    RuleConfig::from_yaml_str(DOUZERO_POST_BID_YAML).expect("checked-in profile must be valid")
}

fn deal_from_permutation(cards: &[u8]) -> ([RankCounts; 3], RankCounts) {
    assert_eq!(cards.len(), 54);
    (
        [
            cards_to_rank_counts(&cards[..20]).unwrap(),
            cards_to_rank_counts(&cards[20..37]).unwrap(),
            cards_to_rank_counts(&cards[37..]).unwrap(),
        ],
        cards_to_rank_counts(&cards[17..20]).unwrap(),
    )
}

fn standard_game() -> PostBidGame {
    let (hands, bottom) = deal_from_permutation(&(0_u8..54).collect::<Vec<_>>());
    PostBidGame::new(hands, bottom, 0, rules()).unwrap()
}

fn counts(entries: &[(u8, u8)]) -> RankCounts {
    let mut cards = EMPTY_RANK_COUNTS;
    for &(rank, count) in entries {
        cards[usize::from(rank)] = count;
    }
    cards
}

fn action(cards: RankCounts) -> GameAction {
    GameAction::Play(detect_move_with_rules(cards, &rules()).unwrap())
}

fn assert_round_trip(game: &PostBidGame) {
    let bytes = game.serialize_state().unwrap();
    let restored = PostBidGame::deserialize_state(&bytes, rules()).unwrap();

    assert_eq!(restored, *game);
    assert_eq!(restored.serialize_state().unwrap(), bytes);
    assert_eq!(
        restored.legal_actions().unwrap(),
        game.legal_actions().unwrap()
    );
}

#[test]
fn initial_intermediate_and_terminal_states_round_trip_exactly() {
    let mut game = standard_game();
    assert_round_trip(&game);

    game.step(&action(counts(&[(0, 4)]))).unwrap();
    game.step(&GameAction::Play(Move::pass())).unwrap();
    assert_round_trip(&game);

    game.step(&GameAction::Play(Move::pass())).unwrap();
    for rank in 1_u8..5 {
        game.step(&action(counts(&[(rank, 4)]))).unwrap();
        if rank != 4 {
            game.step(&GameAction::Play(Move::pass())).unwrap();
            game.step(&GameAction::Play(Move::pass())).unwrap();
        }
    }
    assert!(game.is_terminal());
    assert_round_trip(&game);
}

#[test]
fn decoded_states_are_rejected_when_history_or_cached_fields_are_forged() {
    let mut game = standard_game();
    game.step(&action(counts(&[(0, 1)]))).unwrap();

    let mut wrong_count = game.state().clone();
    wrong_count.cards_left[0] -= 1;
    let bytes = serialize_game_state(&wrong_count).unwrap();
    assert!(matches!(
        PostBidGame::deserialize_state(&bytes, rules()),
        Err(GameDeserializeError::Restore(
            GameRestoreError::StateMismatch
        ))
    ));

    let mut wrong_actor = game.state().clone();
    wrong_actor.history[0].actor = 2;
    assert!(matches!(
        PostBidGame::from_state(&wrong_actor, rules()),
        Err(GameRestoreError::HistoryActor {
            sequence: 0,
            expected: 0,
            actual: 2,
        })
    ));

    let mut illegal_history = game.state().clone();
    illegal_history.history[0].action = GameAction::Play(Move::pass());
    assert!(matches!(
        PostBidGame::from_state(&illegal_history, rules()),
        Err(GameRestoreError::Replay {
            sequence: 0,
            source: GameError::IllegalAction { .. },
        })
    ));

    let mut wrong_rule_id = game.state().clone();
    wrong_rule_id.rule_config_id = 99;
    assert!(matches!(
        PostBidGame::from_state(&wrong_rule_id, rules()),
        Err(GameRestoreError::RuleConfigIdMismatch {
            state: 99,
            rules: 1,
        })
    ));
}

#[test]
fn every_initial_branch_restores_state_and_bytes_exactly() {
    let mut game = standard_game();
    let initial = game.clone();
    let initial_bytes = game.serialize_state().unwrap();
    let legal = game.legal_actions().unwrap();

    for candidate in legal {
        let undo = game.apply_in_place(&candidate).unwrap();
        game.undo(&undo).unwrap();
        assert_eq!(game, initial);
        assert_eq!(game.serialize_state().unwrap(), initial_bytes);
    }
}

#[test]
fn pass_reset_and_terminal_transitions_are_reversible_in_lifo_order() {
    let mut game = standard_game();
    game.step(&action(counts(&[(0, 4)]))).unwrap();
    let after_opening = game.clone();

    let first_pass = game
        .apply_in_place(&GameAction::Play(Move::pass()))
        .unwrap();
    let after_first_pass = game.clone();
    let second_pass = game
        .apply_in_place(&GameAction::Play(Move::pass()))
        .unwrap();
    let after_second_pass = game.clone();
    assert_eq!(game.state().last_non_pass, None);

    assert!(matches!(
        game.undo(&first_pass),
        Err(UndoError::RevisionMismatch { .. })
    ));
    assert_eq!(game, after_second_pass);
    game.undo(&second_pass).unwrap();
    assert_eq!(game, after_first_pass);
    game.undo(&first_pass).unwrap();
    assert_eq!(game, after_opening);

    game.step(&GameAction::Play(Move::pass())).unwrap();
    game.step(&GameAction::Play(Move::pass())).unwrap();
    for rank in 1_u8..4 {
        game.step(&action(counts(&[(rank, 4)]))).unwrap();
        game.step(&GameAction::Play(Move::pass())).unwrap();
        game.step(&GameAction::Play(Move::pass())).unwrap();
    }
    let before_terminal = game.clone();
    let terminal_undo = game.apply_in_place(&action(counts(&[(4, 4)]))).unwrap();
    assert!(game.is_terminal());
    game.undo(&terminal_undo).unwrap();
    assert_eq!(game, before_terminal);
    assert!(!game.is_terminal());
}

#[test]
fn invalid_in_place_action_is_transactional_and_tokens_are_compact() {
    let mut game = standard_game();
    let before = game.clone();
    assert!(matches!(
        game.apply_in_place(&GameAction::Play(Move::pass())),
        Err(GameError::IllegalAction { .. })
    ));
    assert_eq!(game, before);
    assert!(std::mem::size_of::<UndoToken>() < 256);
}

#[test]
fn tokens_cannot_cross_engine_instances_or_cloned_search_branches() {
    let mut first = standard_game();
    let candidate = first.legal_actions().unwrap()[0];
    let first_token = first.apply_in_place(&candidate).unwrap();

    let mut cloned_branch = first.clone();
    let before = cloned_branch.clone();
    assert_eq!(
        cloned_branch.undo(&first_token),
        Err(UndoError::InstanceMismatch)
    );
    assert_eq!(cloned_branch, before);
    first.undo(&first_token).unwrap();
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn random_transition_stacks_unwind_to_identical_initial_bytes(
        keys in proptest::collection::vec(any::<u64>(), 54),
        selectors in proptest::collection::vec(any::<usize>(), 60),
    ) {
        let mut permutation = (0_u8..54).collect::<Vec<_>>();
        permutation.sort_unstable_by_key(|&card| (keys[usize::from(card)], card));
        let (hands, bottom) = deal_from_permutation(&permutation);
        let mut game = PostBidGame::new(hands, bottom, 0, rules()).unwrap();
        let initial = game.clone();
        let initial_bytes = game.serialize_state().unwrap();
        let mut undo_stack = Vec::new();

        for selector in selectors {
            if game.is_terminal() {
                break;
            }
            let legal = game.legal_actions().unwrap();
            let chosen = legal[selector % legal.len()];

            let branch_before = game.clone();
            let branch_bytes = game.serialize_state().unwrap();
            let branch_undo = game.apply_in_place(&chosen).unwrap();
            game.undo(&branch_undo).unwrap();
            prop_assert_eq!(&game, &branch_before);
            prop_assert_eq!(game.serialize_state().unwrap(), branch_bytes);

            undo_stack.push(game.apply_in_place(&chosen).unwrap());
        }

        while let Some(undo) = undo_stack.pop() {
            game.undo(&undo).unwrap();
        }
        prop_assert_eq!(&game, &initial);
        prop_assert_eq!(game.serialize_state().unwrap(), initial_bytes);
    }
}
