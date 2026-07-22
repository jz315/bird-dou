use ddz_core::{
    cards_to_rank_counts, max_count_for_rank, CardError, DoubleAction, GameAction, Move, MoveKind,
    Phase, RankCounts, Role, EMPTY_RANK_COUNTS, RANK_COUNT,
};
use ddz_rules::{
    detect_move_with_rules, GameError, GameInitError, PostBidGame, RewardMode, RuleConfig,
};
use proptest::prelude::*;

const DOUZERO_POST_BID_YAML: &str = include_str!("../../../configs/rules/douzero_post_bid.yaml");
const CANONICAL_FULL_YAML: &str = include_str!("../../../configs/rules/canonical_full.yaml");

fn douzero_rules() -> RuleConfig {
    RuleConfig::from_yaml_str(DOUZERO_POST_BID_YAML).expect("checked-in profile must be valid")
}

fn standard_deal() -> ([RankCounts; 3], RankCounts) {
    deal_from_permutation(&(0_u8..54).collect::<Vec<_>>())
}

fn deal_from_permutation(cards: &[u8]) -> ([RankCounts; 3], RankCounts) {
    assert_eq!(cards.len(), 54);
    let landlord = cards_to_rank_counts(&cards[..20]).unwrap();
    let farmer_down = cards_to_rank_counts(&cards[20..37]).unwrap();
    let farmer_up = cards_to_rank_counts(&cards[37..]).unwrap();
    let bottom = cards_to_rank_counts(&cards[17..20]).unwrap();
    ([landlord, farmer_down, farmer_up], bottom)
}

fn play(cards: RankCounts, rules: &RuleConfig) -> GameAction {
    GameAction::Play(detect_move_with_rules(cards, rules).expect("scripted move must be legal"))
}

fn counts(entries: &[(u8, u8)]) -> RankCounts {
    let mut result = EMPTY_RANK_COUNTS;
    for &(rank, count) in entries {
        result[usize::from(rank)] = count;
    }
    result
}

#[test]
fn initializes_a_complete_post_bid_state_and_landlord_leads() {
    let rules = douzero_rules();
    let (hands, bottom) = standard_deal();
    let game = PostBidGame::new(hands, bottom, 0, rules).unwrap();
    let state = game.state();

    assert_eq!(state.rule_config_id, rules.rule_config_id);
    assert_eq!(state.phase, Phase::CardPlay);
    assert_eq!(state.current_player, 0);
    assert_eq!(state.landlord, Some(0));
    assert_eq!(state.cards_left, [20, 17, 17]);
    assert_eq!(state.last_non_pass, None);
    assert_eq!(state.last_non_pass_player, None);
    assert_eq!(state.consecutive_passes, 0);
    assert_eq!(state.bomb_count, 0);
    assert_eq!(state.multiplier_exp, 0);
    assert!(state.history.is_empty());
    assert!(!state.terminal);
    assert_eq!(state.raw_payoff, [0; 3]);

    let legal = game.legal_actions().unwrap();
    assert!(!legal.is_empty());
    assert!(!legal.contains(&GameAction::Play(Move::pass())));
}

#[test]
fn two_passes_reset_the_trick_to_the_last_player() {
    let rules = douzero_rules();
    let (hands, bottom) = standard_deal();
    let mut game = PostBidGame::new(hands, bottom, 0, rules).unwrap();
    let opening = play(counts(&[(0, 1)]), &rules);

    game.step(&opening).unwrap();
    assert_eq!(game.state().current_player, 1);
    assert_eq!(
        game.state().last_non_pass,
        Some(match opening {
            GameAction::Play(played) => played,
            GameAction::Bid(_) | GameAction::Double(_) => unreachable!(),
        })
    );

    game.step(&GameAction::Play(Move::pass())).unwrap();
    assert_eq!(game.state().current_player, 2);
    assert_eq!(game.state().consecutive_passes, 1);
    assert!(game.state().last_non_pass.is_some());

    game.step(&GameAction::Play(Move::pass())).unwrap();
    assert_eq!(game.state().current_player, 0);
    assert_eq!(game.state().consecutive_passes, 0);
    assert_eq!(game.state().last_non_pass, None);
    assert_eq!(game.state().last_non_pass_player, None);
    assert!(!game
        .legal_actions()
        .unwrap()
        .contains(&GameAction::Play(Move::pass())));
}

fn scripted_landlord_bomb_win(rules: RuleConfig) -> (PostBidGame, ddz_core::StepResult) {
    rules.validate().unwrap();
    let (hands, bottom) = standard_deal();
    let mut game = PostBidGame::new(hands, bottom, 0, rules).unwrap();
    let mut terminal_result = None;

    for rank in 0_u8..5 {
        let result = game.step(&play(counts(&[(rank, 4)]), &rules)).unwrap();
        if rank == 4 {
            terminal_result = Some(result);
        } else {
            assert!(!result.terminal);
            assert_eq!(result.objective_payoff, [0; 3]);
            game.step(&GameAction::Play(Move::pass())).unwrap();
            game.step(&GameAction::Play(Move::pass())).unwrap();
        }
    }

    (game, terminal_result.expect("fifth bomb ends the game"))
}

#[test]
fn last_play_ends_immediately_and_all_reward_modes_match_douzero() {
    let mut rules = douzero_rules();
    for (mode, expected_objective) in [
        (RewardMode::WinPercentage, [1, -1, -1]),
        (RewardMode::AverageDifferencePoints, [32, -32, -32]),
        (RewardMode::LogAverageDifferencePoints, [6, -6, -6]),
    ] {
        rules.reward_mode = mode;
        let (mut game, result) = scripted_landlord_bomb_win(rules);

        assert!(result.terminal);
        assert_eq!(result.next_player, None);
        assert_eq!(result.raw_payoff, [64, -32, -32]);
        assert_eq!(result.objective_payoff, expected_objective);
        assert!(game.is_terminal());
        assert_eq!(game.state().phase, Phase::Terminal);
        assert_eq!(game.state().current_player, 0);
        assert_eq!(game.state().cards_left, [0, 17, 17]);
        assert_eq!(game.state().bomb_count, 5);
        assert_eq!(game.state().multiplier_exp, 5);
        assert_eq!(game.state().spring_state.landlord_non_pass_plays, 5);
        assert_eq!(game.state().spring_state.farmer_non_pass_plays, 0);
        assert_eq!(game.state().history.len(), 13);
        assert_eq!(game.state().history[12].sequence, 12);
        assert!(game.legal_actions().unwrap().is_empty());
        assert!(matches!(
            game.step(&GameAction::Play(Move::pass())),
            Err(GameError::Terminal)
        ));
    }
}

#[test]
fn farmer_team_win_has_the_correct_payoff_signs() {
    let mut rules = douzero_rules();
    rules.reward_mode = RewardMode::WinPercentage;
    let (hands, bottom) = standard_deal();
    let mut game = PostBidGame::new(hands, bottom, 0, rules).unwrap();

    let mut terminal_result = None;
    while !game.is_terminal() {
        let legal = game.legal_actions().unwrap();
        let action = if game.state().current_player == 1 {
            legal
                .iter()
                .filter(|action| !matches!(action, GameAction::Play(played) if played.kind() == MoveKind::Pass))
                .max_by_key(|action| match action {
                    GameAction::Play(played) => played.total_cards(),
                    GameAction::Bid(_) | GameAction::Double(_) => 0,
                })
                .copied()
                .expect("favored farmer always has a non-pass action")
        } else if legal.contains(&GameAction::Play(Move::pass())) {
            GameAction::Play(Move::pass())
        } else {
            legal[0]
        };
        let result = game.step(&action).unwrap();
        if result.terminal {
            terminal_result = Some(result);
        }
    }

    assert_eq!(game.state().current_player, 1);
    assert!(game.state().raw_payoff[0] < 0);
    assert!(game.state().raw_payoff[1] > 0);
    assert_eq!(game.state().raw_payoff[1], game.state().raw_payoff[2]);
    assert_eq!(game.state().raw_payoff.iter().sum::<i32>(), 0);
    assert_eq!(terminal_result.unwrap().objective_payoff, [-1, 1, 1]);
}

#[test]
fn nonzero_landlord_seat_controls_turn_order_and_payoff_rotation() {
    let rules = douzero_rules();
    let (hands, bottom) = standard_deal();
    let rotated_hands = [hands[1], hands[2], hands[0]];
    let mut game = PostBidGame::new(rotated_hands, bottom, 2, rules).unwrap();
    assert_eq!(game.state().current_player, 2);

    let mut terminal_result = None;
    for rank in 0_u8..5 {
        let result = game.step(&play(counts(&[(rank, 4)]), &rules)).unwrap();
        if result.terminal {
            terminal_result = Some(result);
        } else {
            assert_eq!(game.state().current_player, 0);
            game.step(&GameAction::Play(Move::pass())).unwrap();
            game.step(&GameAction::Play(Move::pass())).unwrap();
            assert_eq!(game.state().current_player, 2);
        }
    }

    let result = terminal_result.unwrap();
    assert_eq!(game.state().current_player, 2);
    assert_eq!(result.raw_payoff, [-32, -32, 64]);
    assert_eq!(result.objective_payoff, [-32, -32, 32]);
}

#[test]
fn illegal_actions_are_rejected_without_mutating_state() {
    let rules = douzero_rules();
    let (hands, bottom) = standard_deal();
    let mut game = PostBidGame::new(hands, bottom, 0, rules).unwrap();

    for illegal in [
        GameAction::Play(Move::pass()),
        GameAction::Double(DoubleAction::Double),
        play(counts(&[(5, 1)]), &rules),
    ] {
        let before = game.clone();
        assert!(matches!(
            game.step(&illegal),
            Err(GameError::IllegalAction { actor: 0, .. })
        ));
        assert_eq!(game, before);
    }
    assert!(matches!(
        game.observe(3),
        Err(GameError::InvalidSeat { seat: 3 })
    ));
}

#[test]
fn observations_hide_the_two_opponent_hand_allocations() {
    let rules = douzero_rules();
    let (hands, bottom) = standard_deal();
    let mut swapped = hands;
    swapped.swap(1, 2);
    let first = PostBidGame::new(hands, bottom, 0, rules).unwrap();
    let second = PostBidGame::new(swapped, bottom, 0, rules).unwrap();

    let first_observation = first.observe(0).unwrap();
    let second_observation = second.observe(0).unwrap();
    assert_eq!(first_observation, second_observation);
    assert_eq!(first_observation.role, Role::Landlord);
    assert_eq!(first_observation.own_hand, first.state().hands[0]);
    assert_eq!(
        serde_yaml_ng::to_string(&first_observation).unwrap(),
        serde_yaml_ng::to_string(&second_observation).unwrap()
    );

    let farmer_observation = first.observe(1).unwrap();
    assert_eq!(farmer_observation.role, Role::Farmer);
    assert_eq!(farmer_observation.own_hand, first.state().hands[1]);
    assert_ne!(farmer_observation.own_hand, first_observation.own_hand);

    let mut alternative = hands;
    alternative[0][0] -= 1;
    alternative[2][0] += 1;
    alternative[2][9] -= 1;
    alternative[0][9] += 1;
    let alternative_game = PostBidGame::new(alternative, bottom, 0, rules).unwrap();
    assert_eq!(
        first.observe(1).unwrap(),
        alternative_game.observe(1).unwrap()
    );
}

#[test]
fn malformed_deals_and_unsupported_profiles_are_rejected() {
    let rules = douzero_rules();
    let (hands, bottom) = standard_deal();
    assert!(matches!(
        PostBidGame::new(hands, bottom, 3, rules),
        Err(GameInitError::InvalidLandlord { landlord: 3 })
    ));

    let mut invalid_hand = hands;
    invalid_hand[0][0] = 5;
    assert!(matches!(
        PostBidGame::new(invalid_hand, bottom, 0, rules),
        Err(GameInitError::InvalidHand {
            source: CardError::TooManyCardsForRank { rank_id: 0, .. },
            ..
        })
    ));

    let mut wrong_size = hands;
    wrong_size[0][0] -= 1;
    wrong_size[1][0] += 1;
    assert!(matches!(
        PostBidGame::new(wrong_size, bottom, 0, rules),
        Err(GameInitError::WrongHandSize { seat: 0, .. })
    ));

    let mut deck_mismatch = hands;
    deck_mismatch[0][0] -= 1;
    deck_mismatch[0][5] += 1;
    assert!(matches!(
        PostBidGame::new(deck_mismatch, bottom, 0, rules),
        Err(GameInitError::DeckCountMismatch { rank_id: 0, .. })
    ));

    assert!(matches!(
        PostBidGame::new(hands, counts(&[(5, 3)]), 0, rules),
        Err(GameInitError::BottomCardNotHeldByLandlord { rank_id: 5, .. })
    ));
    assert!(matches!(
        PostBidGame::new(hands, counts(&[(4, 2)]), 0, rules),
        Err(GameInitError::WrongBottomCardCount { actual: 2 })
    ));

    let canonical = RuleConfig::from_yaml_str(CANONICAL_FULL_YAML).unwrap();
    assert!(matches!(
        PostBidGame::new(hands, bottom, 0, canonical),
        Err(GameInitError::UnsupportedProfile { .. })
    ));
}

fn assert_state_invariants(game: &PostBidGame, initial_hands: &[RankCounts; 3]) {
    let state = game.state();
    for (seat, initial_hand) in initial_hands.iter().enumerate() {
        assert_eq!(state.cards_left[seat], state.hands[seat].iter().sum::<u8>());
        for (rank, &initial_count) in initial_hand.iter().enumerate() {
            assert_eq!(
                state.hands[seat][rank] + state.played_cards[seat][rank],
                initial_count
            );
        }
    }
    for (rank_id, rank) in (0_u8..).zip(0..RANK_COUNT) {
        let conserved = (0..3)
            .map(|seat| state.hands[seat][rank] + state.played_cards[seat][rank])
            .sum::<u8>();
        assert_eq!(conserved, max_count_for_rank(rank_id).unwrap());
    }
    assert_eq!(state.terminal, state.phase == Phase::Terminal);
    assert_eq!(
        state.last_non_pass.is_some(),
        state.last_non_pass_player.is_some()
    );
    if state.last_non_pass.is_none() {
        assert_eq!(state.consecutive_passes, 0);
    } else {
        assert!(state.consecutive_passes <= 1);
    }
    for (sequence, event) in state.history.iter().enumerate() {
        assert_eq!(event.sequence, u32::try_from(sequence).unwrap());
    }
}

#[test]
fn hidden_assignment_sampling_is_replay_validated_and_information_set_consistent() {
    let rules = douzero_rules();
    let (hands, bottom) = standard_deal();
    let game = PostBidGame::new(hands, bottom, 0, rules).unwrap();
    let exact = game.with_hidden_assignment(0, hands[1]).unwrap();
    assert_eq!(
        exact.serialize_state().unwrap(),
        game.serialize_state().unwrap()
    );

    let mut alternative = hands[1];
    let rank_from_a = (0..RANK_COUNT)
        .find(|&rank| {
            hands[1][rank] > 0
                && hands[2][rank] < max_count_for_rank(u8::try_from(rank).unwrap()).unwrap()
        })
        .unwrap();
    let rank_from_b = (0..RANK_COUNT)
        .find(|&rank| {
            rank != rank_from_a
                && hands[2][rank] > 0
                && hands[1][rank] < max_count_for_rank(u8::try_from(rank).unwrap()).unwrap()
        })
        .unwrap();
    alternative[rank_from_a] -= 1;
    alternative[rank_from_b] += 1;
    let sampled = game.with_hidden_assignment(0, alternative).unwrap();
    assert_ne!(
        sampled.serialize_state().unwrap(),
        game.serialize_state().unwrap()
    );
    assert_eq!(sampled.observe(0).unwrap(), game.observe(0).unwrap());

    let mut invalid = alternative;
    invalid[rank_from_a] = invalid[rank_from_a].saturating_add(1);
    assert!(game.with_hidden_assignment(0, invalid).is_err());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn random_legal_games_terminate_with_cards_conserved(
        keys in proptest::collection::vec(any::<u64>(), 54),
        selectors in proptest::collection::vec(any::<usize>(), 170),
        landlord in 0_u8..3,
    ) {
        let mut permutation = (0_u8..54).collect::<Vec<_>>();
        permutation.sort_unstable_by_key(|&card| (keys[usize::from(card)], card));
        let (dealt_hands, bottom) = deal_from_permutation(&permutation);
        let mut hands = [EMPTY_RANK_COUNTS; 3];
        hands[usize::from(landlord)] = dealt_hands[0];
        hands[usize::from((landlord + 1) % 3)] = dealt_hands[1];
        hands[usize::from((landlord + 2) % 3)] = dealt_hands[2];
        let rules = douzero_rules();
        let mut game = PostBidGame::new(hands, bottom, landlord, rules).unwrap();

        assert_state_invariants(&game, &hands);
        for selector in selectors {
            if game.is_terminal() {
                break;
            }
            let legal = game.legal_actions().unwrap();
            prop_assert!(!legal.is_empty());
            let action = legal[selector % legal.len()];
            game.step(&action).unwrap();
            assert_state_invariants(&game, &hands);
        }

        prop_assert!(game.is_terminal());
        prop_assert!(game.state().hands.iter().any(|hand| hand.iter().all(|&count| count == 0)));
        prop_assert_eq!(game.state().raw_payoff.iter().sum::<i32>(), 0);
        prop_assert!(game.legal_actions().unwrap().is_empty());
    }
}
