use std::hint::black_box;
use std::time::Instant;

use ddz_core::{cards_to_rank_counts, GameAction, Move, RankCounts};
use ddz_rules::{PostBidGame, RuleConfig};

const DOUZERO_POST_BID_YAML: &str = include_str!("../../../configs/rules/douzero_post_bid.yaml");
const ITERATIONS: u32 = 50_000;

fn benchmark_state() -> PostBidGame {
    let cards = (0_u8..54).collect::<Vec<_>>();
    let hands: [RankCounts; 3] = [
        cards_to_rank_counts(&cards[..20]).unwrap(),
        cards_to_rank_counts(&cards[20..37]).unwrap(),
        cards_to_rank_counts(&cards[37..]).unwrap(),
    ];
    let bottom = cards_to_rank_counts(&cards[17..20]).unwrap();
    let rules = RuleConfig::from_yaml_str(DOUZERO_POST_BID_YAML).unwrap();
    let mut game = PostBidGame::new(hands, bottom, 0, rules).unwrap();

    while game.state().history.len() < 45 {
        let legal = game.legal_actions().unwrap();
        let action = if legal.contains(&GameAction::Play(Move::pass())) {
            GameAction::Play(Move::pass())
        } else {
            legal[0]
        };
        game.step(&action).unwrap();
    }
    game
}

fn main() {
    let base = benchmark_state();
    let action = base.legal_actions().unwrap()[0];

    let enumerate_started = Instant::now();
    for _ in 0..ITERATIONS {
        let legal = black_box(&base).legal_actions().unwrap();
        black_box(legal.contains(black_box(&action)));
    }
    let enumerate_elapsed = enumerate_started.elapsed();

    let clone_started = Instant::now();
    for _ in 0..ITERATIONS {
        let mut branch = black_box(base.clone());
        branch.step(black_box(&action)).unwrap();
        black_box(branch.state().cards_left);
    }
    let clone_elapsed = clone_started.elapsed();

    let mut reversible = base.clone();
    let undo_started = Instant::now();
    for _ in 0..ITERATIONS {
        let token = reversible.apply_in_place(black_box(&action)).unwrap();
        black_box(reversible.state().cards_left);
        reversible.undo(black_box(&token)).unwrap();
    }
    let undo_elapsed = undo_started.elapsed();

    println!("history_events={}", base.state().history.len());
    println!("iterations={ITERATIONS}");
    println!(
        "enumerate_validation_seconds={:.6}",
        enumerate_elapsed.as_secs_f64()
    );
    println!("clone_step_seconds={:.6}", clone_elapsed.as_secs_f64());
    println!("apply_undo_seconds={:.6}", undo_elapsed.as_secs_f64());
    println!(
        "clone_step_ops_per_second={:.2}",
        f64::from(ITERATIONS) / clone_elapsed.as_secs_f64()
    );
    println!(
        "apply_undo_ops_per_second={:.2}",
        f64::from(ITERATIONS) / undo_elapsed.as_secs_f64()
    );
    println!(
        "apply_undo_vs_clone_speedup={:.3}",
        clone_elapsed.as_secs_f64() / undo_elapsed.as_secs_f64()
    );
    println!(
        "apply_undo_vs_enumerate_validation_speedup={:.3}",
        enumerate_elapsed.as_secs_f64() / undo_elapsed.as_secs_f64()
    );
}
