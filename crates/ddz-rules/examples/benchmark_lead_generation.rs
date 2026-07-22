use std::hint::black_box;
use std::time::Instant;

use ddz_core::{RankCounts, EMPTY_RANK_COUNTS};
use ddz_rules::{generate_lead_moves, RuleConfig};

const DOUZERO_POST_BID_YAML: &str = include_str!("../../../configs/rules/douzero_post_bid.yaml");
const ITERATIONS: u32 = 10_000;

fn benchmark_hand() -> RankCounts {
    let mut hand = EMPTY_RANK_COUNTS;
    hand[..9].copy_from_slice(&[4, 4, 3, 3, 2, 1, 1, 1, 1]);
    hand
}

fn main() {
    let rules = RuleConfig::from_yaml_str(DOUZERO_POST_BID_YAML)
        .expect("checked-in DouZero profile must be valid");
    let hand = benchmark_hand();
    let actions_per_generation = generate_lead_moves(&hand, &rules)
        .expect("benchmark hand must be valid")
        .len();

    let started = Instant::now();
    let mut total_actions = 0_u32;
    for _ in 0..ITERATIONS {
        let generated_actions = black_box(
            generate_lead_moves(black_box(&hand), black_box(&rules))
                .expect("benchmark generation must succeed")
                .len(),
        );
        total_actions +=
            u32::try_from(generated_actions).expect("one hand's action count must fit in u32");
    }
    let elapsed = started.elapsed();
    let generations_per_second = f64::from(ITERATIONS) / elapsed.as_secs_f64();

    println!("profile=douzero_post_bid");
    println!("hand_cards={}", hand.iter().sum::<u8>());
    println!("actions_per_generation={actions_per_generation}");
    println!("iterations={ITERATIONS}");
    println!("elapsed_seconds={:.6}", elapsed.as_secs_f64());
    println!("generations_per_second={generations_per_second:.2}");
    println!(
        "actions_per_second={:.2}",
        f64::from(total_actions) / elapsed.as_secs_f64()
    );
}
