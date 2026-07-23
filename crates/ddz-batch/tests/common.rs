#![allow(dead_code)]

use ddz_core::{GameAction, Seat};
use ddz_rules::{RewardMode, RuleConfig};

use ddz_batch::{BatchEnv, PackedActions, ResetSpec};

pub fn post_bid_rules(id: u32) -> RuleConfig {
    RuleConfig::douzero_post_bid(id, RewardMode::WinPercentage)
}

pub fn huanle_rules(id: u32) -> RuleConfig {
    RuleConfig::huanle_classic(id, [0; 18])
}

pub fn post_bid_batch(size: usize) -> BatchEnv {
    let mut batch = BatchEnv::new(post_bid_rules(100)).expect("valid rules");
    let specs = (0..size)
        .map(|index| ResetSpec::post_bid(u64::try_from(index).expect("test index fits in u64") + 10, Seat::ZERO))
        .collect::<Vec<_>>();
    batch.reset_all(&specs).expect("valid reset");
    batch
}

pub fn local_index(
    actions: &PackedActions,
    slot: usize,
    predicate: impl Fn(GameAction) -> bool,
    batch: &mut BatchEnv,
) -> i64 {
    let count = actions.local_count(slot).expect("slot range");
    (0..count)
        .find(|&index| {
            batch
                .legal_action(slot, index)
                .expect("legal action query")
                .is_some_and(|action| predicate(action))
        })
        .and_then(|index| i64::try_from(index).ok())
        .expect("matching legal action")
}
