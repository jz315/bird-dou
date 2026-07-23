use ddz_batch::{BatchEnv, ResetSpec};
use ddz_core::Seat;
use ddz_rules::{RewardMode, RuleConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rules = RuleConfig::douzero_post_bid(1, RewardMode::WinPercentage);
    let mut batch = BatchEnv::new(rules)?;
    batch.reset_all(&[
        ResetSpec::post_bid(100, Seat::ZERO),
        ResetSpec::post_bid(101, Seat::ONE),
    ])?;

    let actions = batch.legal_actions_packed()?;
    let selected = [0_i64, 0_i64];
    let result = batch.step_packed_checked(
        &selected,
        &actions.generation,
        &actions.revision,
    )?;

    println!("transitioned={:?}", result.transitioned);
    println!("revision={:?}", result.revision);
    Ok(())
}
