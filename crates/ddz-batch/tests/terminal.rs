mod common;

use common::post_bid_batch;

#[test]
fn terminal_transition_emits_reward_once_and_raw_payoff_is_zero_sum() {
    let mut batch = post_bid_batch(1);
    let mut terminal_result = None;
    for _ in 0..1_024 {
        if batch.all_terminal() {
            break;
        }
        let actions = batch.legal_actions_packed().expect("actions");
        let start = usize::try_from(actions.offsets[0]).unwrap();
        let end = usize::try_from(actions.offsets[1]).unwrap();
        let global = (start..end)
            .max_by_key(|&index| actions.actions.move_total_cards[index])
            .expect("legal action");
        let local = i64::try_from(global - start).unwrap();
        let result = batch.step_packed(&[local]).expect("step");
        if result.became_terminal[0] == 1 {
            terminal_result = Some(result);
            break;
        }
    }
    let result = terminal_result.expect("deterministic policy must terminate");
    assert_eq!(result.done, vec![1]);
    assert!(result.reward.iter().any(|value| *value != 0));
    assert_eq!(result.raw_payoff.iter().sum::<i64>(), 0);

    let skipped = batch.step_packed(&[-1]).expect("skip terminal");
    assert_eq!(skipped.reward, vec![0, 0, 0]);
    assert_eq!(skipped.done, vec![1]);
}
