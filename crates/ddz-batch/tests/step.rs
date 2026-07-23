mod common;

use ddz_batch::{BatchError, SKIP_ACTION_INDEX};

use common::post_bid_batch;

#[test]
fn invalid_local_index_mutates_no_slot() {
    let mut batch = post_bid_batch(2);
    let before = batch.encoded_states().expect("encode");
    let error = batch
        .step_packed(&[0, i64::MAX])
        .expect_err("out-of-range index");
    assert!(matches!(error, BatchError::ActionIndexOutOfRange { slot: 1, .. }));
    assert_eq!(batch.encoded_states().unwrap(), before);
}

#[test]
fn checked_step_rejects_stale_policy_results() {
    let mut batch = post_bid_batch(2);
    let actions = batch.legal_actions_packed().expect("actions");
    let old_generation = actions.generation.clone();
    let old_revision = actions.revision.clone();

    let first = batch
        .step_packed_checked(&[0, 0], &old_generation, &old_revision)
        .expect("first step");
    assert_eq!(first.transitioned, vec![1, 1]);
    assert_eq!(first.revision, vec![1, 1]);

    let error = batch
        .step_packed_checked(&[0, 0], &old_generation, &old_revision)
        .expect_err("stale revisions");
    assert!(matches!(error, BatchError::StaleRevision { slot: 0, .. }));
}

#[test]
fn skip_is_allowed_for_active_slots_and_preserves_versions() {
    let mut batch = post_bid_batch(2);
    let before = [batch.version(0).unwrap(), batch.version(1).unwrap()];
    let result = batch
        .step_packed(&[SKIP_ACTION_INDEX, SKIP_ACTION_INDEX])
        .expect("skip");
    assert_eq!(result.transitioned, vec![0, 0]);
    assert_eq!(batch.version(0), Some(before[0]));
    assert_eq!(batch.version(1), Some(before[1]));
}

#[test]
fn checked_step_ignores_stale_versions_for_explicitly_skipped_slots() {
    let mut batch = post_bid_batch(2);
    let actions = batch.legal_actions_packed().expect("actions");
    batch.step_packed(&[-1, 0]).expect("advance only slot one");

    let result = batch
        .step_packed_checked(
            &[0, SKIP_ACTION_INDEX],
            &actions.generation,
            &actions.revision,
        )
        .expect("stale masked slot must not reject an unrelated transition");
    assert_eq!(result.transitioned, vec![1, 0]);
}
