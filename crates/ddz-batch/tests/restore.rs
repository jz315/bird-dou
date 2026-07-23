mod common;

use ddz_batch::{BatchEnv, BatchError};

use common::{post_bid_batch, post_bid_rules};

#[test]
fn exact_snapshot_restores_states_and_slot_versions() {
    let mut source = post_bid_batch(2);
    source.step_packed(&[0, 0]).expect("step");
    let snapshot = source.snapshot().expect("snapshot");
    let states = source.encoded_states().expect("states");
    let versions = [source.version(0).unwrap(), source.version(1).unwrap()];

    let mut restored = BatchEnv::new(post_bid_rules(100)).expect("same rules");
    restored.restore_snapshot(&snapshot).expect("restore");
    assert_eq!(restored.encoded_states().unwrap(), states);
    assert_eq!(restored.version(0), Some(versions[0]));
    assert_eq!(restored.version(1), Some(versions[1]));
}

#[test]
fn snapshot_rejects_different_rules() {
    let source = post_bid_batch(1);
    let snapshot = source.snapshot().expect("snapshot");
    let mut different = BatchEnv::new(post_bid_rules(999)).expect("rules");
    let error = different
        .restore_snapshot(&snapshot)
        .expect_err("rules mismatch");
    assert!(matches!(error, BatchError::SnapshotRulesMismatch { .. }));
}
