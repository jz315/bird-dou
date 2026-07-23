mod common;

use ddz_core::Seat;
use ddz_rules::RuleProfile;

use ddz_batch::{BatchEnv, BatchError, ResetSpec, SlotReset};

use common::{huanle_rules, post_bid_batch};

#[test]
fn full_and_partial_reset_are_transactional_and_versioned() {
    let mut batch = post_bid_batch(3);
    let versions_before = (0..3)
        .map(|slot| batch.version(slot).unwrap())
        .collect::<Vec<_>>();
    let state0 = batch.encoded_states().expect("encode");

    let observation = batch
        .reset_slots(&[SlotReset {
            slot: 1,
            spec: ResetSpec::post_bid(999, Seat::ONE),
        }])
        .expect("partial reset");
    assert_eq!(observation.batch_size, 3);
    assert_eq!(batch.version(0), Some(versions_before[0]));
    assert_ne!(batch.version(1), Some(versions_before[1]));
    assert_eq!(batch.version(2), Some(versions_before[2]));

    let before_failed_reset = batch.encoded_states().expect("encode");
    let error = batch
        .reset_slots(&[
            SlotReset {
                slot: 0,
                spec: ResetSpec::post_bid(1, Seat::ZERO),
            },
            SlotReset {
                slot: 0,
                spec: ResetSpec::post_bid(2, Seat::ZERO),
            },
        ])
        .expect_err("duplicate reset must fail");
    assert!(matches!(error, BatchError::DuplicateSlot { slot: 0 }));
    assert_eq!(batch.encoded_states().unwrap(), before_failed_reset);
    assert_ne!(batch.encoded_states().unwrap(), state0);
}

#[test]
fn reset_variant_must_match_shared_profile() {
    let mut batch = BatchEnv::new(huanle_rules(200)).expect("rules");
    let error = batch
        .reset_all(&[ResetSpec::post_bid(1, Seat::ZERO)])
        .expect_err("wrong reset variant");
    assert!(matches!(
        error,
        BatchError::ResetProfileMismatch {
            profile: RuleProfile::HuanleClassic,
            ..
        }
    ));

    let observation = batch
        .reset_all(&[ResetSpec::huanle(1), ResetSpec::huanle(2)])
        .expect("Huanle reset");
    assert_eq!(observation.batch_size, 2);
    assert_eq!(batch.active_count(), 2);
}
