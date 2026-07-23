mod common;

use ddz_core::{CallAction, DoubleAction, GameAction, Phase, RevealAction, RobAction};
use ddz_rules::EconomyContext;

use ddz_batch::protocol::{ACTION_DOUBLE, PackedEvents};
use ddz_batch::{BatchEnv, ResetSpec};

use common::{huanle_rules, local_index, post_bid_batch};

#[test]
fn packed_action_ranges_and_fixed_observation_shapes_are_consistent() {
    let mut batch = post_bid_batch(4);
    let observation = batch.observations_current().expect("observation");
    observation.validate().expect("packed invariant");
    assert_eq!(observation.batch_size, 4);
    assert_eq!(observation.cards.own_hand.len(), 4 * 15);
    assert_eq!(observation.card_play.played_cards.len(), 4 * 3 * 15);

    let actions = batch.legal_actions_packed().expect("actions");
    actions.validate().expect("packed action invariant");
    assert_eq!(actions.offsets.len(), 5);
    assert!(actions.local_count(0).unwrap() > 0);
    assert_eq!(actions.owner.len(), actions.action_count());
}

#[test]
fn unresolved_double_choices_are_absent_from_public_history() {
    let rules = huanle_rules(300);
    let mut batch = BatchEnv::new(rules).expect("rules");
    batch
        .reset_all(&[ResetSpec::Huanle {
            match_seed: 55,
            economy: EconomyContext::unlimited(),
        }])
        .expect("reset");

    while matches!(batch.state(0).unwrap().phase, Phase::PreDeal | Phase::Dealing) {
        let actions = batch.legal_actions_packed().unwrap();
        let index = local_index(
            &actions,
            0,
            |action| matches!(action, GameAction::Reveal(RevealAction::Continue)),
            &mut batch,
        );
        batch.step_packed(&[index]).unwrap();
    }
    assert_eq!(batch.state(0).unwrap().phase, Phase::Calling);
    let actions = batch.legal_actions_packed().unwrap();
    let call = local_index(
        &actions,
        0,
        |action| matches!(action, GameAction::Call(CallAction::CallLandlord)),
        &mut batch,
    );
    batch.step_packed(&[call]).unwrap();
    while batch.state(0).unwrap().phase == Phase::Robbing {
        let actions = batch.legal_actions_packed().unwrap();
        let pass = local_index(
            &actions,
            0,
            |action| matches!(action, GameAction::Rob(RobAction::Pass)),
            &mut batch,
        );
        batch.step_packed(&[pass]).unwrap();
    }
    assert_eq!(batch.state(0).unwrap().phase, Phase::PostBottomReveal);
    let actions = batch.legal_actions_packed().unwrap();
    let decline_reveal = local_index(
        &actions,
        0,
        |action| matches!(action, GameAction::Reveal(RevealAction::Continue)),
        &mut batch,
    );
    batch.step_packed(&[decline_reveal]).unwrap();
    assert_eq!(batch.state(0).unwrap().phase, Phase::Doubling);

    let actions = batch.legal_actions_packed().unwrap();
    let double = local_index(
        &actions,
        0,
        |action| matches!(action, GameAction::Double(DoubleAction::Double)),
        &mut batch,
    );
    batch.step_packed(&[double]).unwrap();

    let authoritative = batch.authoritative_history_packed().unwrap();
    let public = batch.public_history_packed().unwrap();
    assert!(contains_action_kind(&authoritative, ACTION_DOUBLE));
    assert!(!contains_action_kind(&public, ACTION_DOUBLE));
}

fn contains_action_kind(events: &PackedEvents, action_kind: u8) -> bool {
    events
        .kind
        .iter()
        .zip(events.actions.kind.iter())
        .any(|(&event_kind, &kind)| event_kind == 0 && kind == action_kind)
}
