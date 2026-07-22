"""Fixed-deal and seat-rotation acceptance tests for E012."""

import pytest

from birddou.eval.paired_deals import (
    SEAT_ROLES,
    SeatRole,
    generate_cross_play_schedule,
    generate_paired_comparisons,
    generate_paired_deals,
    role_for_seat,
)


def test_fixed_deal_generator_has_golden_unique_seeds() -> None:
    """The versioned seed stream is stable, unique, and reproducible."""
    first = generate_paired_deals(20260722, 5)
    second = generate_paired_deals(20260722, 5)
    other = generate_paired_deals(20260723, 5)

    assert first == second
    assert first != other
    assert [deal.seed for deal in first.deals] == [
        9380731410078756576,
        2086612728407527173,
        9772240184971607783,
        17247963054820208864,
        3870543711815643814,
    ]
    assert len({deal.seed for deal in first.deals}) == first.count
    assert first.to_dict()["master_seed"] == 20260722


@pytest.mark.parametrize("seat", [-1, 3, 99])
def test_role_lookup_rejects_python_negative_and_out_of_range_indices(seat: int) -> None:
    """Only the three actual seats map to formal evaluation roles."""
    with pytest.raises(ValueError, match="outside 0..=2"):
        role_for_seat(seat)


def test_paired_schedule_rotates_policy_identity_over_one_identical_deal() -> None:
    """Every role receives a symmetric A/B pair without redrawing the cards."""
    deal_set = generate_paired_deals(7, 1)
    comparisons = generate_paired_comparisons(deal_set, "A", "B")

    assert tuple(item.focal_role for item in comparisons) == SEAT_ROLES
    assert all(item.deal is deal_set.deals[0] for item in comparisons)
    for comparison in comparisons:
        seat = comparison.focal_seat
        assert comparison.candidate_assignment.policy_ids[seat] == "A"
        assert comparison.baseline_assignment.policy_ids[seat] == "B"
        assert all(
            policy_id == "B"
            for index, policy_id in enumerate(comparison.candidate_assignment.policy_ids)
            if index != seat
        )
        assert all(
            policy_id == "A"
            for index, policy_id in enumerate(comparison.baseline_assignment.policy_ids)
            if index != seat
        )
    assert comparisons[2].focal_role is SeatRole.LANDLORD_UP


def test_cross_play_schedule_is_complete_and_stable() -> None:
    """Each ordered matrix cell uses every fixed deal exactly once."""
    deal_set = generate_paired_deals(9, 2)
    schedule = generate_cross_play_schedule(deal_set, ("L1", "L2"), ("F1", "F2"))

    assert len(schedule) == 8
    assert schedule == generate_cross_play_schedule(deal_set, ("L1", "L2"), ("F1", "F2"))
    assert [match.assignment.policy_ids for match in schedule[:2]] == [
        ("L1", "F1", "F1"),
        ("L1", "F1", "F1"),
    ]
