use ddz_core::{Rank, RankCounts, RANK_COUNT};

use crate::AttachmentMultiplicity;

pub(super) fn attachment_capacities(hand: RankCounts, body: RankCounts) -> [u8; RANK_COUNT] {
    std::array::from_fn(|index| {
        let rank = Rank::ALL[index];
        if body[rank] == 0 {
            hand[rank] - body[rank]
        } else {
            0
        }
    })
}

pub(super) fn allocations(
    capacities: [u8; RANK_COUNT],
    required_units: u8,
    cards_per_unit: u8,
    multiplicity: AttachmentMultiplicity,
) -> Vec<RankCounts> {
    let unit_capacities = std::array::from_fn(|index| {
        let available = capacities[index] / cards_per_unit;
        match multiplicity {
            AttachmentMultiplicity::DistinctRanks => available.min(1),
            AttachmentMultiplicity::MayShareRank => available,
        }
    });
    let mut result = Vec::new();
    let mut current = [0_u8; RANK_COUNT];
    enumerate(
        &unit_capacities,
        required_units,
        cards_per_unit,
        0,
        &mut current,
        &mut result,
    );
    result
}

fn enumerate(
    unit_capacities: &[u8; RANK_COUNT],
    units_left: u8,
    cards_per_unit: u8,
    rank_index: usize,
    current: &mut [u8; RANK_COUNT],
    result: &mut Vec<RankCounts>,
) {
    if units_left == 0 {
        result.push(
            RankCounts::new(*current)
                .expect("attachment allocations never exceed physical rank capacity"),
        );
        return;
    }
    if rank_index == RANK_COUNT {
        return;
    }
    let remaining_capacity: u16 = unit_capacities[rank_index..]
        .iter()
        .map(|value| u16::from(*value))
        .sum();
    if remaining_capacity < u16::from(units_left) {
        return;
    }

    let maximum = unit_capacities[rank_index].min(units_left);
    for units in 0..=maximum {
        current[rank_index] = units * cards_per_unit;
        enumerate(
            unit_capacities,
            units_left - units,
            cards_per_unit,
            rank_index + 1,
            current,
            result,
        );
    }
    current[rank_index] = 0;
}

pub(super) fn combine(body: RankCounts, attachments: RankCounts) -> RankCounts {
    body.checked_add(attachments)
        .expect("generated body and attachments fit in one physical hand")
}
