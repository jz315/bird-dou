use crate::Rank;

pub(crate) fn rank_strength(rank: Rank, level: Rank) -> u8 {
    if rank == Rank::BigJoker {
        15
    } else if rank == Rank::SmallJoker {
        14
    } else if rank == level {
        13
    } else {
        rank.natural_index()
            .expect("non-joker ranks have a natural index")
    }
}
