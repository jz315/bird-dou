use ddz_core::{Move, MoveKind};

/// Return whether `candidate` legally outranks the active non-pass `target` by shape and rank.
#[must_use]
pub fn move_beats(candidate: Move, target: Move) -> bool {
    match (candidate.kind(), target.kind()) {
        (_, MoveKind::Rocket) => false,
        (MoveKind::Rocket, _) => true,
        (MoveKind::Bomb, MoveKind::Bomb) => candidate.main_rank() > target.main_rank(),
        (MoveKind::Bomb, _) => true,
        (_, MoveKind::Bomb) => false,
        _ => {
            candidate.kind() == target.kind()
                && candidate.chain_len() == target.chain_len()
                && candidate.main_rank() > target.main_rank()
        }
    }
}
