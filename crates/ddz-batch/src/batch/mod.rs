mod env;
mod legal;
mod observe;
mod reset;
mod restore;
mod step;

pub use env::BatchEnv;
pub use restore::{BatchSnapshot, SlotSnapshot};

use ddz_rules::Game;

use crate::SlotVersion;

#[derive(Clone, Debug)]
pub(crate) struct Slot {
    pub(crate) game: Game,
    pub(crate) version: SlotVersion,
}
