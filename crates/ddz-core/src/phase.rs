use serde::{Deserialize, Serialize};

/// Complete Huanle-style game flow. DouZero post-bid starts directly at `CardPlay`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    PreDeal,
    Dealing,
    Calling,
    Robbing,
    BottomReveal,
    PostBottomReveal,
    Doubling,
    CardPlay,
    Terminal,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Unassigned,
    Landlord,
    Farmer,
}
