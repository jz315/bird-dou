# Migration from the previous ddz-rules

This is intentionally a breaking cleanup. Delete, do not adapt, these previous concepts:

```text
RuleConfigV1 / RuleConfigV2
VersionedRuleConfig
CanonicalFullLegacyV1
BiddingMode::Score
match_v2.rs
PhaseV2 / GameActionV2 / MatchStateV2
```

Use:

```text
RuleProfile::DouzeroPostBid
RuleProfile::HuanleClassic
```

against the existing unsuffixed core domain model.

## API replacements

```text
old PostBidGame::reset/...       -> Game::new_post_bid / Game::step
old HuanleMatchV2                -> Game::new_huanle
old generate_*(&RankCounts, ...) -> generate_*(RankCounts, ...)
old versioned config wrapper     -> RuleConfig::from_yaml_str
```

Rebuild old smoke artifacts and replays. The project has not yet promised a stable public
format, so retaining two state machines would create more risk than regenerating those
artifacts.

## Workspace impact

This crate deliberately removes the old APIs, so `ddz-batch`, `ddz-pyo3`, Python
feature encoders, and the web adapter must be migrated in later tickets. Validate this
crate first with `cargo test -p ddz-rules`; do not add compatibility shims back into the
rule layer merely to keep old callers compiling.
