# Artifacts

Only lightweight, auditable manifests belong in version control. Model weights,
datasets, and generated evaluation outputs are ignored by default.

`baselines/douzero/manifest.toml` pins the source-only E008 reference. Its fetched
`source/` checkout remains ignored and is recreated by
`scripts/fetch_douzero_baseline.py`; no upstream code or weights are committed.

`baselines/perfectdou/manifest.toml` likewise pins the official evaluation commit
and checksums its ONNX models plus CPython-3.7 Linux binary helpers. The ignored
checkout is recreated and verified by `scripts/fetch_perfectdou_baseline.py`.
