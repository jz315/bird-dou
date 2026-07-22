# License Audit

## E001 inventory

The repository contains only original project scaffolding and the user-supplied
implementation specification. No third-party source code, model weights, datasets,
or generated artifacts are vendored by E001.

Any future baseline import must record its source repository, pinned commit,
license, weight provenance, and redistribution terms here before it is merged.

## E002 Rust dependencies

E002 adds YAML configuration parsing. Versions are reproducibly pinned by
`Cargo.lock`; no dependency source is vendored.

| Dependency | Relationship | Locked version | Declared license |
|---|---|---:|---|
| `serde` | Direct | 1.0.229 | MIT OR Apache-2.0 |
| `serde_yaml_ng` | Direct | 0.10.0 | MIT |
| `equivalent` | Transitive | 1.0.2 | Apache-2.0 OR MIT |
| `hashbrown` | Transitive | 0.17.1 | MIT OR Apache-2.0 |
| `indexmap` | Transitive | 2.14.0 | Apache-2.0 OR MIT |
| `itoa` | Transitive | 1.0.18 | MIT OR Apache-2.0 |
| `proc-macro2` | Transitive | 1.0.107 | MIT OR Apache-2.0 |
| `quote` | Transitive | 1.0.47 | MIT OR Apache-2.0 |
| `ryu` | Transitive | 1.0.23 | Apache-2.0 OR BSL-1.0 |
| `serde_core` | Transitive | 1.0.229 | MIT OR Apache-2.0 |
| `serde_derive` | Transitive | 1.0.229 | MIT OR Apache-2.0 |
| `syn` | Transitive | 3.0.2 | MIT OR Apache-2.0 |
| `unicode-ident` | Transitive | 1.0.24 | (MIT OR Apache-2.0) AND Unicode-3.0 |
| `unsafe-libyaml` | Transitive | 0.2.11 | MIT |

This inventory was generated from Cargo package metadata. It must be refreshed when
`Cargo.lock` changes materially.

## E008 differential baseline and JSON protocol

E008 does not vendor `DouZero` source or weights. The differential workflow fetches
the Apache-2.0-licensed upstream repository into the ignored local artifact cache
and verifies commit `718a5c920bf3361e34178a38f3b80458e176b351` before importing
its game engine. The tracked
[`manifest.toml`](../artifacts/baselines/douzero/manifest.toml) records repository,
commit, license, required files, and the fact that no weights are needed.

The Rust probe adds `serde_json` for its persistent JSON-lines subprocess protocol;
E009 also uses it as the direct runtime state-envelope codec in `ddz-core`. Its
lockfile additions are:

| Dependency | Locked version | Declared license |
|---|---:|---|
| `serde_json` | 1.0.151 | MIT OR Apache-2.0 |
| `memchr` | 2.8.3 | Unlicense OR MIT |
| `zmij` | 1.0.23 | MIT |

No third-party source from either dependency is vendored.

## E013 official DouZero checkpoints and PyTorch

E013 continues the external-artifact policy: neither the DouZero source nor its
model checkpoints are committed or redistributed. The official repository README
publishes ADP and WP download locations. Because the original Google Drive folder
is no longer available anonymously, the fetcher uses the byte-identical files in
the public `palemoky/douzero-baselines` mirror pinned at revision
`57b3914046c2a0877016b8b8830fd07cf5b0ba08`. The tracked manifest records both
original distribution URLs, every file size, and every SHA-256 digest. Any missing
or mismatched file stops inference.

The upstream source remains pinned to Apache-2.0 commit
`718a5c920bf3361e34178a38f3b80458e176b351`. The upstream project does not state
separate checkpoint redistribution terms in its README, so BIRD-Dou conservatively
downloads weights only into the ignored local artifact cache and makes no license
claim for those files.

PyTorch is an optional runtime dependency declared as `torch>=2.6,<3.0`; it is
used to load the official state dictionaries in weights-only mode and perform
inference. PyTorch binaries and source are resolved by the installer and are not
vendored.

E014 adds an original, interoperability-oriented NumPy feature encoder and
PyTorch module definitions matching the documented upstream dimensions and state
dictionary interface. No upstream Python file is copied into the package. The
Apache-2.0 pinned source remains an external test oracle; production-native
inference no longer imports it. E014 adds no dependency beyond the already
declared optional PyTorch runtime.

E015 adds original actor, loss, checkpoint, and CLI code using the same optional
PyTorch runtime. It adds no third-party dependency and does not package training
outputs or official checkpoints; generated runs remain under the ignored
`artifacts/` tree.

E016 adds original feature-packing and exact hand-decomposition code. The
`ddz-search` crate depends only on the workspace's existing `ddz-core`,
`ddz-rules`, and Serde packages; no new external dependency or artifact is added.

E017 adds original PyTorch rank-embedding, convolution, SwiGLU, normalization,
stochastic-depth, and relative-attention modules. It uses the already declared
optional PyTorch runtime and adds no dependency or external model artifact.

E018 adds original event-embedding, GRU, causal-Transformer, scalar-encoding, and
seat-gating modules using the same optional PyTorch runtime. No dependency or
third-party model artifact is added.

E019 adds original ragged segment reductions, action/post-hand encoders,
categorical action metadata embeddings, rank cross-attention, and legal-set
context fusion. It uses the existing optional PyTorch runtime and adds no
dependency or external model artifact.

E020 adds original role/seat adapters, multi-task output heads, structured DMC
collection, losses, checkpointing, and CLI code. It reuses the existing NumPy and
optional PyTorch dependencies; no external source, dataset, weight, or new package
is added.

M5 adds original cardinality-DP, marginal, sampler, calibration, Belief-fusion,
dataset, and offline-training code. Generated `.npz` data and checkpoints remain
under ignored `artifacts/`; no dataset or model weight is redistributed and no new
dependency is introduced.

M6 adds original full-hand Teacher, privileged critic, Oracle Dropout, and IS-KD
code. It reuses PyTorch and the project's own generated training labels; no new
dependency or third-party artifact is added.

M7 adds original bounded queue, asynchronous versioned inference, compact replay,
V-trace, Hybrid-loss, policy-lag, learner, and fair-comparison code. It uses only
Python's standard library and the existing optional PyTorch runtime; no dependency,
third-party source, dataset, weight, or generated artifact is added.

M8 adds original Farmer Team Critic, counterfactual loss, specialist optimizer,
native-state rollout data, exploiter schedule, and promotion-gate code. It reuses
the existing Rust engine and optional PyTorch runtime; no new dependency, external
source, dataset, model weight, or generated artifact is introduced.

## E010 Python binding dependencies

E010 uses PyO3 to expose the Rust environment as an ABI3 extension built by
Maturin. PyO3 is linked as a library; no dependency source or Python runtime is
vendored. The newly locked Rust packages are:

| Dependency | Locked version | Declared license |
|---|---:|---|
| `pyo3` | 0.29.0 | MIT OR Apache-2.0 |
| `pyo3-build-config` | 0.29.0 | MIT OR Apache-2.0 |
| `pyo3-ffi` | 0.29.0 | MIT OR Apache-2.0 |
| `pyo3-macros` | 0.29.0 | MIT OR Apache-2.0 |
| `pyo3-macros-backend` | 0.29.0 | MIT OR Apache-2.0 |
| `heck` | 0.5.0 | MIT OR Apache-2.0 |
| `once_cell` | 1.21.4 | MIT OR Apache-2.0 |
| `portable-atomic` | 1.14.0 | Apache-2.0 OR MIT |
| `target-lexicon` | 0.13.5 | Apache-2.0 WITH LLVM-exception |

The Python build backend is `maturin>=1.10,<2.0`, resolved in an isolated PEP 517
build environment rather than vendored or recorded in `Cargo.lock`.

## E011 NumPy transport dependencies

E011 adds the PyO3-compatible Rust `numpy` crate and the Python runtime constraint
`numpy>=1.26,<2.5`. The upper bound retains the project's declared Python 3.11
support because NumPy 2.5 drops Python 3.11. No NumPy source or binary is vendored.

| Rust dependency | Locked version | Declared license |
|---|---:|---|
| `numpy` | 0.29.0 | BSD-2-Clause |
| `ndarray` | 0.17.2 | MIT OR Apache-2.0 |
| `matrixmultiply` | 0.3.11 | MIT OR Apache-2.0 |
| `num-complex` | 0.4.6 | MIT OR Apache-2.0 |
| `num-integer` | 0.1.46 | MIT OR Apache-2.0 |
| `portable-atomic-util` | 0.2.7 | Apache-2.0 OR MIT |
| `rawpointer` | 0.2.1 | MIT OR Apache-2.0 |
| `rustc-hash` | 2.1.3 | Apache-2.0 OR MIT |

The locally resolved Python NumPy 2.4.6 distribution declares
`BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0`; downstream installations
resolve a compatible release inside the declared range.

## M9 dependency audit

Complete bidding, three-container dynamic programming, Bid Head training, and the
complete Arena add no third-party dependencies. They use the workspace's existing
Rust/PyO3/Serde and optional PyTorch/NumPy stacks. No upstream source, model weight,
or dataset is copied into the repository by M9.

## M10 dependency audit

Proposal, root-consistent Belief rollout, the native exact solver, both distillation
losses, and deployment bundle handling are original project code. They reuse the
existing Rust standard library/Serde stack and optional PyTorch/NumPy runtime. M10
adds no third-party dependency, source, model weight, or redistributed dataset.
Generated benchmark reports and deployment bundles remain under ignored artifact
paths.

## External RLCard and PerfectDou evaluation baselines

RLCard 1.0.7 is an optional PyPI dependency declared under the `rlcard` extra. Its
upstream project is MIT licensed. BIRD-Dou imports only the installed public rule
agent interface and contains an original observation/action adapter; no RLCard
source or model artifact is committed here. The refreshed `uv.lock` pins RLCard
1.0.7 and its `termcolor` 3.3.0 dependency; termcolor declares MIT.

The official PerfectDou evaluation release is Apache-2.0 licensed and pinned to
commit `594404922ee3810e2d84b80bb2c2846cb20e5390`. The tracked
[`manifest.toml`](../artifacts/baselines/perfectdou/manifest.toml) records the
repository, license, three ONNX checkpoint sizes and SHA-256 digests, and the two
binary encoder/helper digests. Source, weights, and shared objects are fetched
only into the ignored local artifact cache and are not redistributed.

PerfectDou publishes evaluation code and pretrained weights, but its feature
engineering and left-hand modules are binary-only CPython-3.7 x86-64 Linux shared
objects. BIRD-Dou therefore communicates with a separately provisioned compatible
runtime over an original, versioned JSONL adapter. No compatibility or
redistribution claim is made for those upstream binaries on other platforms.

## E004 property-test dependencies

E004 adds `proptest` 1.11.0 as a test-only dependency with default features
disabled and only its `std` feature enabled. Its newly locked transitive packages
are listed below; several are target-conditional and are not built on every host.

| Dependency | Locked version | Declared license |
|---|---:|---|
| `proptest` | 1.11.0 | MIT OR Apache-2.0 |
| `autocfg` | 1.5.1 | Apache-2.0 OR MIT |
| `bitflags` | 2.13.1 | MIT OR Apache-2.0 |
| `cfg-if` | 1.0.4 | MIT OR Apache-2.0 |
| `getrandom` | 0.3.4 | MIT OR Apache-2.0 |
| `libc` | 0.2.189 | MIT OR Apache-2.0 |
| `num-traits` | 0.2.19 | MIT OR Apache-2.0 |
| `ppv-lite86` | 0.2.21 | MIT OR Apache-2.0 |
| `r-efi` | 5.3.0 | MIT OR Apache-2.0 OR LGPL-2.1-or-later |
| `rand` | 0.9.5 | MIT OR Apache-2.0 |
| `rand_chacha` | 0.9.0 | MIT OR Apache-2.0 |
| `rand_core` | 0.9.5 | MIT OR Apache-2.0 |
| `rand_xorshift` | 0.4.0 | MIT OR Apache-2.0 |
| `regex-syntax` | 0.8.11 | MIT OR Apache-2.0 |
| `syn` | 2.0.119 | MIT OR Apache-2.0 |
| `unarray` | 0.1.4 | MIT OR Apache-2.0 |
| `wasip2` | 1.0.4+wasi-0.2.12 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| `wit-bindgen` | 0.57.1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| `zerocopy` | 0.8.55 | BSD-2-Clause OR Apache-2.0 OR MIT |
| `zerocopy-derive` | 0.8.55 | BSD-2-Clause OR Apache-2.0 OR MIT |
