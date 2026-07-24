#![forbid(unsafe_code)]
#![allow(
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate
)]
#![doc = "Python boundary for the authoritative Guandan rules engine."]

mod game;
mod legal;
mod model;
mod python;
mod state;

use pyo3::prelude::*;

pub use game::PyGuandanGame;

#[pymodule]
#[pyo3(name = "_guandan_native")]
fn native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyGuandanGame>()?;
    module.add("API_SCHEMA_VERSION", 1)?;
    Ok(())
}
