use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use serde::Serialize;

pub(crate) fn to_python<'py, T: Serialize>(
    py: Python<'py>,
    value: &T,
    label: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let json = serde_json::to_string(value)
        .map_err(|error| PyRuntimeError::new_err(format!("failed to encode {label}: {error}")))?;
    py.import("json")?.call_method1("loads", (json,))
}

pub(crate) fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}
