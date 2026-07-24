use guandan_rules::{Action, Card, Rank, Round, Seat};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::legal::apply_action_index;
use crate::model::{kind_code, EventView};
use crate::python::{to_python, value_error};
use crate::state::build_state;

#[pyclass(module = "birddou._guandan_native")]
pub struct PyGuandanGame {
    pub(crate) round: Round,
    pub(crate) human_seat: Seat,
    pub(crate) history: Vec<EventView>,
}

#[pymethods]
impl PyGuandanGame {
    #[new]
    fn new(seed: u64, level: u8, human_seat: u8) -> PyResult<Self> {
        let level = Rank::STANDARD
            .get(usize::from(level))
            .copied()
            .ok_or_else(|| value_error("level must be in 0..13"))?;
        let human_seat = Seat::new(human_seat).map_err(value_error)?;
        let first_player = Seat::new((seed % 4) as u8).expect("seed modulo four is a seat");
        let round = Round::new(seed, level, first_player).map_err(value_error)?;
        Ok(Self {
            round,
            human_seat,
            history: Vec::new(),
        })
    }

    fn state<'py>(&self, py: Python<'py>, observer: u8) -> PyResult<Bound<'py, PyAny>> {
        let observer = Seat::new(observer).map_err(value_error)?;
        let state = build_state(&self.round, self.human_seat, observer, &self.history)
            .map_err(value_error)?;
        to_python(py, &state, "Guandan state")
    }

    fn step_action<'py>(
        &mut self,
        py: Python<'py>,
        actor: u8,
        action_index: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let actor = Seat::new(actor).map_err(value_error)?;
        let result =
            apply_action_index(&mut self.round, actor, action_index).map_err(value_error)?;
        self.record(actor, &result);
        self.state(py, u8::from(self.human_seat))
    }

    fn play_cards<'py>(
        &mut self,
        py: Python<'py>,
        actor: u8,
        card_ids: Option<Vec<u8>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let actor = Seat::new(actor).map_err(value_error)?;
        let action = match card_ids {
            Some(ids) => Action::Play(
                ids.into_iter()
                    .map(Card::from_id)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(value_error)?,
            ),
            None => Action::Pass,
        };
        let result = self.round.step(actor, action).map_err(value_error)?;
        self.record(actor, &result);
        self.state(py, u8::from(self.human_seat))
    }

    #[getter]
    fn current_player(&self) -> Option<u8> {
        self.round
            .outcome()
            .is_none()
            .then_some(u8::from(self.round.current_player()))
    }

    #[getter]
    fn terminal(&self) -> bool {
        self.round.outcome().is_some()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyGuandanGame(level={:?}, current_player={:?}, terminal={})",
            self.round.level(),
            self.current_player(),
            self.terminal()
        )
    }
}

impl PyGuandanGame {
    fn record(&mut self, actor: Seat, result: &guandan_rules::StepResult) {
        let (kind, cards) = result
            .played
            .as_ref()
            .map_or(("pass", Vec::new()), |movement| {
                (kind_code(*movement.kind()), movement.cards().to_vec())
            });
        self.history.push(EventView {
            sequence: self.history.len(),
            actor,
            kind,
            cards,
        });
    }
}
