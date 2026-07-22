"""End-to-end acceptance tests for the E015 DMC smoke-training loop."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import numpy as np
import torch

from birddou import PyDdzEnv
from birddou.actors import collect_dmc_episode
from birddou.cli.policy_artifacts import load_dmc_checkpoint_policy
from birddou.eval.baselines import PolicyDecisionContext
from birddou.eval.paired_deals import SEAT_ROLES, SeatRole, role_for_game_seat
from birddou.rl.dmc import DmcTrainer, load_dmc_config
from birddou.rl.losses import dmc_value_loss

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "train" / "dmc_smoke.yaml"


def test_one_episode_trains_all_roles_and_writes_auditable_checkpoint(tmp_path: Path) -> None:
    """Every chosen action receives its seat's terminal return and finite update."""
    config = replace(
        load_dmc_config(CONFIG_PATH),
        episodes=1,
        checkpoint_every=1,
        output_directory=tmp_path / "one-episode",
        evaluation_deals=2,
        bootstrap_resamples=100,
    )
    trainer = DmcTrainer(config)
    episode = collect_dmc_episode(
        15015,
        trainer.rules,
        trainer.models,
        np.random.default_rng(15015),
        epsilon=0.05,
        policy_version=0,
    )
    assert episode.action_count == len(episode.transitions)
    assert {item.role for item in episode.transitions} == set(SeatRole)
    assert all(item.target == episode.objective_payoff[item.seat] for item in episode.transitions)
    assert all(np.isfinite(item.behavior_logprob) for item in episode.transitions)
    assert all(item.serialized_state for item in episode.transitions)
    result = trainer.train()

    assert result.state.episodes == 1
    assert result.state.frames > 0
    assert result.state.learner_updates == 3
    assert result.state.policy_version == 1
    assert result.state.frames == sum(
        (
            result.state.landlord_frames,
            result.state.landlord_down_frames,
            result.state.landlord_up_frames,
        )
    )
    assert all(np.isfinite(value) and value >= 0.0 for value in result.role_losses.values())
    assert len(result.metrics_history) == 1
    assert result.checkpoint_path.is_file()
    assert result.manifest_path.is_file()
    manifest = cast(
        dict[str, object],
        json.loads(result.manifest_path.read_text(encoding="utf-8")),
    )
    assert manifest["trainer_mode"] == "dmc"
    assert manifest["optimizer_state"] is True
    assert manifest["scheduler_state"] is True
    assert manifest["amp_scaler_state"] is True
    assert manifest["rng_state"] is True
    assert isinstance(manifest["league_snapshot"], str)
    assert (config.output_directory / "league.json").is_file()
    assert manifest["episodes"] == 1
    assert manifest["metrics_file"] == "metrics.jsonl"
    assert (
        manifest["checkpoint_sha256"]
        == hashlib.sha256(result.checkpoint_path.read_bytes()).hexdigest()
    )
    assert all((config.output_directory / f"{role.value}.ckpt").is_file() for role in SEAT_ROLES)
    metrics_lines = (
        (config.output_directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert len(metrics_lines) == 1
    assert json.loads(metrics_lines[0])["episode"] == 1

    loaded = load_dmc_checkpoint_policy("loaded-dmc", result.checkpoint_path, "cpu")
    environment = PyDdzEnv()
    observation = environment.reset(4321, trainer.rules)
    seat = observation["observer"]
    assert (
        loaded.select_action(
            observation,
            environment.legal_actions(),
            PolicyDecisionContext(
                0,
                4321,
                "loaded-dmc",
                seat,
                role_for_game_seat(seat, 0),
                0,
            ),
        )
        >= 0
    )

    evaluation = trainer.evaluate_against_random()
    assert evaluation.beats_random
    assert evaluation.report.match_count == 12
    assert (
        evaluation.report.overall.win_rate.candidate_mean
        > evaluation.report.overall.win_rate.baseline_mean
    )


def test_checkpoint_resume_is_bit_exact_for_the_next_update(tmp_path: Path) -> None:
    """A restored learner produces the same second episode as uninterrupted training."""
    base = replace(
        load_dmc_config(CONFIG_PATH),
        episodes=2,
        checkpoint_every=1,
        output_directory=tmp_path / "continuous",
        evaluation_deals=1,
        bootstrap_resamples=20,
    )
    continuous = DmcTrainer(base)
    continuous.train(1)

    resumed = DmcTrainer(replace(base, output_directory=tmp_path / "resumed"))
    resumed.load_checkpoint(continuous.checkpoint_path)
    assert asdict(resumed.state) == asdict(continuous.state)
    assert resumed.role_losses == continuous.role_losses
    assert resumed.metrics_history == continuous.metrics_history
    assert resumed.league.to_dict() == continuous.league.to_dict()
    assert resumed.rng.bit_generator.state == continuous.rng.bit_generator.state
    for role in SEAT_ROLES:
        for key, value in continuous.models[role].state_dict().items():
            assert torch.equal(value, resumed.models[role].state_dict()[key])
        assert resumed.schedulers[role].state_dict() == continuous.schedulers[role].state_dict()
        assert (
            resumed.optimizers[role].state_dict()["param_groups"]
            == continuous.optimizers[role].state_dict()["param_groups"]
        )

    continuous_result = continuous.train(1)
    resumed_result = resumed.train(1)
    assert asdict(resumed_result.state) == asdict(continuous_result.state)
    assert resumed_result.role_losses == continuous_result.role_losses
    assert resumed_result.metrics_history == continuous_result.metrics_history
    for role in SEAT_ROLES:
        for key, value in continuous.models[role].state_dict().items():
            assert torch.equal(value, resumed.models[role].state_dict()[key])


def test_dmc_losses_validate_shapes_and_backpropagate_finite_gradients() -> None:
    """Both configured regression losses have an explicit finite gradient contract."""
    for name in ("mse", "huber"):
        prediction = torch.tensor([[0.5], [-0.25]], requires_grad=True)
        target = torch.tensor([1.0, -1.0])
        loss = dmc_value_loss(prediction, target, name)
        torch.autograd.backward((loss,))

        assert torch.isfinite(loss)
        assert prediction.grad is not None
        assert torch.isfinite(prediction.grad).all()

    with np.testing.assert_raises_regex(ValueError, "differs"):
        dmc_value_loss(torch.zeros((2, 1)), torch.zeros(3), "mse")


def test_default_smoke_config_declares_100_complete_games() -> None:
    """The checked-in gate is the requested 100-game single-actor smoke run."""
    config = load_dmc_config(CONFIG_PATH)

    assert config.trainer_mode == "dmc"
    assert config.episodes == 100
    assert config.initialization == "douzero_ADP"
    assert config.require_beats_random
    assert config.device == "cpu"
    assert config.loss == "mse"
    assert set(SEAT_ROLES) == set(SeatRole)
