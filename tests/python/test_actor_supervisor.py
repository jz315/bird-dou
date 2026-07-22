"""Spawned actor lifecycle, bounded IPC, and crash-recovery tests."""

from __future__ import annotations

from pathlib import Path

from birddou.actors import (
    ActorProcessSupervisor,
    ActorSupervisorConfig,
    ActorWorkerContext,
    load_actor_supervisor_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ActorItem = tuple[int, int, int, int]
ActorPayload = tuple[str, int]


def _test_actor_worker(
    context: ActorWorkerContext[ActorItem],
    payload: ActorPayload,
) -> None:
    mode, item_count = payload
    if mode == "crash_once" and context.generation == 0:
        raise RuntimeError("injected actor crash")
    for item_index in range(item_count):
        if context.stop.is_set():
            return
        context.trajectories.put(
            (context.actor_id, context.generation, context.envs_per_actor, item_index),
            timeout=5.0,
        )


def _config(*, actors: int, capacity: int = 4, restarts: int = 1) -> ActorSupervisorConfig:
    return ActorSupervisorConfig(
        actor_processes=actors,
        envs_per_actor=3,
        trajectory_queue_capacity=capacity,
        max_restarts_per_actor=restarts,
        process_join_timeout_s=2.0,
        queue_poll_timeout_s=0.02,
        start_method="spawn",
    )


def test_repository_actor_topology_is_explicit() -> None:
    config = load_actor_supervisor_config(
        REPOSITORY_ROOT / "configs" / "train" / "actor_system.yaml"
    )
    assert config.actor_processes == 8
    assert config.envs_per_actor == 32
    assert config.trajectory_queue_capacity == 256
    assert config.start_method == "spawn"


def test_single_and_multiple_spawned_actors_publish_identity_and_env_fanout() -> None:
    for actor_count in (1, 2):
        supervisor = ActorProcessSupervisor(
            _config(actors=actor_count), _test_actor_worker, ("ok", 1)
        )
        try:
            supervisor.start()
            items = tuple(supervisor.get_trajectory(timeout=5.0) for _ in range(actor_count))
            assert {item[0] for item in items} == set(range(actor_count))
            assert all(item[1:] == (0, 3, 0) for item in items)
            assert supervisor.stats().configured_processes == actor_count
        finally:
            supervisor.shutdown()


def test_actor_crash_restarts_with_new_generation_without_deadlock() -> None:
    supervisor = ActorProcessSupervisor(
        _config(actors=1, restarts=1),
        _test_actor_worker,
        ("crash_once", 1),
    )
    try:
        supervisor.start()
        assert supervisor.get_trajectory(timeout=8.0) == (0, 1, 3, 0)
        stats = supervisor.stats()
        assert stats.crash_exits == 1
        assert stats.restart_count == 1
        assert stats.permanently_failed_actors == ()
    finally:
        supervisor.shutdown()


def test_full_interprocess_queue_applies_backpressure_and_stays_bounded() -> None:
    supervisor = ActorProcessSupervisor(
        _config(actors=1, capacity=1),
        _test_actor_worker,
        ("ok", 2),
    )
    try:
        supervisor.start()
        first = supervisor.get_trajectory(timeout=5.0)
        second = supervisor.get_trajectory(timeout=5.0)
        assert first[-1] == 0
        assert second[-1] == 1
        assert supervisor.stats().maximum_observed_queue_size <= 1
    finally:
        supervisor.shutdown()
