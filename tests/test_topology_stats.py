from __future__ import annotations

import pytest

from flyarm.whole_brain.stats import sign_flip_p, topology_statistics


def _model(kind: str, seed: int, lifted: list[bool], replicate: int | None = None) -> dict:
    episodes = [{"ever_grasped": True, "ever_lifted": value, "success": False} for value in lifted]
    return {
        "kind": kind,
        "seed": seed,
        "shuffle_replicate": replicate,
        "clean": {"episodes": episodes},
    }


def test_sign_flip_p_is_exact() -> None:
    assert sign_flip_p([1.0, 1.0, 1.0]) == pytest.approx(1 / 8)
    assert sign_flip_p([0.5] * 6) == pytest.approx(1 / 64)
    assert sign_flip_p([1.0, -1.0]) == pytest.approx(3 / 4)


def test_replicates_are_averaged_within_a_seed_and_paired_by_episode() -> None:
    models = [
        _model("connectome", 0, [True, True, False, False]),
        _model("shuffled", 0, [True, False, False, False], 0),
        _model("shuffled", 0, [False, False, True, False], 1),
        _model("connectome", 1, [True, False, False, False]),
        _model("shuffled", 1, [True, True, False, False], 0),
    ]
    lift = topology_statistics(models)["outcomes"]["lift"]
    first, second = lift["per_seed"]
    assert first["shuffled"] == [1, 1] and first["difference"] == pytest.approx(0.25)
    assert second["difference"] == pytest.approx(-0.25)
    # Seed 0: replicate 0 loses episode 1; replicate 1 loses episodes 0, 1 and wins 2.
    # Seed 1: the shuffle wins episode 1.
    assert lift["episode_pairs"] == {"measured_only": 3, "shuffled_only": 2}
    assert lift["seeds_measured_better"] == 1 and lift["seeds_measured_worse"] == 1
