from __future__ import annotations

import json
from dataclasses import fields

import numpy as np
import pytest

from flyarm.grasp.objects import load_split, split_objects
from flyarm.manipulation import furniture as fu
from flyarm.manipulation import tasks as tk
from flyarm.manipulation.sim import RewardConfig, stalling_floor
from flyarm.manipulation.splits import SPLITS, record, record_path


def _bigrams(template: tk.Template) -> set[tuple[str, str]]:
    signature = template.signature()
    return set(zip(signature, signature[1:], strict=False))


def test_committed_split_record_matches_the_code() -> None:
    committed = json.loads(record_path().read_text())
    assert committed == json.loads(json.dumps(record()))


def test_unseen_objects_are_the_grasp_tasks_held_out_objects() -> None:
    split = load_split()
    assert SPLITS["unseen_objects"].objects == "test"
    assert all(SPLITS[name].objects == "train" for name in ("train", "iid_test"))
    train = {item.name for item in split_objects("train")}
    test = {item.name for item in split_objects("test")}
    assert not train & test and test == set(split["test"])


def test_held_out_furniture_ranges_do_not_overlap_the_training_ranges() -> None:
    for field in fields(fu.FurnitureRanges):
        train, held = getattr(fu.TRAIN_RANGES, field.name), getattr(fu.HELD_OUT_RANGES, field.name)
        if isinstance(train, fu.Range):
            assert train.high <= held.low or held.high <= train.low, field.name
    assert not set(fu.TRAIN_RANGES.lid_handles) & set(fu.HELD_OUT_RANGES.lid_handles)


def test_held_out_compositions_never_appear_in_training() -> None:
    train = [tk.TEMPLATES[name] for name in tk.TRAIN_TEMPLATES]
    seen_signatures = {template.signature() for template in train}
    seen_pairs = set().union(*(_bigrams(template) for template in train))
    skills = {step for template in train for step in template.signature()}
    for name in tk.HELD_OUT_TEMPLATES:
        template = tk.TEMPLATES[name]
        assert template.signature() not in seen_signatures
        assert _bigrams(template) - seen_pairs, f"{name} has no unseen ordering"
        assert set(template.signature()) <= skills, f"{name} uses a skill never trained"
        assert 3 <= len(template.steps) <= tk.MAX_SUBGOALS
    assert not set(tk.TRAIN_TEMPLATES) & set(tk.HELD_OUT_TEMPLATES)


def test_every_template_length_and_a_long_composite() -> None:
    lengths = [len(template.steps) for template in tk.TEMPLATES.values()]
    assert min(lengths) >= 3 and max(lengths) == 8
    assert any(len(tk.TEMPLATES[name].steps) >= 6 for name in tk.TRAIN_TEMPLATES)


def test_consumers_keep_an_opening_done_once_what_needed_it_is_done() -> None:
    put_away = [
        tk.Subgoal(tk.OPEN_DRAWER, target=0),
        tk.Subgoal(tk.PLACE, 0, 0),
        tk.Subgoal(tk.CLOSE_DRAWER, target=0),
    ]
    assert tk.consumers(put_away) == [[1], [], []]
    retrieve = [
        tk.Subgoal(tk.OPEN_DRAWER, target=1),
        tk.Subgoal(tk.PICK, 0),
        tk.Subgoal(tk.PLACE, 0, tk.BIN),
        tk.Subgoal(tk.CLOSE_DRAWER, target=1),
    ]
    assert tk.consumers(retrieve) == [[1], [2], [], []]
    shelve = [
        tk.Subgoal(tk.OPEN_DOOR),
        tk.Subgoal(tk.PLACE, 0, tk.SHELF),
        tk.Subgoal(tk.PLACE, 1, tk.SHELF, 1),
        tk.Subgoal(tk.CLOSE_DOOR),
    ]
    assert tk.consumers(shelve) == [[1, 2], [], [], []]


def _fits(item, config: fu.FurnitureConfig, receptacle: int, shelf: int) -> bool:
    if receptacle in (tk.DRAWER_0, tk.DRAWER_1):
        return tk.fits_drawer(item, config)
    if receptacle == tk.SHELF:
        return tk.fits_shelf(item, config, shelf)
    if receptacle == tk.BIN:
        return tk.fits_bin(item, config)
    return tk.fits_region(item, config)


@pytest.mark.parametrize("split", list(SPLITS))
def test_sampled_episodes_bind_objects_that_fit_their_receptacles(split: str) -> None:
    chosen = SPLITS[split]
    objects = split_objects(chosen.objects)
    for seed in range(12):
        generator = np.random.default_rng(seed)
        episode = tk.sample_episode(generator, chosen.templates, chosen.ranges, objects)
        config = episode.furniture
        assert episode.template in chosen.templates
        shelf = sum(
            1 for goal in episode.subgoals if goal.kind == tk.PLACE and goal.target == tk.SHELF
        )
        for goal in episode.subgoals:
            if goal.kind == tk.PLACE:
                item = objects[episode.objects[goal.obj]]
                assert _fits(item, config, goal.target, shelf), (episode.template, item.name)
            if goal.kind == tk.STACK:
                top, base = (
                    objects[episode.objects[goal.obj]],
                    objects[episode.objects[goal.target]],
                )
                assert tk.stackable(top) and tk.stackable(base) and tk.stable_on(top, base)
            if goal.kind in (tk.OPEN_DRAWER, tk.CLOSE_DRAWER):
                assert goal.target < config.drawer_count
        used = [index for index in episode.objects if index >= 0]
        assert 2 <= len(used) <= tk.MAX_OBJECTS and len(set(used)) == len(used)


def test_reward_bonus_exceeds_the_stalling_floor_and_the_shaping() -> None:
    assert stalling_floor(0.99) == 0.0  # every level term is a penalty
    config = RewardConfig()
    assert config.subgoal_bonus > config.shaping > 0
    with pytest.raises(ValueError, match="shaping"):
        RewardConfig(subgoal_bonus=5.0, shaping=10.0)
    with pytest.raises(ValueError, match="stalling floor"):
        RewardConfig(subgoal_bonus=0.0)
