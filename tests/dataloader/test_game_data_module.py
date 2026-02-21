from pathlib import Path

import joblib
import torch
import pytest

from carl.dataloader.game_data_module import GameDataModule
from carl.environment.training_goal import TrainingGoal
from carl.planners.base import Experience
from carl.planners.base import SearchInfo
from carl.planners.base import Solution
from carl.solver.nodes import SearchTreeNode


def _to_int(value) -> int:
    if isinstance(value, tuple):
        return sum(_to_int(v) for v in value)
    if isinstance(value, str):
        return int(value) if value.lstrip("-").isdigit() else len(value)
    return int(value)


class TinyTokenizer:
    def x_y_tokenizer(self, x, y, training_goal):  # noqa: ARG002
        return (
            torch.tensor([[[_to_int(x)]]], dtype=torch.long),
            torch.tensor([[[_to_int(y)]]], dtype=torch.long),
        )


class TinyEnv:
    name = "tiny"

    def __init__(self) -> None:
        self.tokenizer = TinyTokenizer()
        self.state = 0

    def detect_action(self, before, after):
        return _to_int(after) - _to_int(before)

    def restore_full_state_from_np_array_version(self, state):
        self.state = _to_int(state)

    def get_state(self):
        return self.state

    def step(self, action):
        self.state += int(action)
        return self.state, 0.0, False, {}


class BadEnv:
    name = "bad"

    def __init__(self) -> None:
        self.tokenizer = TinyTokenizer()

    def detect_action(self, before, after):  # noqa: ARG002
        return 0


def _make_experience() -> Experience:
    root = SearchTreeNode(0, 0.0, [], None, metadata={"depth": 0})
    node_a = SearchTreeNode(3, 0.0, [1, 1, 1], root, next_expand_with_k_generator=3)
    node_b = SearchTreeNode(5, 0.0, [1, 1], node_a, next_expand_with_k_generator=2)
    return Experience(
        solution=Solution(
            solved=True,
            subgoal_path=[],
            action_path=[],
            subgoal_distance_path=[3, 2],
        ),
        search_info=SearchInfo(finished_reason="solved", solving_node=node_b),
    )


def _module(
    tmp_path: Path,
    *,
    env=None,
    training_goal: TrainingGoal | str = TrainingGoal.GENERATOR,
    dataset_path: str | Path | None = None,
    experiences_path: str | Path | None = None,
    untokenized_data=None,
    subgoal_distance_interval=None,
    generator_target_mode=None,
    generator_k=None,
    generator_offsets=None,
    for_testing: bool = False,
    validation_split: float = 0.5,
) -> GameDataModule:
    return GameDataModule(
        env=env or TinyEnv(),
        dataset_path=dataset_path,
        save_tokenized_dataset_path=str(tmp_path / "tokenized"),
        training_goal=training_goal,
        untokenized_data=untokenized_data,
        experiences_path=experiences_path,
        subgoal_distance_interval=subgoal_distance_interval,
        generator_target_mode=generator_target_mode,
        generator_k=generator_k,
        generator_offsets=generator_offsets,
        validation_split=validation_split,
        num_workers=0,
        for_testing=for_testing,
    )


def test_load_experiences_flattens_nested_content_and_applies_limit(tmp_path: Path):
    exp1 = _make_experience()
    exp2 = _make_experience()
    path = tmp_path / "exp.joblib"
    joblib.dump([exp1, [exp2, None], "ignore"], path)

    dm = _module(
        tmp_path,
        experiences_path=path,
        subgoal_distance_interval=[2],
    )
    dm.num_of_trajectories = 1

    loaded = dm._load_experiences()
    assert len(loaded) == 1
    assert isinstance(loaded[0], Experience)


def test_load_experiences_supports_glob_pattern(tmp_path: Path, monkeypatch):
    exp = _make_experience()
    file_a = tmp_path / "exp_a.joblib"
    file_b = tmp_path / "exp_b.joblib"
    joblib.dump([exp], file_a)
    joblib.dump([exp], file_b)
    monkeypatch.chdir(tmp_path)

    dm = _module(
        tmp_path,
        experiences_path="exp_*.joblib",
        subgoal_distance_interval=[2],
    )
    loaded = dm._load_experiences()
    assert len(loaded) == 2


def test_get_env_with_restore_raises_when_methods_missing(tmp_path: Path):
    dm = _module(
        tmp_path,
        env=BadEnv(),
        experiences_path=tmp_path / "exp.joblib",
        subgoal_distance_interval=[2],
    )
    with pytest.raises(ValueError, match="restore_full_state_from_np_array_version"):
        dm._get_env_with_restore()


def test_generator_tokenize_from_experiences_sliding_window(tmp_path: Path):
    exp = _make_experience()
    path = tmp_path / "exp.joblib"
    joblib.dump([exp], path)

    dm = _module(
        tmp_path,
        experiences_path=path,
        subgoal_distance_interval=[2],
        generator_target_mode="sliding_window",
    )
    x_tensors, y_tensors = dm._generator_tokenize_from_experiences()

    assert set(x_tensors.keys()) == {0}
    assert x_tensors[0].shape[0] == 5
    assert y_tensors[0].shape[0] == 5
    assert torch.equal(y_tensors[0].flatten(), torch.tensor([2, 3, 4, 5, 5]))


def test_generator_tokenize_from_experiences_k_offsets(tmp_path: Path):
    exp = _make_experience()
    path = tmp_path / "exp.joblib"
    joblib.dump([exp], path)

    dm = _module(
        tmp_path,
        experiences_path=path,
        subgoal_distance_interval=[2],
        generator_target_mode="k_offsets",
        generator_k=3,
        generator_offsets=[0],
    )
    x_tensors, y_tensors = dm._generator_tokenize_from_experiences()

    assert x_tensors[0].shape[0] == 1
    assert y_tensors[0].shape[0] == 1
    assert int(x_tensors[0].flatten()[0].item()) == 0
    assert int(y_tensors[0].flatten()[0].item()) == 3


def test_generator_tokenize_from_experiences_k_offsets_requires_k(tmp_path: Path):
    exp = _make_experience()
    path = tmp_path / "exp.joblib"
    joblib.dump([exp], path)

    dm = _module(
        tmp_path,
        experiences_path=path,
        subgoal_distance_interval=[2],
        generator_target_mode="k_offsets",
        generator_k=None,
    )

    with pytest.raises(ValueError, match="generator_k must be provided"):
        dm._generator_tokenize_from_experiences()


def test_generator_tokenize_all_pairs_when_distance_interval_is_none(tmp_path: Path):
    dm = _module(
        tmp_path,
        untokenized_data={0: ["0", "1", "2", "3"]},
        subgoal_distance_interval=None,
    )
    x_tensors, y_tensors = dm._generator_tokenize({0: ["0", "1", "2", "3"]})
    assert x_tensors[0].shape[0] == 6
    assert y_tensors[0].shape[0] == 6


def test_prepare_data_for_testing_writes_all_dataset(tmp_path: Path):
    dm = _module(
        tmp_path,
        training_goal=TrainingGoal.GENERATOR,
        untokenized_data={0: ["0", "1", "2"]},
        subgoal_distance_interval=[1],
        for_testing=True,
    )

    dm.prepare_data()

    output = dm.save_tokenized_dataset_path / f"{dm.env.name}_{dm.training_goal.value}_tokenized_all_x_y"
    assert output.exists()
    x_all, y_all = joblib.load(output)
    assert x_all.shape[0] == 2
    assert y_all.shape[0] == 2


def test_prepare_data_train_val_and_setup_fit(tmp_path: Path):
    dm = _module(
        tmp_path,
        training_goal=TrainingGoal.GENERATOR,
        untokenized_data={
            0: ["0", "1", "2"],
            1: ["1", "2", "3"],
        },
        subgoal_distance_interval=[1],
        validation_split=0.5,
    )

    dm.prepare_data()
    dm.setup("fit")

    train_dataset = dm.get_train_dataset()
    val_dataset = dm.get_val_dataset()
    assert len(train_dataset) > 0
    assert len(val_dataset) > 0
