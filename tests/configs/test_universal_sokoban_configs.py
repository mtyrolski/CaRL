from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]


def test_existing_sokoban_ada_solve_config_backwards_compatible_shape():
    cfg = OmegaConf.load(ROOT / "configs/solve/sokoban/sokoban_ada_solve.yaml")
    assert cfg.subgoal_generator._target_ == "carl.inference_components.subgoal_generator.AdaptiveSubgoalGenerator"
    assert "generator_k_list" in cfg.subgoal_generator
    # Existing config should remain valid without explicitly setting mode.
    assert "mode" not in cfg.subgoal_generator


def test_universal_solve_and_train_configs_parse_and_expose_expected_switches():
    solve_cfg = OmegaConf.load(ROOT / "configs/solve/sokoban/sokoban_universal_ada_solve.yaml")
    assert solve_cfg.subgoal_generator._target_ == "carl.inference_components.subgoal_generator.UniversalPropositionalSubgoalGenerator"
    assert solve_cfg.subgoal_generator.mode == "propositional_universal"
    assert len(solve_cfg.carl_grid) == 3

    raw_cfg = OmegaConf.load(ROOT / "configs/offline_training/sokoban/sokoban_train_universal_raw.yaml")
    ft_cfg = OmegaConf.load(ROOT / "configs/offline_training/sokoban/sokoban_train_universal_finetune.yaml")
    con_cfg = OmegaConf.load(ROOT / "configs/offline_training/sokoban/sokoban_train_universal_contrastive.yaml")
    teacher_cfg = OmegaConf.load(ROOT / "configs/offline_training/sokoban/sokoban_generate_universal_teacher.yaml")

    assert raw_cfg.algorithm.recipe == "raw"
    assert raw_cfg.algorithm.datamodule.recipe == "raw"
    assert ft_cfg.algorithm.recipe == "finetune"
    assert ft_cfg.algorithm.datamodule.recipe == "finetune"
    assert con_cfg.algorithm.recipe == "contrastive"
    assert con_cfg.algorithm.datamodule.recipe == "contrastive"
    assert teacher_cfg.algorithm._target_ == "carl.algorithms.generate_universal_teacher_annotations.GenerateUniversalTeacherAnnotations"
