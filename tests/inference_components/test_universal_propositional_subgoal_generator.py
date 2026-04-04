from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from carl.solver.nodes import SearchTreeNode


class DummyUniversalTokenizer:
    def x_y_tokenizer(self, x, y, training_goal):  # noqa: ARG002
        _ = (x, y)
        return torch.tensor([[7, 8, 9]], dtype=torch.long), torch.tensor([[7, 8, 9]], dtype=torch.long)

    def board_detokenizer(self, sequence_of_tokens):
        if not sequence_of_tokens:
            return None
        return f"sg_{sequence_of_tokens[-1]}"


class DummyUniversalEnv:
    def __init__(self) -> None:
        self.tokenizer = DummyUniversalTokenizer()


class DummyHFGenerator(torch.nn.Module):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):  # noqa: ARG003
        assert Path(pretrained_model_name_or_path).exists()
        return cls()

    def generate(self, input_ids, **kwargs):  # noqa: ARG002
        return torch.tensor([[10, 11, 12], [10, 11, 13], [10, 11, 13]], dtype=torch.long)


def test_universal_propositional_generator_hydra_instantiation_and_smoke(tmp_path: Path):
    ckpt_dir = tmp_path / "checkpoint-1"
    ckpt_dir.mkdir()
    (ckpt_dir / "config.json").write_text("{}")
    (ckpt_dir / "model.safetensors").write_bytes(b"stub")

    cfg = OmegaConf.create(
        {
            "_target_": "carl.inference_components.subgoal_generator.UniversalPropositionalSubgoalGenerator",
            "mode": "propositional_universal",
            "env": {"_target_": "tests.inference_components.test_universal_propositional_subgoal_generator.DummyUniversalEnv"},
            "path_to_generator_weights": str(ckpt_dir),
            "subgoal_generation_kwargs": {
                "max_new_tokens": 5,
                "num_beams": 2,
                "num_return_sequences": 3,
                "do_sample": False,
                "temperature": 1.0,
            },
            "generator_network_class": {
                "_target_": "tests.inference_components.test_universal_propositional_subgoal_generator.DummyHFGenerator.from_pretrained",
                "_partial_": True,
            },
        }
    )

    generator = hydra.utils.instantiate(cfg)
    generator.construct_network()

    node = SearchTreeNode(state="root", value=0.0, low_level_path=[], parent_node=None, metadata={"depth": 2})
    proposals = generator.get_subgoals(node)

    assert generator.mode == "propositional_universal"
    assert generator.generator_k_list == []
    assert len(proposals) == 3
    assert proposals[0].state == "sg_12"
    assert "proposal_confidence" in proposals[0].generation_metadata
    assert "proposal_rank" in proposals[0].generation_metadata
    assert proposals[1].generation_metadata["proposal_rank"] == 1
