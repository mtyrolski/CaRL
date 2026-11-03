"""Check conditioning and positional capacity without GPU or private weights."""
import torch
from pathlib import Path
from transformers import BartForConditionalGeneration
from scripts.training.train_policy_relative_bart import ScaleBart


def test_scratch_loads_no_weights_and_supports_conditioning_and_long_input(monkeypatch):
    torch.set_num_threads(2)
    def unexpected_weight_load(*args,**kwargs):
        raise AssertionError('Scratch must never load pretrained generator weights')
    monkeypatch.setattr(BartForConditionalGeneration,'from_pretrained',unexpected_weight_load)
    config=Path(__file__).resolve().parents[2]/'configs/generator/sokoban_bart_small.json'
    model=ScaleBart(config,'scratch','relative').eval()
    assert sum(p.numel() for p in model.parameters())==7_375_616
    with torch.no_grad():
        model.scale_projection.weight.fill_(.1)
        tokens=torch.full((1,290),15,dtype=torch.long)
        low=model.embedding(tokens,torch.tensor([.2]))
        high=model.embedding(tokens,torch.tensor([.8]))
        torch.testing.assert_close(high-low,torch.full_like(low,.06))
        model.condition='none'
        torch.testing.assert_close(model.embedding(tokens,torch.tensor([.2])),
                                   model.embedding(tokens,torch.tensor([.8])))
        result=model(tokens,torch.tensor([.5]),torch.tensor([[15,16,17]]))
        assert result.shape==(1,3,22) and torch.isfinite(result).all()
