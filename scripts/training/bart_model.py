"""Scalar-conditioned BART architecture shared by the training CLI and offline tests."""
import torch
from torch import nn
from transformers import BartConfig, BartForConditionalGeneration


class ScaleBart(nn.Module):
    def __init__(self, model_source, initialization, condition):
        super().__init__()
        if initialization not in {"scratch", "pretrained"}:
            raise ValueError("initialization must be scratch or pretrained")
        config = BartConfig.from_pretrained(model_source, local_files_only=True)
        expected = dict(
            d_model=256,
            encoder_layers=3,
            decoder_layers=3,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=2048,
            decoder_ffn_dim=1024,
        )
        assert all(getattr(config, k) == v for k, v in expected.items()), "Use What Matters Table 2 architecture"
        if initialization == "pretrained":
            self.bart = BartForConditionalGeneration.from_pretrained(model_source, local_files_only=True)
            for module in [self.bart.model.encoder, self.bart.model.decoder]:
                original = module.embed_positions
                extended = type(original)(512, config.d_model)
                nn.init.normal_(extended.weight, mean=0, std=config.init_std)
                with torch.no_grad():
                    extended.weight[: len(original.weight)].copy_(original.weight)
                module.embed_positions = extended
            self.bart.config.max_position_embeddings = 512
        else:
            config.max_position_embeddings = 512
            self.bart = BartForConditionalGeneration(config)
        self.bart.config.use_cache = False
        self.scale_projection = nn.Linear(1, 256, bias=False)
        nn.init.zeros_(self.scale_projection.weight)
        self.condition = condition

    def embedding(self, tokens, scale):
        if self.condition == "none":
            scale = torch.zeros_like(scale)
        return self.bart.model.shared(tokens) + self.scale_projection(scale[:, None])[:, None, :]

    def forward(self, tokens, scale, labels):
        return self.bart(inputs_embeds=self.embedding(tokens, scale), labels=labels, use_cache=False).logits

    @torch.inference_mode()
    def generate(self, tokens, scale):
        self.eval()
        return self.bart.generate(
            inputs_embeds=self.embedding(tokens, scale),
            max_new_tokens=144,
            min_new_tokens=144,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            forced_eos_token_id=None,
            suppress_tokens=list(range(15)),
        )[:, 1:]
