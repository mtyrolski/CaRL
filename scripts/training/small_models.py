"""Small board and latent models with the bounded comparison training loop."""
import json
import time

import torch
from torch import nn
from torch.nn import functional as F


class SmallBoardGenerator(nn.Module):
    """Seven 3x3/64-channel layers, then a per-cell classifier; no edit decoder."""

    def __init__(self):
        super().__init__()
        layers = []
        for i in range(7):
            layers += [nn.Conv2d(15 if i == 0 else 64, 64, 3, padding=1), nn.ReLU()]
        self.network = nn.Sequential(*layers, nn.Conv2d(64, 7, 1))

    def forward(self, states, goals, scale):
        channels = torch.cat(
            [
                F.one_hot(states, 7).permute(0, 3, 1, 2),
                F.one_hot(goals, 7).permute(0, 3, 1, 2),
                scale[:, None, None, None].expand(-1, 1, 12, 12),
            ],
            1,
        ).float()
        return self.network(channels)


class SmallLatentGenerator(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(width + 1, 256), nn.GELU(), nn.Linear(256, width))

    def forward(self, z, scale):
        return self.network(torch.cat([z, scale[:, None]], -1))


def fit(args, dataset, architecture, condition):
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = dataset["rows"]
    tensors = {name: torch.as_tensor(dataset[name]).cuda() for name in ["starts", "targets", "goals", "z", "h"]}
    scale = torch.tensor(
        [
            r["u"] if condition == "relative" else r["mean_hit_time"] / args.cap if condition == "absolute" else 0
            for r in rows
        ],
        device="cuda",
        dtype=torch.float32,
    )
    masks = {
        split: torch.tensor([r["split"] == split for r in rows], device="cuda") for split in ["train", "val", "test"]
    }
    weights = torch.tensor([r["weight"] for r in rows], device="cuda")
    train_indices = masks["train"].nonzero().flatten()
    probabilities = weights[train_indices].cpu()
    sample_rng = torch.Generator().manual_seed(args.seed)
    net = (SmallBoardGenerator() if architecture == "cnn" else SmallLatentGenerator(tensors["z"].shape[-1])).cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    tag = f"{architecture}_{condition}"
    history = []

    def predictions(indices, mode=None, model=net, data=tensors):
        values = scale[indices] if mode is None else torch.full_like(scale[indices], mode)
        if architecture == "cnn":
            return model(data["starts"][indices], data["goals"][indices], values)
        return model(data["z"][indices], values)

    def row_losses(pred, indices, data=tensors):
        if architecture == "cnn":
            return F.cross_entropy(pred, data["targets"][indices], reduction="none").mean((1, 2))
        return (pred - data["h"][indices]).square().mean(-1)

    @torch.inference_mode()
    def metrics(split, model=net, data=tensors, predict=predictions, loss_fn=row_losses):
        model.eval()
        indices = masks[split].nonzero().flatten()
        weight = weights[indices]
        losses, accuracies, changed_correct, changed_total, exact = [], [], [], [], []
        for part in indices.split(args.batch):
            pred = predict(part)
            losses.append(loss_fn(pred, part))
            if architecture == "cnn":
                match = pred.argmax(1) == data["targets"][part]
                changed = data["starts"][part] != data["targets"][part]
                accuracies.append(match.float().mean((1, 2)))
                changed_correct.append((match & changed).sum((1, 2)))
                changed_total.append(changed.sum((1, 2)))
                exact.append(match.flatten(1).all(1).float())
        result = dict(loss=float((torch.cat(losses) * weight).sum() / weight.sum()))
        if architecture == "cnn":
            result.update(
                token_accuracy=float((torch.cat(accuracies) * weight).sum() / weight.sum()),
                changed_cell_accuracy=float(
                    (torch.cat(changed_correct) * weight).sum() / (torch.cat(changed_total) * weight).sum()
                ),
                exact_board_accuracy=float((torch.cat(exact) * weight).sum() / weight.sum()),
                copy_token_accuracy=float(
                    ((data["starts"][indices] == data["targets"][indices]).float().mean((1, 2)) * weight).sum()
                    / weight.sum()
                ),
            )
        else:
            result["copy_latent_mse"] = float(
                ((data["z"][indices] - data["h"][indices]).square().mean(-1) * weight).sum() / weight.sum()
            )
        return result

    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            values = {"run": tag, "step": step}
            for split in ["train", "val", "test"]:
                values.update({f"{split}_{key}": value for key, value in metrics(split).items()})
            history.append(values)
            print(json.dumps(values), flush=True)
        if step == args.steps:
            break
        net.train()
        chosen = torch.multinomial(probabilities, args.batch, replacement=True, generator=sample_rng).cuda()
        indices = train_indices[chosen]
        loss = row_losses(predictions(indices), indices).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
    # Diagnostic only: whether the network responds to the scale on held-out contexts.
    with torch.inference_mode():
        net.eval()
        idx = masks["test"].nonzero().flatten()
        low, high = [], []
        for part in idx.split(args.batch):
            low.append(predictions(part, 0.25).cpu())
            high.append(predictions(part, 0.75).cpu())
        low, high = torch.cat(low), torch.cat(high)
        response = (
            float((low.argmax(1) != high.argmax(1)).float().mean())
            if architecture == "cnn"
            else float((low - high).square().mean())
        )
    folder = args.output / tag
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), folder / "generator.pt")
    (folder / "trainer_state.json").write_text(json.dumps({"log_history": history}, indent=2) + "\n")
    summary = dict(
        run=tag,
        parameters=sum(p.numel() for p in net.parameters()),
        seconds=time.perf_counter() - started,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
        scale_response=response,
        initial=history[0],
        final=history[-1],
    )
    del predictions, row_losses, metrics
    del optimizer, net, tensors
    torch.cuda.empty_cache()
    return summary, history
