"""Bounded, local feasibility tests for policy-relative subgoal conditioning.

The CNN is an explicit-state ablation, not the proposal's latent planner. The
MLP tests its weighted latent MSE using cached features from the frozen CLLP;
those features do not make the existing board-input controller latent-compatible.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertForSequenceClassification

from carl.environment.sokoban.env import SokobanEnv
from carl.environment.sokoban.tokenizer import SokobanTokenizer
from carl.environment.training_goal import TrainingGoal


def weighted_midpoints(times, successes):
    """Success-weighted empirical CDF midpoints, sharing an interval at ties."""
    times, successes = np.asarray(times, float), np.asarray(successes, float)
    if len(times) == 0 or np.any(~np.isfinite(times)) or np.any(successes <= 0):
        raise ValueError('Only finite times and positive-hit targets are labelable')
    weights = successes / successes.sum()
    u = np.empty_like(times)
    lower = 0.0
    for value in np.unique(times):
        tied = times == value
        mass = weights[tied].sum()
        u[tied] = lower + mass / 2
        lower += mass
    return weights, u


def level_key(state):
    tiles = state.argmax(-1)
    return hashlib.sha256(np.stack([tiles == 0, np.isin(tiles, [2, 3, 6])]).tobytes()).hexdigest()


def valid_board(prediction, source):
    return bool(np.array_equal(prediction == 0, source == 0)
                and np.array_equal(np.isin(prediction, [2, 3, 6]), np.isin(source, [2, 3, 6]))
                and np.isin(prediction, [3, 4]).sum() == 4
                and np.isin(prediction, [5, 6]).sum() == 1)


def pair_tokens(tokenizer, states, goals):
    return torch.cat([tokenizer.x_y_tokenizer((s, g), 0, TrainingGoal.CLLP)[0]
                      for s, g in zip(states, goals)])


@torch.inference_mode()
def rollout_trials(policy, tokenizer, env, starts, targets, *, repeats, cap, seed, batch_size):
    """Sample unmasked categorical actions at T=1; exact hits, terminal stop, hard cap.

    Every trial restarts at its supplied state. Blocked/no-op actions count.
    The categorical policy differs deliberately from CaRL's greedy BasicValidator.
    """
    rng = torch.Generator(device='cuda').manual_seed(seed)
    trial_hits = np.zeros((len(starts), repeats), dtype=np.int16)
    attempted = 0
    terminal_misses = 0
    jobs = [(i, r) for i in range(len(starts)) for r in range(repeats)]
    for begin in range(0, len(jobs), batch_size):
        batch = jobs[begin:begin + batch_size]
        states = [starts[i].copy() for i, _ in batch]
        active = list(range(len(batch)))
        for step in range(1, cap + 1):
            if not active:
                break
            tokens = pair_tokens(tokenizer, [states[j] for j in active],
                                 [targets[batch[j][0]] for j in active]).cuda()
            probabilities = policy(tokens).logits.softmax(-1)
            actions = torch.multinomial(probabilities, 1, generator=rng).flatten().cpu().tolist()
            following = []
            for j, action in zip(active, actions):
                i, repeat = batch[j]
                state = env.next_state(states[j], action)
                states[j] = state
                attempted += 1
                if np.array_equal(state, targets[i]):
                    trial_hits[i, repeat] = step
                elif env.done:
                    terminal_misses += 1
                else:
                    following.append(j)
            active = following
        if begin % (batch_size * 12) == 0:
            print(f'calibration trials {min(begin + batch_size, len(jobs))}/{len(jobs)}', flush=True)
    return trial_hits, dict(attempted_actions=attempted, terminal_without_target=terminal_misses)


@torch.inference_mode()
def encode_pairs(policy, tokenizer, starts, goals, batch_size):
    parts = []
    for begin in range(0, len(starts), batch_size):
        tokens = pair_tokens(tokenizer, starts[begin:begin+batch_size], goals[begin:begin+batch_size]).cuda()
        # Mean contextualised current-state tokens; final goal is the second board.
        hidden = policy.bert(tokens).last_hidden_state[:, 1:145].mean(1)
        parts.append(hidden.cpu())
    return torch.cat(parts)


def prepare(args, policy, tokenizer, env):
    raw = joblib.load(args.data)
    rng = np.random.default_rng(args.seed)
    keys = list(raw)
    rng.shuffle(keys)
    trajectories, seen = [], set()
    replayed = 0
    for key in keys:
        trajectory = np.asarray(raw[key])
        if len(trajectory) < args.window + 3 or level_key(trajectory[0]) in seen:
            continue
        solved = [j for j, state in enumerate(trajectory) if env.is_solved(state)]
        if not solved:
            continue
        trajectory = trajectory[:solved[0]+1]
        if len(trajectory) < args.window + 3:
            continue
        assert np.all(trajectory.sum(-1) == 1)
        for before, after in zip(trajectory[:-1], trajectory[1:]):
            if np.array_equal(before, after):
                assert any(np.array_equal(env.next_state(before, a), after) for a in range(4))
            else:
                action = env.detect_action(before, after)
                assert np.array_equal(env.next_state(before, action), after), (key, replayed)
            replayed += 1
        seen.add(level_key(trajectory[0]))
        trajectories.append((str(key), trajectory))
        if len(trajectories) == args.boards:
            break
    assert len(trajectories) == args.boards
    holdout = max(2, args.boards // 6)
    train_count = args.boards - 2*holdout
    assert train_count > 0
    contexts, starts, targets, goals, records = [], [], [], [], []
    for board_index, (key, trajectory) in enumerate(trajectories):
        split = 'train' if board_index < train_count else ('val' if board_index < train_count+holdout else 'test')
        positions = rng.choice(len(trajectory)-args.window, args.contexts, replace=False)
        for position in sorted(positions):
            current = trajectory[position]
            unique = {}
            for offset in range(1, args.window+1):
                future = trajectory[position+offset]
                if not np.array_equal(current, future):
                    unique.setdefault(future.tobytes(), offset)
            selected = sorted(rng.choice(list(unique.values()), min(args.candidates, len(unique)), replace=False))
            context = len(contexts)
            contexts.append(dict(context=context, trajectory=key, level=level_key(current),
                                 position=int(position), split=split))
            for offset in selected:
                starts.append(current)
                targets.append(trajectory[position+offset])
                goals.append(trajectory[-1])
                records.append(dict(context=context, trajectory=key, split=split, offset=int(offset)))
    before = time.perf_counter()
    hits, rollout_info = rollout_trials(policy, tokenizer, env, starts, targets,
        repeats=args.repeats, cap=args.cap, seed=args.seed, batch_size=args.policy_batch)
    elapsed = time.perf_counter()-before
    keep = hits.astype(bool).any(1)
    for j, record in enumerate(records):
        successes = hits[j][hits[j] > 0]
        record.update(hits=int(len(successes)), success_rate=float(len(successes)/args.repeats),
                      mean_hit_time=float(successes.mean()) if len(successes) else None,
                      trial_hit_times=hits[j].tolist())
    kept_records = [record for j, record in enumerate(records) if keep[j]]
    for context in contexts:
        rows = [r for r in kept_records if r['context'] == context['context']]
        if not rows:
            continue
        weights, u = weighted_midpoints([r['mean_hit_time'] for r in rows], [r['hits'] for r in rows])
        for row, weight, scale in zip(rows, weights, u):
            row.update(weight=float(weight), u=float(scale))
    retained = np.flatnonzero(keep)
    starts, targets, goals = [[arr[j] for j in retained] for arr in (starts, targets, goals)]
    z = encode_pairs(policy, tokenizer, starts, goals, args.policy_batch)
    h = encode_pairs(policy, tokenizer, targets, goals, args.policy_batch)
    retained_contexts = {r['context'] for r in kept_records}
    metadata = dict(contexts=contexts, records=records,
        loaded_trajectories_in_shard=len(raw), selected_boards=args.boards,
        replayed_transitions=replayed, candidate_pairs=len(records), retained_pairs=len(kept_records),
        excluded_pairs=int((~keep).sum()), excluded_contexts=len(contexts)-len(retained_contexts),
        total_trials=int(hits.size), exact_hit_rate=float((hits > 0).mean()),
        calibration_seconds=elapsed, **rollout_info)
    dataset = dict(starts=np.stack(starts).argmax(-1), targets=np.stack(targets).argmax(-1),
                   goals=np.stack(goals).argmax(-1), z=z, h=h, rows=kept_records)
    return dataset, metadata


class SmallBoardGenerator(nn.Module):
    """Seven 3x3/64-channel layers, then a per-cell classifier; no edit decoder."""
    def __init__(self):
        super().__init__()
        layers = []
        for i in range(7):
            layers += [nn.Conv2d(15 if i == 0 else 64, 64, 3, padding=1), nn.ReLU()]
        self.network = nn.Sequential(*layers, nn.Conv2d(64, 7, 1))

    def forward(self, states, goals, scale):
        channels = torch.cat([F.one_hot(states, 7).permute(0, 3, 1, 2),
                              F.one_hot(goals, 7).permute(0, 3, 1, 2),
                              scale[:, None, None, None].expand(-1, 1, 12, 12)], 1).float()
        return self.network(channels)


class SmallLatentGenerator(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(width+1, 256), nn.GELU(), nn.Linear(256, width))

    def forward(self, z, scale):
        return self.network(torch.cat([z, scale[:, None]], -1))


def fit(args, dataset, architecture, condition):
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = dataset['rows']
    tensors = {name: torch.as_tensor(dataset[name]).cuda() for name in ['starts', 'targets', 'goals', 'z', 'h']}
    scale = torch.tensor([r['u'] if condition == 'relative' else r['mean_hit_time']/args.cap
                          if condition == 'absolute' else 0 for r in rows], device='cuda', dtype=torch.float32)
    masks = {split: torch.tensor([r['split'] == split for r in rows], device='cuda') for split in ['train','val','test']}
    weights = torch.tensor([r['weight'] for r in rows], device='cuda')
    train_indices = masks['train'].nonzero().flatten()
    probabilities = weights[train_indices].cpu()
    sample_rng = torch.Generator().manual_seed(args.seed)
    net = (SmallBoardGenerator() if architecture == 'cnn' else SmallLatentGenerator(tensors['z'].shape[-1])).cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    tag = f'{architecture}_{condition}'
    history = []

    def predictions(indices, mode=None):
        values = scale[indices] if mode is None else torch.full_like(scale[indices], mode)
        if architecture == 'cnn':
            return net(tensors['starts'][indices], tensors['goals'][indices], values)
        return net(tensors['z'][indices], values)

    def row_losses(pred, indices):
        if architecture == 'cnn':
            return F.cross_entropy(pred, tensors['targets'][indices], reduction='none').mean((1,2))
        return (pred-tensors['h'][indices]).square().mean(-1)

    @torch.inference_mode()
    def metrics(split):
        net.eval()
        indices = masks[split].nonzero().flatten()
        weight = weights[indices]
        losses, accuracies, changed_correct, changed_total, exact = [], [], [], [], []
        for part in indices.split(args.batch):
            pred = predictions(part)
            losses.append(row_losses(pred, part))
            if architecture == 'cnn':
                match = pred.argmax(1) == tensors['targets'][part]
                changed = tensors['starts'][part] != tensors['targets'][part]
                accuracies.append(match.float().mean((1,2)))
                changed_correct.append((match & changed).sum((1,2)))
                changed_total.append(changed.sum((1,2)))
                exact.append(match.flatten(1).all(1).float())
        result = dict(loss=float((torch.cat(losses)*weight).sum()/weight.sum()))
        if architecture == 'cnn':
            result.update(token_accuracy=float((torch.cat(accuracies)*weight).sum()/weight.sum()),
                changed_cell_accuracy=float((torch.cat(changed_correct)*weight).sum()/(torch.cat(changed_total)*weight).sum()),
                exact_board_accuracy=float((torch.cat(exact)*weight).sum()/weight.sum()),
                copy_token_accuracy=float(((tensors['starts'][indices] == tensors['targets'][indices]).float().mean((1,2))*weight).sum()/weight.sum()))
        else:
            result['copy_latent_mse'] = float(((tensors['z'][indices]-tensors['h'][indices]).square().mean(-1)*weight).sum()/weight.sum())
        return result

    for step in range(args.steps+1):
        if step % args.eval_every == 0 or step == args.steps:
            values = {'run':tag, 'step':step}
            for split in ['train','val','test']:
                values.update({f'{split}_{key}': value for key, value in metrics(split).items()})
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
        idx = masks['test'].nonzero().flatten()
        low, high = [], []
        for part in idx.split(args.batch):
            low.append(predictions(part, .25).cpu())
            high.append(predictions(part, .75).cpu())
        low, high = torch.cat(low), torch.cat(high)
        response = float((low.argmax(1) != high.argmax(1)).float().mean()) if architecture == 'cnn' else float((low-high).square().mean())
    folder = args.output/tag
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), folder/'generator.pt')
    (folder/'trainer_state.json').write_text(json.dumps({'log_history':history},indent=2)+'\n')
    summary = dict(run=tag, parameters=sum(p.numel() for p in net.parameters()),
        seconds=time.perf_counter()-started, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20, scale_response=response,
        initial=history[0], final=history[-1])
    del optimizer, net, tensors
    torch.cuda.empty_cache()
    return summary, history


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=Path('rl-data/validation/sokoban/offline/12-12-4/sokoban_12_12_4_trajectories_part_1.pkl'))
    parser.add_argument('--policy',type=Path,default=Path('rl-data/validation/sokoban/components/full_data/cllp/8/checkpoint-167585'))
    for name,default in [('boards',48),('contexts',2),('window',16),('candidates',8),('repeats',4),('cap',20),('policy-batch',32),('batch',32),('steps',400),('eval-every',100),('seed',20261029)]:
        parser.add_argument('--'+name,type=int,default=default)
    parser.add_argument('--lr',type=float,default=3e-4)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'summary.json').exists():
        raise FileExistsError('Choose a fresh output directory; completed experiments are immutable')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    started=time.perf_counter()
    tokenizer=SokobanTokenizer()
    env=SokobanEnv(tokenizer)
    policy=BertForSequenceClassification.from_pretrained(args.policy,local_files_only=True).eval()
    cast(nn.Module, policy).cuda()
    policy.requires_grad_(False)
    dataset,calibration=prepare(args,policy,tokenizer,env)
    joblib.dump(dataset,args.output/'calibrated_dataset.joblib')
    (args.output/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    print('CALIBRATION',json.dumps({k:v for k,v in calibration.items() if k not in ['contexts','records']}),flush=True)
    # Keep the controller out of GPU memory while optimising the small generators.
    policy_parameters=sum(p.numel() for p in policy.parameters())
    del policy
    torch.cuda.empty_cache()
    summaries,history=[],[]
    for architecture in ['cnn','latent_mlp']:
        for condition in ['relative','absolute','none']:
            summary,logs=fit(args,dataset,architecture,condition)
            summaries.append(summary)
            history.extend(logs)
    config={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    provenance=dict(config=config,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        policy_parameters=policy_parameters,data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checkpoint_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.policy.iterdir()
                          if p.name in ['config.json','model.safetensors','pytorch_model.bin']})
    report=dict(provenance=provenance,calibration={k:v for k,v in calibration.items() if k not in ['contexts','records']},
        runs=summaries,total_seconds=time.perf_counter()-started)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    write_csv(args.output/'learning_curves.csv',history)
    write_csv(args.output/'metrics.csv',[dict(parameters=r['parameters'],seconds=r['seconds'],
        peak_allocated_mib=r['peak_allocated_mib'],initial_train_loss=r['initial']['train_loss'],
        final_train_loss=r['final']['train_loss'],initial_val_loss=r['initial']['val_loss'],
        final_val_loss=r['final']['val_loss'],**r['final']) for r in summaries])
    print('COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':
    main()
