"""Training-only label calibration and rollout helpers for the relative-policy studies."""
import hashlib
import time

import joblib
import numpy as np
import torch

from carl.environment.training_goal import TrainingGoal


def weighted_midpoints(times, successes):
    """Success-weighted empirical CDF midpoints, sharing an interval at ties."""
    times, successes = np.asarray(times, float), np.asarray(successes, float)
    if len(times) == 0 or np.any(~np.isfinite(times)) or np.any(successes <= 0):
        raise ValueError("Only finite times and positive-hit targets are labelable")
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
    return bool(
        np.array_equal(prediction == 0, source == 0)
        and np.array_equal(np.isin(prediction, [2, 3, 6]), np.isin(source, [2, 3, 6]))
        and np.isin(prediction, [3, 4]).sum() == 4
        and np.isin(prediction, [5, 6]).sum() == 1
    )


def pair_tokens(tokenizer, states, goals):
    return torch.cat([tokenizer.x_y_tokenizer((s, g), 0, TrainingGoal.CLLP)[0] for s, g in zip(states, goals)])


@torch.inference_mode()
def rollout_trials(policy, tokenizer, env, starts, targets, *, repeats, cap, seed, batch_size):
    """Sample unmasked categorical actions at T=1; exact hits, terminal stop, hard cap.

    Every trial restarts at its supplied state. Blocked/no-op actions count.
    The categorical policy differs deliberately from CaRL's greedy BasicValidator.
    """
    rng = torch.Generator(device="cuda").manual_seed(seed)
    trial_hits = np.zeros((len(starts), repeats), dtype=np.int16)
    attempted = 0
    terminal_misses = 0
    jobs = [(i, r) for i in range(len(starts)) for r in range(repeats)]
    for begin in range(0, len(jobs), batch_size):
        batch = jobs[begin : begin + batch_size]
        states = [starts[i].copy() for i, _ in batch]
        active = list(range(len(batch)))
        for step in range(1, cap + 1):
            if not active:
                break
            tokens = pair_tokens(tokenizer, [states[j] for j in active], [targets[batch[j][0]] for j in active]).cuda()
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
            print(f"calibration trials {min(begin + batch_size, len(jobs))}/{len(jobs)}", flush=True)
    return trial_hits, dict(attempted_actions=attempted, terminal_without_target=terminal_misses)


@torch.inference_mode()
def encode_pairs(policy, tokenizer, starts, goals, batch_size):
    parts = []
    for begin in range(0, len(starts), batch_size):
        tokens = pair_tokens(tokenizer, starts[begin : begin + batch_size], goals[begin : begin + batch_size]).cuda()
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
        trajectory = trajectory[: solved[0] + 1]
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
    train_count = args.boards - 2 * holdout
    assert train_count > 0
    contexts, starts, targets, goals, records = [], [], [], [], []
    for board_index, (key, trajectory) in enumerate(trajectories):
        split = "train" if board_index < train_count else ("val" if board_index < train_count + holdout else "test")
        positions = rng.choice(len(trajectory) - args.window, args.contexts, replace=False)
        for position in sorted(positions):
            current = trajectory[position]
            unique = {}
            for offset in range(1, args.window + 1):
                future = trajectory[position + offset]
                if not np.array_equal(current, future):
                    unique.setdefault(future.tobytes(), offset)
            selected = sorted(rng.choice(list(unique.values()), min(args.candidates, len(unique)), replace=False))
            context = len(contexts)
            contexts.append(
                dict(context=context, trajectory=key, level=level_key(current), position=int(position), split=split)
            )
            for offset in selected:
                starts.append(current)
                targets.append(trajectory[position + offset])
                goals.append(trajectory[-1])
                records.append(dict(context=context, trajectory=key, split=split, offset=int(offset)))
    before = time.perf_counter()
    hits, rollout_info = rollout_trials(
        policy,
        tokenizer,
        env,
        starts,
        targets,
        repeats=args.repeats,
        cap=args.cap,
        seed=args.seed,
        batch_size=args.policy_batch,
    )
    elapsed = time.perf_counter() - before
    keep = hits.astype(bool).any(1)
    for j, record in enumerate(records):
        successes = hits[j][hits[j] > 0]
        record.update(
            hits=int(len(successes)),
            success_rate=float(len(successes) / args.repeats),
            mean_hit_time=float(successes.mean()) if len(successes) else None,
            trial_hit_times=hits[j].tolist(),
        )
    kept_records = [record for j, record in enumerate(records) if keep[j]]
    for context in contexts:
        rows = [r for r in kept_records if r["context"] == context["context"]]
        if not rows:
            continue
        weights, u = weighted_midpoints([r["mean_hit_time"] for r in rows], [r["hits"] for r in rows])
        for row, weight, scale in zip(rows, weights, u):
            row.update(weight=float(weight), u=float(scale))
    retained = np.flatnonzero(keep)
    starts, targets, goals = [[arr[j] for j in retained] for arr in (starts, targets, goals)]
    z = encode_pairs(policy, tokenizer, starts, goals, args.policy_batch)
    h = encode_pairs(policy, tokenizer, targets, goals, args.policy_batch)
    retained_contexts = {r["context"] for r in kept_records}
    metadata = dict(
        contexts=contexts,
        records=records,
        loaded_trajectories_in_shard=len(raw),
        selected_boards=args.boards,
        replayed_transitions=replayed,
        candidate_pairs=len(records),
        retained_pairs=len(kept_records),
        excluded_pairs=int((~keep).sum()),
        excluded_contexts=len(contexts) - len(retained_contexts),
        total_trials=int(hits.size),
        exact_hit_rate=float((hits > 0).mean()),
        calibration_seconds=elapsed,
        **rollout_info,
    )
    dataset = dict(
        starts=np.stack(starts).argmax(-1),
        targets=np.stack(targets).argmax(-1),
        goals=np.stack(goals).argmax(-1),
        z=z,
        h=h,
        rows=kept_records,
    )
    return dataset, metadata
