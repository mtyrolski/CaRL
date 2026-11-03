"""Post-hoc audit only: saved semantic judgments and physical action replay.

Never imported by inference/calibration and never changes recorded labels.
Unknown language is explicitly not adjudicated, rather than treated as false.
"""
import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

from carl.icl.boards import from_rows, to_rows
from carl.icl.datasets import environment
from scripts.generator_probe.prepare_sources import sha256
from scripts.goal_probe.data import OUTPUT, goals

SIGNATURE = '3907728c160d892bf5763a7054b9729c05244562491854905d5b984ff6a2091a'
RUN = OUTPUT / 'verbal' / SIGNATURE
GROUNDED_RUN = OUTPUT / 'verbal_grounded_retry' / '7bfd933812e61e587cf83225d94519513c2de4cc1ce16c9f761e837f7bc709d4'
COORD = r'\[(\d+),\s*(\d+)\]'


class Judgment(TypedDict):
    """The model's saved claim, preserved without correction."""
    reached: bool
    progress: float
    reason: str


class PhysicalStep(TypedDict):
    """Saved physical transition independent of semantic judgment."""
    before: list[str]
    action: int
    after: list[str]
    solved: bool


class Step(PhysicalStep):
    """Transition and its subsequent model judgment."""
    judgment: Judgment


class Attempt(TypedDict):
    """Common calibration/query attempt fields."""
    goal: str
    actions: list[int]
    steps: list[Step]


@dataclass(frozen=True)
class Predicate:
    """Exact interpretation of the narrowly recognized coordinate instruction."""
    kind: str
    targets: tuple[tuple[int, int], ...]
    original_box: tuple[int, int] | None = None


def interpret(goal: str) -> Predicate | None:
    """Recognize only exact observed unambiguous instruction templates."""
    coordinates = tuple((int(r), int(c)) for r, c in re.findall(COORD, goal))
    if re.fullmatch(r'Move the player to cell ' + COORD + r'\.', goal):
        return Predicate('player', coordinates)
    if re.fullmatch(r'Place boxes on these cells: ' + COORD + r'(?:, ' + COORD + r')*\.', goal):
        return Predicate('box', coordinates)
    if re.fullmatch(r'Place a box on each of these goal squares: ' + COORD + r'(?:, ' + COORD + r')*\.', goal):
        return Predicate('goal', coordinates)
    if re.fullmatch(r'Place a box on goal ' + COORD + r'\.', goal):
        return Predicate('goal', coordinates)
    if re.fullmatch(r'(?:Push the box at ' + COORD + r' onto|Place the box at ' + COORD + r' on) the goal at ' + COORD + r'\.', goal):
        return Predicate('specific_box_goal', (coordinates[-1],), coordinates[0])
    if re.fullmatch(r'(?:Push|Move) the box at ' + COORD + r' onto goal(?: square)? ' + COORD + r'\.', goal):
        return Predicate('specific_box_goal', (coordinates[-1],), coordinates[0])
    return None


def cell(rows: list[str], coordinate: tuple[int, int]) -> str:
    """Return a literal tile, explicitly handling coordinates outside the board."""
    r, c = coordinate
    return rows[r][c] if 0 <= r < len(rows) and 0 <= c < len(rows[r]) else 'OUTSIDE'


def boxes(rows: list[str]) -> set[tuple[int, int]]:
    """Find literal box coordinates for tracking the specifically named box."""
    return {(r, c) for r, row in enumerate(rows) for c, symbol in enumerate(row) if symbol in '$*'}


def raw_judgments(run_dir: Path = RUN) -> dict[str, dict[str, object]]:
    """Index actual response JSON, distinguishing absent scores from decoder defaults."""
    indexed: dict[str, dict[str, object]] = {}
    for line in (run_dir / 'raw_calls.jsonl').read_text().splitlines():
        record = json.loads(line)
        if record.get('role') != 'judge' or 'response' not in record:
            continue
        query = json.loads(record['payload']['input'][-1]['content'][0]['text'])
        response = record['response']
        claim = json.loads(response['output_text'])
        key = json.dumps([query['source'], query['state'], query['subgoal'], claim['reason'], claim['reached']])
        indexed[key] = {'progress_observed': 'progress' in claim, 'model_progress': claim.get('progress'),
                        'request_hash': response['request_hash'], 'raw_output_text': response['output_text']}
    return indexed


def audit_attempt(attempt: Attempt, raw: dict[str, dict[str, object]]) -> dict[str, object]:
    """Compare every claim, with physical replay recorded separately."""
    predicate = interpret(attempt['goal'])
    source = attempt['steps'][0]['before']
    tracked = predicate.original_box if predicate else None
    initial_box_valid = tracked is None or cell(source, tracked) in '$*'
    reviews: list[dict[str, object]] = []
    for number, step in enumerate(attempt['steps'], 1):
        current = step['after']
        if tracked is not None:
            removed = boxes(step['before']) - boxes(current)
            added = boxes(current) - boxes(step['before'])
            if tracked in removed and len(added) == 1:
                tracked = next(iter(added))
        truth: bool | None = None
        evidence: list[dict[str, object]] = []
        if predicate:
            allowed = '@+' if predicate.kind == 'player' else '$*' if predicate.kind == 'box' else '*'
            truth = all(cell(current, rc) in allowed for rc in predicate.targets)
            if predicate.kind == 'specific_box_goal':
                truth = truth and initial_box_valid and tracked == predicate.targets[0]
            evidence = [{'coordinate': list(rc), 'source_tile': cell(source, rc), 'current_tile': cell(current, rc)}
                        for rc in predicate.targets]
        claim = step['judgment']
        key = json.dumps([source, current, attempt['goal'], claim['reason'], claim['reached']])
        raw_claim = raw[key]
        assessment = ('not_adjudicated' if truth is None else 'agrees' if truth == claim['reached']
                      else 'false_positive' if claim['reached'] else 'false_negative')
        reviews.append({'step': number, 'action': step['action'], 'claim': claim,
                        'literal_predicate': truth, 'assessment': assessment, 'coordinate_evidence': evidence,
                        'tracked_named_box': list(tracked) if tracked is not None else None,
                        'before': step['before'], 'after': current, 'raw_response_evidence': raw_claim,
                        'progress_decoder_placeholder': not raw_claim['progress_observed']})
    env = environment(from_rows(source))
    replay_ok = True
    previous = source
    for action, step in zip(attempt['actions'], attempt['steps'], strict=True):
        replayed = env.next_state(from_rows(step['before']), action)
        replay_ok &= (previous == step['before'] and action == step['action']
                      and to_rows(replayed) == step['after'] and env.is_solved(replayed) == step['solved'])
        previous = step['after']
    return {'goal': attempt['goal'], 'actions': attempt['actions'], 'source': source,
            'occupied_goals_before': goals(source), 'occupied_goals_after': goals(previous),
            'first_literal_hit': next((row['step'] for row in reviews if row['literal_predicate'] is True), None),
            'first_judged_hit': next((number for number, step in enumerate(attempt['steps'], 1)
                                     if step['judgment']['reached']), None),
            'interpretation': predicate.kind if predicate else 'ambiguous_or_unrecognized_not_adjudicated',
            'original_box_reference_valid': initial_box_valid,
            'specified_goal_squares_are_goals': (all(cell(source, rc) in '.*+' for rc in predicate.targets)
                                                if predicate and predicate.kind in {'goal', 'specific_box_goal'} else None),
            'action_replay_verified': replay_ok, 'steps': reviews}


def run(run_dir: Path = RUN, grounded: bool = False) -> None:
    """Audit completed records only, preserving originals and their exact reasons."""
    paths = [] if grounded else sorted((run_dir / 'calibration' / run_dir.name).glob('context_*.json'))
    paths += sorted(run_dir.glob('query_[0-9][0-9].json'))
    if len(paths) != (2 if grounded else 10):
        raise ValueError('Completed-record count differs from the explicitly selected audit scope')
    records: list[dict[str, object]] = []
    lines = ['# Post-hoc semantic judge audit', '',
             'Offline diagnostic only. No private validation witnesses, model calls, label changes, or runtime rejection. '
             'Coordinate predicates audit reached claims; reasons are retained verbatim, not universally verified. '
             'Unknown language is not adjudicated. The validation cohort was selected near goal placements.', '']
    if grounded:
        lines += ['This is a separate post-hoc retry of two previously observed failures with a changed generator. '
                  'It is not an independent four-query retest and is not pooled with the original 46 judgments.', '']
    totals = dict.fromkeys(['agrees', 'false_positive', 'false_negative', 'not_adjudicated', 'progress_decoder_placeholder'], 0)
    raw = raw_judgments(run_dir)
    physical_actions = 0
    for path in paths:
        document = json.loads(path.read_text())
        attempts = cast(list[Attempt], document.get('attempts', [document]))
        audited = [audit_attempt(attempt, raw) for attempt in attempts]
        physical = [cast(PhysicalStep, json.loads(line)) for line in path.with_suffix('.trace.jsonl').read_text().splitlines()]
        recorded_steps = [step for attempt in attempts for step in attempt['steps']]
        physical_match = physical == [PhysicalStep(before=step['before'], action=step['action'],
                                                  after=step['after'], solved=step['solved'])
                                      for step in recorded_steps]
        if not physical_match or not all(item['action_replay_verified'] for item in audited):
            raise ValueError(f'Saved physical trace/replay mismatch: {path}')
        physical_actions += len(physical)
        records.append({'path': str(path.relative_to(OUTPUT)), 'sha256': sha256(path),
                        'physical_trace_sha256': sha256(path.with_suffix('.trace.jsonl')),
                        'physical_trace_matches_record': physical_match, 'attempts': audited})
        lines += [f'## {path.name}', '']
        for item in audited:
            lines += [f"Goal: **{item['goal']}**", '', f"Actions: `{item['actions']}`. Physical replay verified.", '']
            lines += [f"Occupied goals: {item['occupied_goals_before']} → {item['occupied_goals_after']}. "
                      f"First literal hit: {item['first_literal_hit']}; first judged hit: {item['first_judged_hit']}.", '']
            if item['interpretation'] == 'specific_box_goal':
                lines += [f"Original named cell contains a box: {item['original_box_reference_valid']}. "
                          f"Named destination is a goal square: {item['specified_goal_squares_are_goals']}. "
                          'The named box is tracked through any physical pushes.', '']
            reviewed_steps = cast(list[dict[str, object]], item['steps'])
            for step in reviewed_steps:
                assessment = str(step['assessment'])
                totals[assessment] += 1
                totals['progress_decoder_placeholder'] += int(bool(step['progress_decoder_placeholder']))
                claim = cast(Judgment, step['claim'])
                raw_claim = cast(dict[str, object], step['raw_response_evidence'])
                lines += [f"- Step {step['step']}: reached={claim['reached']}, model progress={raw_claim['model_progress']}; "
                          f"literal={step['literal_predicate']}; **{assessment}**. Reason: {claim['reason']}"]
            lines += ['', f"Coordinate evidence: `{reviewed_steps[-1]['coordinate_evidence']}`.", '']
    report = {'version': 1, 'scope': 'Post-hoc only; no feedback to models or calibration labels.',
              'signature': run_dir.name, 'completed_calibration_records': 0 if grounded else 6,
              'completed_validation_queries': 2 if grounded else 4,
              'selection': 'Two observed failures retried with changed generator; not independent retest.' if grounded
                           else 'Original six calibration and four targeted validation runs.',
              'physical_actions_replayed': physical_actions, 'all_physical_traces_match': True,
              'progress_note': 'Absent raw progress is unobserved (null); saved 0.0 is a decoder placeholder, not a model score.',
              'raw_calls_sha256': sha256(run_dir / 'raw_calls.jsonl'), 'claim_totals': totals, 'records': records}
    name = 'grounded_judge_audit' if grounded else 'judge_audit'
    (OUTPUT / f'{name}.json').write_text(json.dumps(report, indent=2) + '\n')
    lines[3:3] = [f'Totals: `{totals}`. Replayed physical actions: **{physical_actions}**.', '',
                  'Absent raw progress is unobserved (null); saved 0.0 is a decoder placeholder, not a model score.', '']
    (OUTPUT / f'{name}.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'physical_actions_replayed': physical_actions, 'claim_totals': totals}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grounded', action='store_true', help='Audit only the two post-hoc grounded retries')
    args = parser.parse_args()
    run(GROUNDED_RUN if args.grounded else RUN, args.grounded)
