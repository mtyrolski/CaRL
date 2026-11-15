#!/usr/bin/env python3
"""Build and execute a compact offline notebook from saved follow-up artifacts.

This only reads local data and accounting files. It never invokes a model,
opens a network connection or resumes an experiment.
"""

import argparse
import os
import textwrap
from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "analysis/luna_policy_relative_resume_20261115T0416Z"


def markdown(text: str):
    return nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbformat.v4.new_code_cell(textwrap.dedent(text).strip())


def notebook(run_dir: Path, root: Path = ROOT):
    """Create the report definition; missing and interrupted stages stay visible."""
    setup = f"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import sqlite3
import matplotlib
from IPython.display import Image, display

matplotlib.use('Agg', force=True)
root = Path({str(root.resolve())!r})
run = Path({str(run_dir.resolve())!r})
print('Stan plików odczytany:', datetime.now(timezone.utc).isoformat())
print('Katalog:', run)

def read_json(path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        print('Nie można odczytać', path, ':', type(error).__name__)
        return None

def read_jsonl(path):
    if not path.is_file():
        return []
    rows = []
    lines = path.read_bytes().splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except (UnicodeDecodeError, ValueError):
            if index == len(lines) - 1 and not line.endswith(b'\\n'):
                print('Pominięto niedomknięty ostatni wiersz:', path)
                break
            raise
    return rows

def present(path):
    return str(path) if path.is_file() else 'brak pliku'
"""
    cells = [
        markdown("""
        # Luna i generatory zależne od polityki — raport z badania

        To zapis dostępnych etapów, także przerwanych. Nieuzyskany wynik nie jest
        porażką kontrolera. Czas `d` oznacza średnią liczbę akcji w udanych
        próbach, a `u` ważony percentyl średnich czasów kandydatów w tym samym
        stanie. Offset demonstracji `k` nie jest ani `d`, ani `u`.

        Komórki czytają wyłącznie lokalne pliki; wykonanie notebooka nie uruchamia
        eksperymentu, treningu ani API. Data przy każdym odczycie oznacza
        moment wykonania notebooka, nie czas zakończenia badania.
        """),
        code(setup),
        markdown("""
        ## Plan i stan etapów

        Docelowe porównania to D–N (informacja od kontrolera), N–U (dodanie
        skali względnej) i T–U (czas bezwzględny wobec skali względnej).
        Porównanie generatorów nie jest testem pełnego rozwiązania zadania.
        Lista zamierzonych ramion nie oznacza ich wykonania.
        """),
        code("""
        manifest = read_json(run / 'run_manifest.json')
        protocol = read_json(run / 'protocol_precalibration.json')
        if manifest is None:
            print('Brak manifestu supervisora.')
        else:
            print('Status supervisora:', manifest.get('status'))
            clock = manifest.get('clock', {})
            print('Start:', clock.get('start_utc'), 'termin:', clock.get('deadline_utc'))
            proposal = manifest.get('proposal')
            if proposal:
                path = Path(proposal)
                digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
                print('Proposal:', proposal, 'SHA-256:', digest)
            for stage in manifest.get('stages', []):
                print(' ', stage.get('name'), stage.get('status'), 'exit=', stage.get('exit_code'),
                      'log=', stage.get('output'))
        if protocol is None:
            print('Brak zamrożonego protokołu.')
        else:
            for key in ('controller', 'action_cap', 'attempts_per_candidate', 'temperature', 'history',
                        'intended_arms', 'tiers', 'selection', 'selection_rule',
                        'new_and_held_reservations_ceiling_usd'):
                print(key + ':', protocol.get(key))
            print('Zamrożone bloki:', json.dumps(protocol.get('files'), indent=2, ensure_ascii=False))
        print('Końcowy status badania:', read_json(run / 'study_status.json') or 'brak osobnego pliku')
        """),
        markdown("""
        ## How to read the earlier Sokoban experiment

        The following results are from the exploratory 14 November experiment,
        not from this campaign. An **independent board** in that table means a
        distinct canonical arrangement of walls and goals: rotations and
        reflections are grouped together, while box and player positions are
        ignored for this grouping. It does not guarantee statistical
        independence or exclude every possible earlier exposure.

        The historical set had 12 training layouts and 50 protected test
        layouts. The 12 were split into 9 training and 3 validation boards;
        20 of the 50 were selected for the older test. Thus 358 training
        fragments came from only 9 boards. Its 100 test contexts were sampled
        from 20 boards, so they are not 100 independent boards.

        **Base** is Qwen3-0.6B before adaptation. **Adapted** is the same base
        with a LoRA trained on checked demonstrations; this older LoRA was not
        trained on Luna's measurements. **Local rule** is a deterministic
        current-board baseline: it finds a walkable pushing position and
        proposes a legal push, preferring one onto a goal. Its own search work
        matters; it is not another trained generator.

        A bounded breadth-first search (BFS) tries to find an action sequence
        satisfying a proposal, with at most 24 actions, 10,000 expanded nodes
        and one second per proposal. **Unresolved** means the search stopped
        without finding a sequence within those limits; it does not mean the
        intent is impossible. This is simulator search, not achievement by an
        LLM controller or proof of solving the whole puzzle.

        The primary older parser demanded strict JSON: 0/100 base outputs and
        100/100 adapted outputs parsed, and all 100 base outputs had a Markdown
        fence. Those 0/100 describe strict formatting, not the semantic validity
        of the proposed intents: the wrapper lines are not part of the JSON.
        Afterwards, an exploratory check removed at most one complete outer Markdown fence
        symmetrically for both models. The table below reports that post-hoc
        normalisation. Proposals differed in difficulty: the base usually
        requested the whole puzzle goal, whereas the adapter generally
        requested a box position. The BFS difference is therefore not a
        matched-difficulty estimate of better planning.
        """),
        code("""
        old = root / 'analysis/intent_generator_20261114T032232Z'
        old_manifest = read_json(old / 'dataset_manifest.json')
        if old_manifest is None:
            print('Older dataset manifest unavailable:', old / 'dataset_manifest.json')
        else:
            for split in ('train', 'validation', 'test'):
                item = old_manifest.get('counts', {}).get(split, {})
                print(split, 'independent boards:', item.get('boards'), 'checked fragments:', item.get('records'))
        summary = old / 'fence_feasibility/summary.csv'
        if not summary.is_file():
            print('Older post-hoc normalisation results unavailable:', summary)
        else:
            with summary.open(encoding='utf-8', newline='') as stream:
                for row in csv.DictReader(stream):
                    print(row['method'], 'valid non-trivial:', row['valid_nontrivial'] + '/' + row['outputs'],
                          'BFS sequence found:', row['witness_found'] + '/' + row['outputs'],
                          'valid but unresolved:', row['unresolved_among_valid'] + '/' + row['valid_nontrivial'])
        print('Previous strict parser, local rule and limitations:', old / 'REPORT.md')
        print('Post-hoc data and solver limits:', old / 'fence_feasibility/summary.json')
        figure = old / 'figures/posthoc_feasibility.png'
        if figure.is_file():
            print('Previous exploratory study — post-hoc fence normalisation and bounded BFS.')
            display(Image(filename=str(figure)))
        else:
            print('Previous exploratory study figure unavailable:', figure)
        """),
        markdown("""
        ## Dane i ich pochodzenie

        Plansze testowe są oddzielone według dostępnej historii ekspozycji;
        ta kontrola nie dowodzi braku jakiejkolwiek wcześniejszej ekspozycji.
        Różne źródła plansz ×4 i ×10 nie izolują liczby skrzynek.
        """),
        code("""
        data = read_json(run / 'inputs/manifest.json')
        if data is None:
            print('Brak manifestu danych.')
        else:
            for key in ('calibration', 'demonstrations', 'test', 'input_fragment_audit',
                        'excluded_semantic_duplicate_contexts', 'limitations'):
                value = data.get(key)
                if key == 'test' and isinstance(value, dict):
                    value = {k: v for k, v in value.items() if k != 'layout_ids'}
                print(key + ':', json.dumps(value, ensure_ascii=False))
            print('Kontrola hashy zamrożonych plików:')
            for name, expected in data.get('output_sha256', {}).items():
                path = run / 'inputs' / name
                actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
                print(' ', name, 'OK' if actual == expected else 'BRAK LUB NIEZGODNOŚĆ', actual)
            print('Źródła i ich SHA-256:', json.dumps(data.get('input_sha256'), ensure_ascii=False, indent=2))
        hashes = read_json(run / 'source_hashes.json')
        if hashes is not None:
            print('Hash kodu i konfiguracji:', json.dumps(hashes, ensure_ascii=False, indent=2))
        """),
        markdown("""
        ## Sprzęt i próba zasobów

        Pomiar diagnostyczny `u=0,5` sprawdzał zasoby i nie jest etykietą Luny.
        """),
        code("""
        benchmark = read_json(run / 'resource_check.json')
        if benchmark is None:
            benchmark = read_json(root / 'analysis/luna_policy_relative_followup_20261115T032143Z/resource_check.json')
        if benchmark is None:
            print('Brak zapisanego pomiaru zasobów.')
        else:
            for key in ('status', 'measurement_kind', 'device', 'model', 'revision', 'dtype', 'rank',
                        'accumulation', 'complete_step_seconds', 'training_tokens_per_second',
                        'generation_tokens_per_second', 'gpu_peak_allocated_bytes',
                        'gpu_peak_reserved_bytes', 'host_peak_rss_bytes', 'code_sha256'):
                print(key + ':', benchmark.get(key))
        hardware = read_json(run / 'hardware.json')
        if hardware is not None:
            print('Sprzęt z audytu:', json.dumps(hardware, ensure_ascii=False))
        """),
        markdown("""
        ## Rozliczenie API

        Kwoty rozliczone w ledgerze są zachowawczym oszacowaniem, nie fakturą.
        Rezerwacja `unknown` to nadal zablokowane środki, nie dowód wydatku.
        Historyczne i nowe rezerwacje pokazujemy osobno; nie są automatycznie
        anulowane ani ponawiane. Jeden zapis usage 0/0 jest niespójny z niepustą
        odpowiedzią, dlatego audyt dodaje pełną rezerwację dla tego żądania;
        rzeczywista opłata nie jest znana. Kopia ledgeru pilota jest już
        uwzględniona w ledgerze kontynuacji i nie należy jej sumować ponownie.
        """),
        code("""
        def ledger(path):
            if not path.is_file():
                print('Brak ledgeru:', path)
                return
            try:
                with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=1) as connection:
                    rows = connection.execute(
                        'SELECT status, COUNT(*), SUM(reserved_nanos), SUM(charged_nanos) '
                        'FROM attempts GROUP BY status ORDER BY status').fetchall()
                print(path)
                for status, count, reserved, charged in rows:
                    if status == 'settled':
                        print(' ', status, 'liczba=', count,
                              'oszacowanie rozliczenia USD=', None if charged is None else round(charged / 1e9, 9))
                    else:
                        print(' ', status, 'liczba=', count,
                              'nierozliczone rezerwacje USD=', None if reserved is None else round(reserved / 1e9, 9),
                              'oszacowanie rozliczenia USD=', None if charged is None else round(charged / 1e9, 9))
            except sqlite3.Error as error:
                print('Ledger chwilowo niedostępny:', path, type(error).__name__)

        carried = run / 'budget_carried/luna_intent_ledger.sqlite3'
        current = carried if carried.is_file() else run / 'budget/luna_intent_ledger.sqlite3'
        print('Bieżący ledger kampanii:')
        ledger(current)
        if carried.is_file():
            print('Wcześniejszy ledger pilota skopiowano do bieżącego; nie dodawaj go ponownie.')
        print('Osobny historyczny ledger poprzedniej kampanii:')
        ledger(root / 'analysis/luna_intent_20261114T060730Z/luna_intent_ledger.sqlite3')
        audits = sorted((current.parent / 'luna_requests').glob('*.json'))
        for path in audits:
            item = read_json(path)
            if item and item.get('status') == 'unknown':
                diagnostic = item.get('error_diagnostics') or {}
                print('Nierozstrzygnięte żądanie:', path.name,
                      'typ=', item.get('error_type'), 'HTTP=', diagnostic.get('http_status'),
                      'typ usługi=', diagnostic.get('service_type'))
        exposure = read_json(run / 'budget_exposure_audit.json')
        if exposure is None:
            print('Brak osobnego audytu ekspozycji budżetowej.')
        else:
            for label, key in (
                ('Szacunkowo rozliczone żądania obecnej kampanii', 'nominal_settled_nanos'),
                ('Nierozliczone rezerwacje obecnej kampanii', 'unknown_reservations_nanos'),
                ('Dodatkowa pełna rezerwacja dla niespójnego usage', 'supplemental_anomaly_reservations_nanos'),
                ('Łączna ekspozycja obecnej kampanii', 'current_campaign_exposure_nanos'),
                ('Osobne nierozliczone rezerwacje poprzedniej kampanii', 'previous_campaign_unknown_reservations_nanos'),
                ('Suma ekspozycji z wyłączeniami opisanymi w audycie',
                 'combined_exposure_excluding_prior_settled_and_other_projects_nanos'),
            ):
                value = exposure.get(key)
                print(label + ' USD:', None if value is None else value / 1e9)
            print('Żądania z niespójnym usage 0/0:', len(exposure.get('anomalous_zero_usage_settlements') or []),
                  '— rzeczywista opłata nieznana.')
            print('Uwagi audytu:', exposure.get('note'))
        """),
        markdown("""
        ## Kalibracja kontrolera

        Zliczamy próby behawioralne osobno od błędów infrastruktury. Powtórzenia
        dotyczą tej samej planszy i intencji, nie są dodatkowymi planszami.
        Niezakończony kontekst nie otrzymuje zmyślonego percentyla. Historyczny
        runner zapisał niektóre odpowiedzi ucięte limitem tokenów jako
        `invalid_response`. Poniżej korygujemy klasyfikację tylko wtedy, gdy
        dokładnie dopasowany surowy audyt potwierdza `max_output_tokens`.
        Zapisane pliki wyników pozostają niezmienione.

        W 50 zapisanych odpowiedziach kontynuowanej kampanii podany model to
        `gpt-6-luna`; żadna nie zawiera pola `system_fingerprint`. Sam alias
        nie dowodzi więc, że wagi usługi pozostały niezmienione. Późniejsza
        poprawka kontroli identyfikatora zmieniła hash kodu kontrolera: starych
        śladów nie należy automatycznie łączyć z pomiarami nowej wersji.
        """),
        code("""
        def attempt_audits(directory):
            by_tag = defaultdict(list)
            for path in (directory / 'luna_requests').glob('*.json'):
                audit = read_json(path)
                if audit and isinstance(audit.get('tag'), str):
                    by_tag[audit['tag']].append(audit)
            return by_tag

        def audited_status(row, audits):
            result = row.get('result') or {}
            status = result.get('status')
            steps = result.get('steps') or []
            if not steps or not isinstance(steps[-1], dict):
                return status
            prompt = steps[-1].get('prompt')
            tag = 'final:' + str(row.get('record_id')) + ':attempt:' + str(row.get('attempt'))
            matches = [item for item in audits.get(tag, [])
                       if isinstance(item.get('request'), dict) and item['request'].get('input') == prompt]
            if len(matches) != 1:
                return status
            item = matches[0]
            response = item.get('response') or {}
            details = response.get('incomplete_details') or {}
            if (status == 'invalid_response' and item.get('status') == 'settled'
                    and response.get('status') == 'incomplete' and details.get('reason') == 'max_output_tokens'):
                return 'output_limit'
            diagnostic = item.get('error_diagnostics') or {}
            message = str(diagnostic.get('service_message') or '')
            if (status == 'api_error' and item.get('status') == 'unknown'
                    and (diagnostic.get('service_code') == 'invalid_prompt' or 'invalid_prompt' in message)):
                return 'api_error_invalid_prompt'
            return status

        available = False
        for stage in ('pilot', 'pilot_continued', 'calibration', 'remainder'):
            directory = run / stage
            status = read_json(directory / 'status.json')
            records = read_jsonl(directory / 'calibrated.jsonl')
            attempts = read_jsonl(directory / 'results.jsonl')
            if status is None and not records and not attempts:
                continue
            available = True
            print('\\nETAP', stage, 'status zapisany:', status)
            budget_directory = run / ('budget' if stage == 'pilot' else 'budget_carried')
            if not budget_directory.is_dir():
                budget_directory = run / 'budget'
            audits = attempt_audits(budget_directory)
            classified = [(row, audited_status(row, audits)) for row in attempts]
            print('zapisane wyniki prób:', len(attempts), 'statusy zapisane:',
                  dict(Counter(row.get('result', {}).get('status') for row in attempts)))
            print('statusy po weryfikacji surowych audytów:', dict(Counter(item for _, item in classified)),
                  '| niewystartowane:', None if status is None else status.get('not_started'))
            corrected = [(row, item) for row, item in classified if item != row.get('result', {}).get('status')]
            if corrected:
                print('Korekta odczytu, bez zmiany surowych plików:', len(corrected),
                      '| ucięte odpowiedzi i błędy API nie są ukończonymi próbami behawioralnymi.')
            action_count = sum(len(row.get('result', {}).get('actions') or []) for row in attempts)
            model_calls = sum(row.get('result', {}).get('model_calls') or 0 for row in attempts)
            no_ops = sum(bool(step.get('no_op')) for row in attempts
                         for step in (row.get('result', {}).get('steps') or []))
            print('zapisane akcje:', action_count, 'no-op:', no_ops, 'wywołania modelu:', model_calls)
            affected_contexts = {row.get('context_id') for row, item in corrected if item == 'output_limit'}
            for row in records:
                measurement = row.get('policy_measurements') or {}
                same_record = [(attempt, item) for attempt, item in classified
                               if attempt.get('record_id') == row.get('record_id')]
                if any(item == 'output_limit' for _, item in same_record):
                    measurement['status'] = 'output_limit'
                    measurement['attempts'] = [
                        {'status': item, 'primitive_actions_attempted': len(attempt['result'].get('actions') or [])}
                        for attempt, item in same_record
                        if item in {'achieved', 'action_limit', 'task_solved', 'preservation_violated',
                                    'invalid_response'}]
                if row.get('context_id') in affected_contexts:
                    measurement.update(complete=False, success_fraction=None, mean_success_actions=None,
                                       u=None, weight=None)
            contexts = defaultdict(list)
            for row in records:
                contexts[row.get('context_id')].append(row)
            complete = [group for group in contexts.values()
                        if group and all((row.get('policy_measurements') or {}).get('complete') is True for row in group)]
            informative = [group for group in complete if len({
                (row.get('policy_measurements') or {}).get('mean_success_actions') for row in group
                if (row.get('policy_measurements') or {}).get('mean_success_actions') is not None}) > 1]
            print('konteksty:', len(contexts), 'kompletne:', len(complete),
                  'z różnymi czasami sukcesów:', len(informative))
            print('statusy rekordów:', dict(Counter((row.get('policy_measurements') or {}).get('status')
                                                     for row in records)))
            repeated = [(row.get('policy_measurements') or {}).get('attempts') or [] for row in records]
            repeated = [items for items in repeated if len(items) >= 2]
            disagreements = sum(len({item.get('status') == 'achieved' for item in items}) > 1 for items in repeated)
            different_durations = sum(len({item.get('primitive_actions_attempted') for item in items
                                           if item.get('status') == 'achieved'}) > 1 for items in repeated)
            print('powtórzone intencje:', len(repeated), 'niezgodny sukces:', disagreements,
                  'różne czasy udanych prób:', different_durations)
            print('rekord | kontekst | status | próby (status, akcje) | p | d | u | w')
            for row in records:
                measurement = row.get('policy_measurements') or {}
                repeats = [(item.get('status'), item.get('primitive_actions_attempted'))
                           for item in measurement.get('attempts') or []]
                print(row.get('record_id'), row.get('context_id'), measurement.get('status'), repeats,
                      measurement.get('success_fraction'), measurement.get('mean_success_actions'),
                      measurement.get('u'), measurement.get('weight'), sep=' | ')
        if not available:
            print('Brak zapisanych wyników kalibracji.')
        initial_plan = read_json(run / 'pilot/plan.json') or {}
        continued_plan = read_json(run / 'pilot_continued/plan.json') or {}
        initial_ids = set(initial_plan.get('record_ids') or [])
        continued_ids = set(continued_plan.get('record_ids') or [])
        if continued_ids and continued_ids <= initial_ids:
            print('Kontynuacja obejmuje', len(continued_ids), 'z', len(initial_ids),
                  'kandydatów pierwotnego pilota; nie sumuj planowanych prób obu etapów.')
        """),
        markdown("""
        ### Kontrola zapisanych akcji i warunków zachowania

        Poniższa komórka odtwarza zapisane akcje w lokalnym środowisku Sokoban
        i porównuje kolejne plansze, akcje bez zmiany oraz sprawdzenia intencji.
        To kontrola śladów, nie nowa próba kontrolera. Pusta lista `preserve`
        jest dozwolona, lecz nie pozwala zaobserwować naruszenia tego warunku.
        """),
        code("""
        import numpy as np
        from carl.icl.boards import from_rows, to_rows
        from carl.icl.data_sources.sokoban import environment
        from carl.icl.intent_data import predicate_value

        replayed = 0
        expected_actions = 0
        replay_errors = []
        for stage in ('pilot', 'pilot_continued'):
            directory = run / stage
            records = {row.get('record_id'): row for row in read_jsonl(directory / 'calibrated.jsonl')}
            for attempt in read_jsonl(directory / 'results.jsonl'):
                result = attempt.get('result') or {}
                actions = result.get('actions') or []
                expected_actions += len(actions)
                record = records.get(attempt.get('record_id'))
                if not record or not record.get('start_observation') or not isinstance(record.get('intent'), dict):
                    replay_errors.append((stage, attempt.get('record_id'), 'brak danych do odtworzenia'))
                    continue
                try:
                    state = from_rows(record['start_observation'])
                    env = environment(state)
                    intent = record['intent']
                    steps = result.get('steps') or []
                    for index, action in enumerate(actions):
                        step = steps[index]
                        before = to_rows(state)
                        following = np.asarray(env.next_state(state, action))
                        after = to_rows(following)
                        preserved = all(predicate_value(item, after) for item in intent['preserve'])
                        holds = all(predicate_value(item, after) for item in intent['effects'])
                        if (step.get('before') != before or step.get('after') != after
                                or step.get('action') != action
                                or step.get('no_op') != bool(np.array_equal(state, following))
                                or step.get('preservation_holds') != preserved
                                or step.get('intent_holds') != holds):
                            replay_errors.append((stage, attempt.get('record_id'), index))
                        state = following
                        replayed += 1
                    if result.get('endpoint') != to_rows(state):
                        replay_errors.append((stage, attempt.get('record_id'), 'endpoint'))
                except (KeyError, IndexError, TypeError, ValueError) as error:
                    replay_errors.append((stage, attempt.get('record_id'), type(error).__name__))
        print('Odtworzone akcje:', replayed, '/', expected_actions,
              '| niezgodności lub brak danych:', len(replay_errors))
        if replay_errors:
            print('Szczegóły:', replay_errors)
        candidates = read_jsonl(run / 'expanded_inputs/calibration_candidates.jsonl')
        if not candidates:
            candidates = read_jsonl(run / 'inputs/calibration_candidates.jsonl')
        if candidates:
            nonempty = sum(bool((row.get('intent') or {}).get('preserve')) for row in candidates)
            print('Kandydaci kalibracji z niepustym preserve:', nonempty, '/', len(candidates))
        else:
            print('Brak kandydatów do audytu preserve.')
        """),
        markdown("""
        ## Generatory, testy i kompletne zadania

        Wyniki pojawią się wyłącznie, jeśli istnieją odpowiednie pliki. Ten sam
        kontekst nie staje się kolejną niezależną planszą przez inne `u` ani
        powtórzenie. Weryfikacja osiągalności intencji różni się od rozwiązania
        całej planszy.
        """),
        code("""
        checkpoints = sorted((run / 'adapters').glob('*/status.json'))
        predictions = sorted((run / 'predictions').glob('*.run.json'))
        print('Trening adapterów:')
        if not checkpoints:
            print('  Brak zapisanych statusów treningu.')
        for path in checkpoints:
            print(' ', path.parent.name, read_json(path))
        curve_paths = [run / 'adapters/D_retry/training_curve.jsonl', run / 'adapters/D/training_curve.jsonl']
        curve_path = next((path for path in curve_paths if path.is_file()), None)
        if curve_path is None:
            print('Brak zapisanej krzywej treningowej D.')
        else:
            curve = read_jsonl(curve_path)
            train = [(row['step'], row['train_loss']) for row in curve
                     if isinstance(row.get('step'), int) and isinstance(row.get('train_loss'), (int, float))]
            validation = [(row['step'], row['validation_loss']) for row in curve
                          if isinstance(row.get('step'), int) and isinstance(row.get('validation_loss'), (int, float))]
            print('Krzywa D:', curve_path, '| zapisane kroki:', len(train),
                  '| walidacje:', len(validation))
            if train:
                from io import BytesIO
                import matplotlib.pyplot as plt
                buffer = BytesIO()
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot([item[0] for item in train], [item[1] for item in train], label='Training loss')
                if validation:
                    ax.scatter([item[0] for item in validation], [item[1] for item in validation],
                               label='Validation loss', marker='o', s=20)
                ax.set(xlabel='Optimizer step', ylabel='Loss', title='D: training diagnostic')
                ax.legend(frameon=False)
                fig.tight_layout()
                fig.savefig(buffer, format='png', dpi=150)
                plt.close(fig)
                display(Image(data=buffer.getvalue()))
        print('Krzywa straty opisuje dopasowanie; nie jest głównym wynikiem porównania.')
        print('Generowanie na teście:')
        if not predictions:
            print('  Brak zapisanych wyników generatorów.')
        for path in predictions:
            item = read_json(path) or {}
            print(' ', path.name, {key: item.get(key) for key in ('status', 'arm', 'requested_scale', 'error')})
        print('Brak CSV porównań D–N, N–U i T–U w ocenie lokalnej; żadnego z nich nie należy wnioskować.')
        print('Wyniki całych zadań:', present(run / 'search/summary.csv'))
        """),
        markdown("""
        ### Lokalne porównanie B, D i R na dwóch rodzinach plansz

        B to bazowy Qwen, D to ten sam model z adapterem LoRA uczonym na
        demonstracjach, a R to lokalna reguła działająca na bieżącej planszy.
        Wyniki dla realnych plansz 12×12×4 i wcześniej oglądanych, syntetycznych
        plansz 12×12×10 pokazujemy osobno. Źródło plansz zmienia się razem z
        liczbą skrzynek, więc różnica między kohortami nie izoluje wpływu ×4/×10.

        Po obejrzeniu wstępnych wyników zdecydowano o rozszerzeniu: do 24
        plansz ×4 dodano z góry osobny blok 76, a do wcześniejszych 20
        historycznych plansz ×10 — osobny blok 30. Wyniki 100/50 obejmują
        odpowiednio pierwotne 24/20 i nie są od nich niezależne. Rozszerzenie
        ma charakter eksploracyjny, a nie uprzednio zaplanowanego testu
        potwierdzającego. Dodatkowe 30 stanów ×10 to stany początkowe z
        `core_pairs`; wcześniejsze 20 to inne stany, wybrane według innej
        reguły. Otrzymane kohorty nie różnią się więc wyłącznie liczebnością.

        Dwa sloty na planszę są etykietami parowania; B i D używają zachłannego
        dekodowania bez podanej skali, więc odpowiedzi mogą się powtarzać. Nie są
        to niezależne próby ani dodatkowe plansze. Przedziały niepewności, jeśli
        dostępne, pochodzą z bootstrapu na poziomie plansz.

        Poprawność zamiaru sprawdza deterministyczny walidator. Ograniczony BFS
        szuka lokalnie sekwencji akcji, która realizuje zamiar; brak znalezionej
        sekwencji przy zadanym limicie nie dowodzi niemożliwości. Te propozycje
        nie zostały wykonane przez Lunę. Wynik BFS nie jest zatem pomiarem
        osiągalności przez Lunę ani dowodem rozwiązania całego zadania.
        """),
        markdown("""
        #### Porównywalność B i D w pierwotnych blokach

        Sprawdzamy zapisane wiadomości parami dla tego samego kontekstu i slotu,
        a także zgodność dostępnych hashy wag bazowych, konfiguracji modelu,
        `tokenizer.json` i głównego kodu oraz parametrów dekodowania. Adapter D
        jest celową różnicą między ramionami.

        Oryginalne metadane nie zawierały osobnych hashy
        `tokenizer_config.json`, `generation_config.json` ani importowanej
        funkcji `generate` w `intent_generator_experiment.py`. Dlatego zgodność
        dostępnych hashy nie dowodzi pełnej tożsamości historycznego stosu
        tokenizacji i generowania. Późniejszy hash obecnego pliku nie odtwarza
        dowodu, jaka była jego treść w chwili wykonania obu przebiegów.
        """),
        code("""
        comparable_fields = ('model_weights_sha256', 'model_config_sha256', 'tokenizer_sha256', 'code_sha256',
                             'dtype', 'decoding', 'max_length', 'max_new_tokens', 'enable_thinking', 'revision', 'seed')
        same_total = paired_total = 0
        compared_blocks = 0
        for block in ('predictions', 'predictions_second_block', 'predictions_historical_x10'):
            directory = run / block
            base_meta = read_json(directory / 'B_slot0.run.json')
            adapted_meta = read_json(directory / 'D_slot0.run.json')
            if (not base_meta or not adapted_meta or base_meta.get('status') != 'complete'
                    or adapted_meta.get('status') != 'complete'):
                print(block + ': brak pary ukończonych przebiegów; nie odczytano częściowych predykcji.')
                continue
            base_rows = read_jsonl(directory / 'B_slot0.jsonl')
            adapted_rows = read_jsonl(directory / 'D_slot0.jsonl')
            base = {row.get('record_id'): row.get('prompt') for row in base_rows}
            adapted = {row.get('record_id'): row.get('prompt') for row in adapted_rows}
            if (len(base) != len(base_rows) or len(adapted) != len(adapted_rows)
                    or set(base) != set(adapted) or None in base or None in adapted):
                print(block + ': nie można jednoznacznie sparować zapisanych kontekstów.')
                continue
            paired = set(base) & set(adapted)
            same = sum(base[identifier] == adapted[identifier] for identifier in paired)
            differences = [key for key in comparable_fields if base_meta.get(key) != adapted_meta.get(key)
                           or base_meta.get(key) is None]
            print(block + ': identyczne zapisane wiadomości B/D:', same, '/', len(paired),
                  '| rozbieżne lub brakujące pola porównania:', differences)
            same_total += same
            paired_total += len(paired)
            compared_blocks += 1
        if compared_blocks:
            print('Łącznie w ukończonych pierwotnych blokach:', same_total, '/', paired_total,
                  '| bloki:', compared_blocks, '/ 3')
        """),
        code("""
        def show_local_assessment(label, directory, expected_boards):
            # examples.md is written last by the assessment script; wait for
            # the complete output set rather than reading a live partial run.
            required = ('summary.json', 'summary.csv', 'results.jsonl', 'validity_bounded_search.png',
                        'validity_bounded_search.pdf', 'validity_bounded_search.svg', 'examples.md')
            missing = [name for name in required if not (directory / name).is_file()]
            print('\\nKohorta:', label, '| planowane plansze:', expected_boards)
            if missing:
                print('Brak ukończonej oceny lokalnej; brakujące pliki:', ', '.join(missing))
                return False
            data = read_json(directory / 'summary.json')
            if not isinstance(data, dict) or not isinstance(data.get('methods'), list):
                print('Nie można odczytać ukończonego podsumowania:', directory / 'summary.json')
                return False
            print('Lokalny BFS — limity:', data.get('solver'))
            print('Sloty:', data.get('slot_labels'))
            print('Interpretacja:', data.get('interpretation'))
            labels = {'B': 'Bazowy Qwen', 'D': 'Qwen + LoRA demonstracyjna', 'R': 'Lokalna reguła'}
            methods = data['methods']
            if {row.get('method') for row in methods} != set(labels):
                print('UWAGA: podsumowanie nie zawiera dokładnie ramion B, D i R.')
            for row in methods:
                denominator = row.get('requested_outputs')
                print('\\n', labels.get(row.get('method'), row.get('method')),
                      '| plansze:', row.get('boards'), '| zamówione odpowiedzi:', denominator)
                if row.get('boards') != expected_boards:
                    print('UWAGA: liczba plansz różni się od planowanego mianownika.')
                for label_text, key in (('Ścisły JSON', 'strict_valid_json'),
                                        ('JSON po normalizacji', 'valid_json'),
                                        ('Poprawne zamiary', 'valid_intents'),
                                        ('BFS znalazł sekwencję', 'paths_found')):
                    print(' ', label_text + ':', row.get(key), '/', denominator)
                print('  Nie znaleziono sekwencji dla poprawnych zamiarów:', row.get('no_path_found_among_valid'),
                      '/', row.get('valid_intents'), '| brak odpowiedzi infrastrukturalny:',
                      row.get('infrastructure_missing'), '| błędy BFS:', row.get('solver_infrastructure_errors'))
                print('  95% bootstrap plansz — poprawność:', row.get('valid_board_bootstrap_95'),
                      '| sekwencja BFS:', row.get('path_board_bootstrap_95'))
                print('  BFS: rozwinięte węzły:', row.get('solver_expanded_nodes'),
                      '| sprawdzone akcje:', row.get('solver_primitive_edges_examined'),
                      '| czas (s):', row.get('solver_seconds'))
                print('  Sloty ze wzrostem liczby skrzynek na celach po BFS '
                      '(paths_increasing_boxes_on_goals):', row.get('paths_increasing_boxes_on_goals'),
                      '| ze spadkiem (paths_decreasing_boxes_on_goals):',
                      row.get('paths_decreasing_boxes_on_goals'))
                if row.get('method') == 'R':
                    print('  Praca reguły R:', row.get('local_rule_work'))
            print('Parowane różnice D − B według plansz:', data.get('paired'))
            print('Dane:', directory / 'summary.csv', '| przykłady:', directory / 'examples.md')
            print('Rysunek wektorowy:', directory / 'validity_bounded_search.pdf',
                  '| SVG:', directory / 'validity_bounded_search.svg')
            display(Image(filename=str(directory / 'validity_bounded_search.png')))
            return True

        real_complete = show_local_assessment('realne 12×12×4', run / 'assessment', 24)
        historical_complete = show_local_assessment('historyczne, syntetyczne 12×12×10',
                                                    run / 'assessment_historical_x10', 20)
        expanded_real_complete = show_local_assessment('rozszerzone realne 12×12×4 (24 + 76)',
                                                       run / 'assessment_100', 100)
        expanded_historical_complete = show_local_assessment('rozszerzone historyczne 12×12×10 (20 + 30)',
                                                             run / 'assessment_historical_x10_50', 50)
        """),
        markdown("""
        ### Diagnostyka wyników na poziomie plansz

        Dwa sloty mogą odpowiadać temu samemu zamiarowi. Liczymy więc zarówno
        wyjścia, jak i unikalne plansze, na których wystąpił zamiar, został
        sprawdzony jako poprawny lub BFS znalazł sekwencję. Błędy walidacji
        mogą współwystępować w jednym wyjściu, więc ich liczby mogą się nakładać.
        Zmiana liczby skrzynek na celach dotyczy stanu osiągniętego przez lokalny
        BFS, nie oznacza ukończenia planszy ani działania Luny.
        """),
        code("""
        def show_generation_diagnostics(directory, complete):
            if not complete:
                print('Brak kompletnej oceny; diagnostyki nie odczytano:', directory)
                return
            diagnostic = read_json(directory / 'generation_diagnostics.json')
            if not isinstance(diagnostic, dict) or not isinstance(diagnostic.get('methods'), dict):
                print('Brak zapisanej diagnostyki generowania:', directory)
                return
            results = read_jsonl(directory / 'results.jsonl')
            print('\\nDiagnostyka:', directory)
            for arm in ('B', 'D', 'R'):
                item = diagnostic['methods'].get(arm) or {}
                rows = [row for row in results if row.get('arm') == arm]
                board_count = item.get('boards')
                print(arm, '| plansze:', board_count, '| wyjścia:', item.get('outputs'),
                      '| poprawny zamiar na planszach:', item.get('valid_boards'), '/', board_count,
                      '| BFS znalazł sekwencję na planszach:', item.get('path_found_boards'), '/', board_count)
                if isinstance(board_count, int) and item.get('outputs') == 2 * board_count:
                    print('  Mianownik:', item['outputs'], '=', board_count, 'plansz × 2 sloty')
                print('  Identyczne surowe odpowiedzi w obu slotach:',
                      item.get('slots_with_identical_raw_output'), '/', board_count, 'plansz')
                print('  Błędy walidacji według wyjść:', item.get('validation_errors'))
                changes_by_board = defaultdict(set)
                for row in rows:
                    change = row.get('goal_box_change')
                    if row.get('path_found') is True and isinstance(change, int) and not isinstance(change, bool):
                        changes_by_board[row.get('layout_id')].add(change)
                increased = sum(any(change > 0 for change in changes) for changes in changes_by_board.values())
                decreased = sum(any(change < 0 for change in changes) for changes in changes_by_board.values())
                distribution = Counter(change for changes in changes_by_board.values() for change in changes)
                print('  Plansze z większą liczbą skrzynek na celach po ścieżce BFS:', increased, '/', board_count,
                      '| z mniejszą:', decreased, '/', board_count)
                print('  Rozkład zmian w osiągniętych zamiarach według plansz:', dict(sorted(distribution.items())),
                      '(plansza może wystąpić w więcej niż jednej kategorii)')
                if arm == 'B':
                    output_counts = Counter()
                    boards_by_count = defaultdict(set)
                    above_box_count = 0
                    for row in rows:
                        parsed = row.get('parsed')
                        intent = parsed if isinstance(parsed, dict) else {}
                        effects = intent.get('effects')
                        effects = effects if isinstance(effects, list) else []
                        counts = {pred.get('count') for pred in effects
                                  if isinstance(pred, dict) and pred.get('predicate') == 'boxes_on_goals_at_least'
                                  and isinstance(pred.get('count'), int)}
                        output_counts.update(counts)
                        for count in counts:
                            boards_by_count[count].add(row.get('layout_id'))
                        boxes = sum(line.count('$') + line.count('*') for line in row['start_observation'])
                        above_box_count += any(count > boxes for count in counts)
                    print('  Żądane liczby skrzynek na celach — wyjścia:', dict(sorted(output_counts.items())),
                          '| plansze:', {key: len(value) for key, value in sorted(boards_by_count.items())})
                    print('  Żądanie większe niż liczba skrzynek:', above_box_count, '/', len(rows), 'wyjść')
                if arm == 'D':
                    common = item.get('most_common_parseable_intents') or []
                    if common:
                        intent = common[0].get('intent')
                        matches = [row for row in rows if row.get('parsed') == intent]
                        boards = {row.get('layout_id') for row in matches}
                        valid_boards = {row.get('layout_id') for row in matches if row.get('valid_intent')}
                        print('  Najczęściej powtarzany zamiar:', json.dumps(intent, ensure_ascii=False),
                              '| wyjścia:', len(matches), '| plansze:', len(boards),
                              '| poprawny na planszach:', len(valid_boards), '/', len(boards))
            print('Uwaga o mianowniku:', diagnostic.get('denominator_note'))

        show_generation_diagnostics(run / 'assessment', real_complete)
        show_generation_diagnostics(run / 'assessment_historical_x10', historical_complete)
        show_generation_diagnostics(run / 'assessment_100', expanded_real_complete)
        show_generation_diagnostics(run / 'assessment_historical_x10_50', expanded_historical_complete)
        """),
        markdown("""
        ### Niezależne odtworzenie zapisanych sekwencji

        Dodatkowy audyt ponownie sprawdza zapisane odpowiedzi, paruje prompty
        B/D i odtwarza znalezione sekwencje w środowisku Sokoban. Liczba
        odtworzonych ścieżek obejmuje oba sloty, więc nie jest liczbą
        niezależnych plansz. Audyt nie uruchamia modeli ani Luny.
        """),
        code("""
        for label, audit_name, assessment_name in (
            ('×4 / 100', 'independent_audit_x4_100.json', 'assessment_100'),
            ('×10 / 50', 'independent_audit_x10_50.json', 'assessment_historical_x10_50'),
        ):
            path = run / audit_name
            audit = read_json(path)
            if not isinstance(audit, dict):
                print(label + ': brak ukończonego niezależnego audytu.')
                continue
            sources = [
                (run / assessment_name / 'results.jsonl', audit.get('results_sha256')),
                (run / assessment_name / 'summary.json', audit.get('summary_sha256')),
            ]
            for item in audit.get('prediction_sources') or []:
                source = Path(item['path'])
                sources.append((source if source.is_absolute() else root / source, item.get('sha256')))
            if any(not source.is_file() or not expected
                   or hashlib.sha256(source.read_bytes()).hexdigest() != expected for source, expected in sources):
                print(label + ': źródła nie zgadzają się z hashami audytu; wyniku nie pokazano.')
                continue
            print(label, '| zgodne pary promptów B/D:', audit.get('prompt_matches'), '/', audit.get('prompt_pairs'),
                  '| odtworzone ścieżki:', audit.get('replayed_paths'), '| akcje:', audit.get('replayed_actions'))
            print('  Surowe zliczenia według metody:', json.dumps(audit.get('methods'), ensure_ascii=False))
            print('  Audyt i komenda odtworzenia:', path, '| skrypt:', run / 'independent_replay_audit.py')
        """),
        markdown("""
        ### Eksploracyjny test przypisania zamiarów do plansz

        Po obejrzeniu wyników zaplanowano dodatkowy test: całe odpowiedzi są
        losowo przypisywane do innych plansz w tej samej rodzinie, bez
        pozostawiania odpowiedzi na pierwotnej planszy. To samo przypisanie
        stosuje się do B, D i R. Zachowuje ono częstość odpowiedzi, a rozrywa
        ich pierwotny związek z planszą. Test wykorzystuje po jednym slocie na
        planszę i nie uruchamia ponownie generatora.

        Rozkład 1000 losowych przypisań opisuje wyniki tego eksperymentu
        permutacyjnego; jego kwantyle **nie są przedziałem ufności**. Lokalny
        BFS uruchomiono tylko dla pierwszego przypisania, a nie dla wszystkich
        1000. Test nie mierzy realizacji zamiarów przez Lunę ani jakości
        planowania całego zadania.
        """),
        code("""
        permutation = run / 'permutation'
        permutation_status = read_json(permutation / 'status.json')
        if not permutation_status or permutation_status.get('state') != 'complete':
            print('Brak ukończonego testu permutacyjnego:', permutation_status)
        else:
            protocol = read_json(permutation / 'protocol.json') or {}
            summary = read_json(permutation / 'summary.json') or {}
            distribution_path = permutation / 'permutation_distribution.csv'
            figures = summary.get('figures') or []
            required = [distribution_path, permutation / 'permutations.json', permutation / 'paired_cases.jsonl']
            required.extend(permutation / name for name in figures)
            if not isinstance(summary.get('methods'), list) or not figures or any(not path.is_file() for path in required):
                print('Ukończony status nie ma kompletu wyników; nie pokazano częściowych danych.')
            else:
                with distribution_path.open(encoding='utf-8', newline='') as stream:
                    distribution = list(csv.DictReader(stream))
                counts = Counter((row.get('family'), row.get('arm')) for row in distribution)
                print('Liczba losowych przypisań:', protocol.get('permutations'),
                      '| seed:', protocol.get('seed'), '| wybrany slot:', protocol.get('slot'))
                print('Wiersze rozkładu według rodziny i metody:', dict(counts))
                for row in summary['methods']:
                    requested = row.get('requested')
                    quantiles = row.get('permuted_valid_fraction_requested_quantiles') or {}
                    print('\\n', row.get('family'), row.get('arm'), '| plansze:', requested,
                          '| poprawne na oryginalnej planszy:', row.get('observed_valid'), '/', requested)
                    print('  Losowe przypisania — średni udział poprawnych:',
                          row.get('permuted_valid_fraction_requested_mean'),
                          '| kwantyle 2,5% / mediana / 97,5%:',
                          quantiles.get('q025'), quantiles.get('median'), quantiles.get('q975'))
                    print('  Udział losowań co najmniej równych oryginalnemu:',
                          row.get('permutation_fraction_at_least_observed'),
                          '| brakujące wyniki infrastrukturalne:', row.get('generator_infrastructure_missing'))
                    before = row.get('original_solver') or {}
                    after = row.get('first_permutation_solver') or {}
                    print('  Lokalny BFS — oryginał:', before.get('paths_found'), '/', before.get('requested'),
                          '| pierwsze przypisanie:', after.get('paths_found'), '/', after.get('requested'))
                    print('  Błędy/braki/niezgodny budżet BFS:',
                          {key: after.get(key) for key in ('solver_infrastructure_errors', 'solver_results_missing',
                                                           'solver_budget_mismatches')})
                print('Surowy rozkład:', distribution_path, '| pełne wyniki:', permutation / 'summary.json')
                print('Wektory PDF/SVG:', [str(permutation / name) for name in figures
                                          if Path(name).suffix.lower() in ('.pdf', '.svg')])
                for name in figures:
                    if Path(name).suffix.lower() == '.png':
                        print('Rozkład permutacyjny:', permutation / name)
                        display(Image(filename=str(permutation / name)))
        """),
        markdown("""
        ### Opcjonalny akapit do rozważenia w proposalu

        Poniższy tekst po angielsku powstaje wyłącznie z ukończonych wyników
        100/50. Jest propozycją do ręcznej oceny, nie zmianą proposalu.
        """),
        code("""
        if not expanded_real_complete or not expanded_historical_complete:
            print('Brak dwóch ukończonych ocen 100/50; akapit nie został wygenerowany.')
        else:
            real_summary = read_json(run / 'assessment_100/summary.json') or {}
            historical_summary = read_json(run / 'assessment_historical_x10_50/summary.json') or {}
            def method_table(summary, expected_boards):
                methods = {item.get('method'): item for item in summary.get('methods', [])}
                if set(methods) != {'B', 'D', 'R'}:
                    return None
                for row in methods.values():
                    if (row.get('boards') != expected_boards or row.get('requested_outputs') != 2 * expected_boards
                            or any(not isinstance(row.get(key), int) or isinstance(row.get(key), bool)
                                   for key in ('valid_intents', 'paths_found'))):
                        return None
                return methods
            four = method_table(real_summary, 100)
            ten = method_table(historical_summary, 50)
            if four is None or ten is None:
                print('Mianowniki lub ramiona 100/50 są niekompletne; akapit nie został wygenerowany.')
            else:
                print(
                    'In an exploratory assessment on 100 real Sokoban boards with four boxes and '
                    '50 previously inspected synthetic boards with ten boxes, two greedy proposals were '
                    'recorded per board. The demonstration-trained LoRA produced valid non-trivial intents '
                    f"in {four['D']['valid_intents']}/200 and {ten['D']['valid_intents']}/100 outputs, "
                    f"compared with {four['B']['valid_intents']}/200 and {ten['B']['valid_intents']}/100 "
                    'for the base model. A bounded local search found action sequences for '
                    f"{four['D']['paths_found']}/200 and {ten['D']['paths_found']}/100 LoRA outputs. "
                    'The two outputs on each board are not independent, and the cohorts differ in both '
                    'source and box count. These findings concern generator validity and bounded local '
                    'search; they do not establish an advantage from policy-relative scale conditioning, '
                    'execution by Luna, transfer, or complete-task planning.'
                )
        """),
        markdown("""
        ### Przykłady zamiarów D i lokalnego BFS

        Te dwa przypadki wybrano **po zobaczeniu wyników**, żeby zilustrować
        poprawny i niepoprawny zamiar. Nie są losową ani reprezentatywną próbą.
        Pokazana sekwencja pochodzi z lokalnego BFS, a nie z próby Luny.
        """),
        code("""
        if not real_complete:
            print('Brak kompletnej oceny ×4 do pokazania przykładów.')
        else:
            examples = read_jsonl(run / 'assessment/results.jsonl')
            successful = next((row for row in examples
                               if row.get('arm') == 'D' and row.get('valid_intent')
                               and row.get('path_found') is True and isinstance(row.get('search'), dict)), None)
            invalid = next((row for row in examples if row.get('arm') == 'D'
                            and row.get('prediction_status') == 'invalid_intent'), None)
            for title, row in (('D: poprawny zamiar i sekwencja BFS', successful),
                               ('D: niepoprawny zamiar', invalid)):
                print('\\n' + title)
                if row is None:
                    print('Brak zapisanego przypadku tego typu.')
                    continue
                print('Plansza startowa:', row.get('layout_id'))
                print('\\n'.join(row.get('start_observation') or []))
                print('Zamiar:', json.dumps(row.get('parsed'), ensure_ascii=False))
                print('Wynik walidacji:', row.get('validation_errors'))
                search = row.get('search')
                if not isinstance(search, dict):
                    print('BFS nie był uruchomiony; brak stanu po sekwencji.')
                    continue
                search_status = 'znaleziono sekwencję' if search.get('witness_found') else search.get('status')
                print('BFS:', search_status, '| akcje:', search.get('witness_actions'),
                      '(0=góra, 1=dół, 2=lewo, 3=prawo)')
                print('Plansza po sekwencji BFS:')
                print('\\n'.join(search.get('reached_observation') or []))
        """),
        markdown("""
        ### Osiągnięty zamiar, który pogarsza stan planszy

        Ten przykład z historycznej kohorty ×10 wybrano po zobaczeniu wyników,
        ponieważ liczba skrzynek na celach spadła. Osiągnięcie zamiaru przez
        lokalny BFS nie gwarantuje postępu w zadaniu. To nie jest próba Luny
        ani wynik rozwiązania całej planszy.
        """),
        code("""
        if not historical_complete:
            print('Brak kompletnej historycznej oceny ×10 do pokazania przykładu.')
        else:
            rows = read_jsonl(run / 'assessment_historical_x10/results.jsonl')
            example = next((row for row in rows if row.get('arm') == 'D' and row.get('path_found') is True
                            and isinstance(row.get('goal_box_change'), int) and row['goal_box_change'] < 0
                            and isinstance(row.get('search'), dict)), None)
            if example is None:
                print('Brak zapisanego pogorszenia w osiągniętych zamiarach D.')
            else:
                search = example['search']
                print('Wybrana plansza:', example.get('layout_id'))
                print('Plansza startowa:\\n' + '\\n'.join(example.get('start_observation') or []))
                print('Zamiar D:', json.dumps(example.get('parsed'), ensure_ascii=False))
                print('Lokalny BFS znalazł sekwencję:', search.get('witness_actions'),
                      '(0=góra, 1=dół, 2=lewo, 3=prawo)')
                print('Plansza po sekwencji BFS:\\n' + '\\n'.join(search.get('reached_observation') or []))
                print('Zmiana liczby skrzynek na celach:', example['goal_box_change'])
        """),
        markdown("""
        ## Prompty, odtwarzanie i kontynuacja

        Pełne surowe prompty oraz odpowiedzi pozostają w plikach audytowych;
        notebook nie powiela ich w dziesiątkach komórek. Stan, hash danych,
        rezerwacje i log błędu należy sprawdzić przed ewentualnym wznowieniem.
        """),
        markdown("""
        Nazwa pola `effects` pochodzi z formatu danych: jego elementy to warunki
        określające **zamiar**, nie osobny cel zadania. Wszystkie muszą być
        fałszywe na początku i spełnione jednocześnie, aby uznać zamiar za
        zrealizowany. `preserve` określa warunki obowiązujące przez całą próbę;
        może być pustą listą. Współrzędne liczymy od zera jako `(wiersz, kolumna)`.
        Obsługiwane są `box_at`, `player_at` i `boxes_on_goals_at_least`.
        Opcjonalne `instruction` nie zastępuje tych sprawdzalnych warunków.
        `task_goal` opisuje natomiast cel całej planszy: umieszczenie wszystkich
        skrzynek na celach. Realizacja jednego zamiaru może nie rozwiązać zadania.

        Poniższy przykład przedstawia jeden możliwy kształt JSON, a nie wynik
        dla konkretnej planszy:

        ```json
        {"effects": [{"predicate": "box_at", "row": 2, "col": 3}], "preserve": []}
        ```
        """),
        markdown("""
        ### Rzeczywiste wiadomości generatorów B i D

        Pokazujemy pierwszy kontekst z zamrożonego bloku testowego i ten sam
        slot dla B oraz D. Wybór nie zależy od jakości odpowiedzi. Są to pełne
        zapisane wiadomości przekazane przez szablon czatu tokenizera, a nie
        rekonstrukcja z późniejszej planszy. B to model bazowy; D to ten sam
        model z adapterem wyuczonym wyłącznie na demonstracjach. Jeśli dane
        któregoś ramienia nie są jeszcze zapisane, pojawi się informacja o braku.
        """),
        code("""
        frozen = read_jsonl(run / 'inputs/test_first_block.jsonl')
        if not frozen:
            frozen = read_jsonl(run / 'inputs/test_full_block.jsonl')
        if not frozen:
            print('Brak zamrożonego bloku testowego do pokazania promptów.')
        else:
            frozen_id = frozen[0].get('record_id')
            print('Pierwszy zamrożony kontekst:', frozen_id, '| slot: 0')
            for arm in ('B', 'D'):
                path = run / 'predictions' / (arm + '_slot0.jsonl')
                matches = [row for row in read_jsonl(path)
                           if row.get('record_id') == frozen_id and row.get('arm') == arm
                           and isinstance(row.get('raw_output'), str) and isinstance(row.get('prompt'), list)]
                if len(matches) != 1:
                    print(arm + ': brak jednoznacznej zapisanej odpowiedzi dla tego kontekstu.', path)
                    continue
                row = matches[0]
                print('\\n---', arm, '| źródło:', path, '| status:', row.get('status'), '---')
                for index, message in enumerate(row['prompt']):
                    print('[' + str(index) + '] role=' + str(message.get('role')))
                    content = message.get('content')
                    print(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2))
                    extra = {key: value for key, value in message.items() if key not in ('role', 'content')}
                    if extra:
                        print('Pozostałe pola wiadomości:', json.dumps(extra, ensure_ascii=False))
                print('Pełna surowa odpowiedź:', row['raw_output'])
        """),
        markdown("""
        ### Rzeczywisty prompt kontrolera Luny

        Kontroler wybiera pojedynczą akcję w każdym wywołaniu. Pokazujemy pełny
        prompt jednego wywołania z zapisanej próby, wraz z wcześniejszymi akcjami
        tej samej próby, jeśli występują. Wyświetlamy wyłącznie treść promptu,
        a nie konfigurację uwierzytelnienia ani pełny rekord transportu.
        """),
        code("""
        shown = False
        for stage, budget_name in (('pilot_continued', 'budget_carried'), ('pilot', 'budget')):
            audits = attempt_audits(run / budget_name)
            for row in read_jsonl(run / stage / 'results.jsonl'):
                steps = (row.get('result') or {}).get('steps') or []
                if not steps:
                    continue
                step = steps[1] if len(steps) > 1 else steps[0]
                prompt = step.get('prompt')
                if not isinstance(prompt, str):
                    continue
                tag = 'final:' + str(row.get('record_id')) + ':attempt:' + str(row.get('attempt'))
                matches = [item for item in audits.get(tag, [])
                           if isinstance(item.get('request'), dict) and item['request'].get('input') == prompt]
                print('Etap:', stage, '| rekord:', row.get('record_id'), '| próba:', row.get('attempt'),
                      '| krok:', step.get('index'), '| dokładnych dopasowań w audycie:', len(matches))
                print('Pełny zapisany prompt Luny:\\n' + prompt)
                shown = True
                break
            if shown:
                break
        if not shown:
            print('Brak zapisanego promptu kontrolera Luny.')
        """),
        code("""
        print('Surowe żądania i odpowiedzi Luny:', current.parent / 'luna_requests')
        print('Zdarzenia prób:', run / 'pilot/events.jsonl', 'oraz', run / 'pilot_continued/events.jsonl')
        print('Prompty treningowe adapterów:', run / 'adapters/<arm>/prompts.jsonl')
        print('Wyjścia generatorów i ich prompty:', run / 'predictions')
        print('Reprodukcja samego notebooka:')
        print('  python -m scripts.build_luna_policy_followup_report --run-dir', run,
              '--output', run / 'luna_policy_followup_report.ipynb')
        continuation = run / 'continuation_commands.txt'
        if continuation.is_file():
            print('Zapisana komenda kontynuacji / warunki:\\n' + continuation.read_text(encoding='utf-8'))
        else:
            print('Brak zweryfikowanej komendy wznowienia. Nie ponawiaj niepewnej rezerwacji automatycznie.')
        """),
    ]
    return nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    })


def execute_notebook(document):
    """Execute our generated cells in-process, without Jupyter's network sockets."""
    shell = InteractiveShell.instance()
    execution_count = 0
    for index, cell in enumerate(document.cells):
        if cell.cell_type != "code":
            continue
        execution_count += 1
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(cell.source, store_history=False)
        if result.error_before_exec or result.error_in_exec:
            error = result.error_before_exec or result.error_in_exec
            raise RuntimeError(f"Report cell {index} failed: {error}") from error
        outputs = []
        if captured.stdout:
            outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
        if captured.stderr:
            outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
        for item in captured.outputs:
            outputs.append(nbformat.v4.new_output("display_data", data=item.data, metadata=item.metadata))
        cell.execution_count = execution_count
        cell.outputs = outputs
    nbformat.validate(document)
    return document


def write_report(run_dir: Path, output: Path, root: Path = ROOT) -> None:
    document = execute_notebook(notebook(run_dir, root))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        nbformat.write(document, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run_dir / "luna_policy_followup_report.ipynb"
    write_report(args.run_dir, output)
    print(f"Wrote executed offline report: {output}")


if __name__ == "__main__":
    main()
