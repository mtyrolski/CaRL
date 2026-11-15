#!/usr/bin/env python3
"""Build an offline notebook from the saved Figure 1 checkpoint comparison."""

import argparse
import os
from pathlib import Path

import nbformat

from scripts.build_luna_policy_followup_report import code
from scripts.build_luna_policy_followup_report import execute_notebook
from scripts.build_luna_policy_followup_report import markdown

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "analysis/figure1_cllp_matched_20261115T0555Z"


def notebook(run_dir: Path, root: Path = ROOT):
    """Describe the completed comparison without rerunning any controller."""
    setup = f"""
        from pathlib import Path
        import csv
        import json
        from collections import Counter
        import matplotlib
        from IPython.display import HTML, Image, Markdown, display

        matplotlib.use('Agg', force=True)
        root = Path({str(root.resolve())!r})
        run = Path({str(run_dir.resolve())!r})

        def read_json(path):
            return json.loads(path.read_text(encoding='utf-8'))

        def read_jsonl(path):
            with path.open(encoding='utf-8') as source:
                return [json.loads(line) for line in source if line.strip()]

        def read_csv(path):
            with path.open(newline='', encoding='utf-8') as source:
                return list(csv.DictReader(source))

        print('Katalog badania:', run)
        """
    cells = [
        markdown("""
        # Figure 1: checkpointy CLLP na 12 × 12, 10 skrzynek

        Sprawdzamy, czy lokalne checkpointy CaRL osiągają **dokładnie te same
        plansze docelowe**, które wcześniej otrzymały Luna i Terra w Figure 1.
        Notebook wyłącznie czyta zapisane lokalnie wyniki i odtwarza przykład
        w symulatorze. Nie uruchamia modelu, treningu ani API.

        To badanie osiągania pełnego stanu, łącznie z pozycją gracza. CLLP
        dostaje pełną planszę docelową. Wynik nie jest pomiarem kontrolera
        otrzymującego wyłącznie zamiar ani skuteczności rozwiązania całej gry.
        """),
        code(setup),
        markdown("""
        ## Co dokładnie porównujemy?

        Wykorzystujemy **20 różnych układów** z wcześniej badanego, syntetycznego
        zbioru 12 × 12 × 10: jeden stan początkowy na układ i po sześć stanów
        docelowych. Wyniki są sparowane po identyfikatorze układu, stanie
        początkowym, stanie docelowym oraz `k`. Razem to **120 par stanów**,
        po jednej próbie na parę dla każdego modelu.

        `k` to liczba akcji dzielących stany w zapisanej demonstracji. Nie musi
        być najkrótszą odległością ani czasem, którego potrzebuje kontroler.
        Każdy kontroler ma limit `L = 32` wykonywanych akcji; nieudane ruchy
        i ruchy bez zmiany planszy też wliczają się do limitu. Próba kończy się
        przy pierwszym dokładnym trafieniu, terminalu zadania albo limicie.

        Luna i Terra otrzymywały po 32 przykłady, a ich akcje były losowane
        z podanego rozkładu (`policy temperature = 1`). CLLP-8 i CLLP-32
        używają osobno wytrenowanych checkpointów, bez przykładów w prompcie,
        i wybierają zachłannie akcję o największym wyniku. Te same stany i limit
        akcji **nie oznaczają jednakowego czasu ani kosztu obliczeń**.

        Układy i wyniki były już wcześniej oglądane. To porównanie eksploracyjne,
        bez gwarancji całkowitego braku wcześniejszej ekspozycji danych.
        """),
        code("""
        summary = read_csv(run / 'figure/summary.csv')
        paired = read_csv(run / 'figure/paired_cases.csv')
        offsets = (1, 2, 4, 8, 12, 16)
        models = ('Luna', 'Terra', 'CLLP-8', 'CLLP-32')
        assert len(summary) == 24 and len(paired) == 120
        assert len({row['layout_id'] for row in paired}) == 20
        assert len({row['context_id'] for row in paired}) == 20
        assert all(int(row['boards']) == 20 for row in summary)
        for model in ('cllp8', 'cllp32'):
            status = read_json(run / model / 'status.json')
            assert status['state'] == 'complete' and status['completed'] == status['requested'] == 120
        print('Układy:', len({row['layout_id'] for row in paired}),
              '| konteksty:', len({row['context_id'] for row in paired}),
              '| pary stanów:', len(paired), '| k:', offsets)
        """),
        markdown("""
        ## Wyniki

        Komórka `17/20; 85% [64%, 95%]` oznacza 17 dokładnych trafień na
        20 prób oraz punktowy 95% przedział Wilsona. Przedziały dotyczą
        poszczególnych wartości `k`; nie są jednoczesnym pasmem ufności i nie
        mierzą zmienności treningu. Jedna próba Luny lub Terry na parę nie
        pozwala precyzyjnie oszacować losowości ich odpowiedzi dla tej pary.
        """),
        code("""
        lookup = {(row['model'], int(row['k'])): row for row in summary}
        lines = ['| k | ' + ' | '.join(models) + ' |', '|---:|' + '---:|' * len(models)]
        for k in offsets:
            entries = []
            for model in models:
                row = lookup[model, k]
                entries.append(f"{row['hits']}/{row['boards']}; {100*float(row['success_fraction']):.0f}% "
                               f"[{100*float(row['wilson_low']):.0f}%, {100*float(row['wilson_high']):.0f}%]")
            lines.append('| ' + str(k) + ' | ' + ' | '.join(entries) + ' |')
        display(Markdown('\\n'.join(lines)))
        display(Image(filename=str(run / 'figure/figure1_cllp_matched.png')))
        print('Wektorowe wersje:', run / 'figure/figure1_cllp_matched.pdf',
              'oraz', run / 'figure/figure1_cllp_matched.svg')
        """),
        markdown("""
        Przedziały na wykresie są policzone dla 20 układów, a nie dla 120
        par jako niezależnych plansz. Kolejne wartości `k` dotyczą tej samej
        dwudziestki i są ze sobą powiązane. Linie jedynie łączą zmierzone
        punkty, bez dopasowywania funkcji.
        """),
        code("""
        differences = read_csv(run / 'figure/paired_differences.csv')
        print('Przykład sparowanej różnicy przy k=16:')
        for row in differences:
            if int(row['k']) == 16 and row['comparator'] == 'Terra':
                print(row['method'], 'minus Terra:',
                      f"{100*float(row['difference']):+.0f} punktów procentowych; "
                      f"bootstrap plansz [{100*float(row['bootstrap_low']):+.0f}, "
                      f"{100*float(row['bootstrap_high']):+.0f}]; n={row['paired_boards']}")
        print('Wszystkie wyniki:', run / 'figure/summary.csv')
        print('Wszystkie różnice:', run / 'figure/paired_differences.csv')
        print('Pary i indywidualne wyniki:', run / 'figure/paired_cases.csv')
        """),
        markdown("""
        Różnice oraz bootstrap są **opisowe** dla tej wcześniej analizowanej
        dwudziestki i różnych procedur modeli. To nie jest kontrolowany test
        wpływu samego treningu lub uczenia w kontekście.

        ## Przykład pary stanów

        Poniżej celowo wybieramy parę, na której wyniki CLLP-8 i CLLP-32
        się różnią. Przykład ilustruje format danych; nie jest losową,
        reprezentatywną planszą. Zapisane akcje odtwarzamy w CaRL.
        """),
        code("""
        from carl.icl.boards import from_rows, state_hash, to_rows
        from carl.icl.data_sources.sokoban import environment

        cases = [row for row in paired if row['CLLP-8'] == 'True' and row['CLLP-32'] == 'False']
        if not cases:
            raise RuntimeError('Brak zapisanej pary z różnymi wynikami CLLP.')
        example = sorted(cases, key=lambda row: (int(row['k']), row['pair_id']))[0]
        states = read_json(root / 'data/icl/reachability_proposal_hard_12_v1/states.json')['states']
        source_rows = states[example['source_hash']]
        target_rows = states[example['target_hash']]
        print('k:', example['k'], '| para:', example['pair_id'], '| układ:', example['layout_id'])
        print('\\nStan początkowy:\\n' + '\\n'.join(source_rows))
        print('\\nPełny stan docelowy:\\n' + '\\n'.join(target_rows))
        names = ('UP', 'DOWN', 'LEFT', 'RIGHT')
        for name, folder in (('CLLP-8', 'cllp8'), ('CLLP-32', 'cllp32')):
            row = next(item for item in read_jsonl(run / folder / 'trials.jsonl')
                       if item['pair_id'] == example['pair_id'])
            state = from_rows(source_rows)
            env = environment(state)
            for action in row['actions']:
                state = env.next_state(state, action)
            assert state_hash(state) == row['state_hashes'][-1]
            print('\\n', name, '| trafienie:', row['exact_hit'], '| akcji:', row['attempted_actions'],
                  '| powód zatrzymania:', row['stop_reason'])
            print('Akcje:', ' '.join(names[action] for action in row['actions']))
            print('Osiągnięty stan:\\n' + '\\n'.join(to_rows(state)))
        """),
        markdown("""
        ## Rzeczywiste prompty dla tej pary

        Poniżej są pełne, zapisane wcześniej wiadomości wejściowe pierwszego
        wywołania Luny i Terry dla pary pokazanej wyżej. Sekcje można rozwinąć.
        Odczytujemy pole `request.payload.input` z oryginalnego cache; nie
        rekonstruujemy promptów i nie wysyłamy nowych zapytań. Wcześniejsze
        przykłady, role wiadomości oraz dwie plansze są częścią tych danych.

        Każdy kolejny krok mógł mieć inne wejście i osobny zapis w śladzie;
        nie należy traktować pierwszego promptu jako całej historii próby.
        CLLP otrzymuje tokeny dwóch pełnych plansz, a nie prompt tekstowy.
        Temperatura losowania **akcji** wynosiła 1; temperatura generowania
        tekstu przez usługę nie była jawnie ustawiona w tych żądaniach.
        """),
        code("""
        import hashlib
        from html import escape
        from carl.icl.reachability_data import canonical_bytes

        pilot = root / 'analysis/icl_reachability/figure1_proposal_hard_recovery_v5'
        pilot_rows = read_csv(pilot / 'pilot20_export/trials.csv')
        manifest = read_json(pilot / 'campaign/study_12/manifest.json')
        for arm, label in (('luna', 'Luna'), ('terra_6_1', 'Terra')):
            matches = [row for row in pilot_rows if row['family'] == '12' and row['arm'] == arm
                       and row['context'] == example['context_id'] and row['k'] == example['k']]
            if len(matches) != 1:
                raise RuntimeError(f'Brak jednoznacznego historycznego śladu: {arm}')
            record = matches[0]
            trace_path = Path(record['trace_path'])
            trace = read_json(trace_path if trace_path.is_absolute() else root / trace_path)
            identity = trace['identity']
            for key, value in (('key', record['trial_key']), ('arm', arm), ('pair', example['pair_id']),
                               ('context', example['context_id']), ('layout', example['layout_id']),
                               ('source_hash', example['source_hash']), ('target_hash', example['target_hash']),
                               ('k', int(example['k']))):
                if identity[key] != value:
                    raise RuntimeError(f'Niezgodna para lub ślad {arm}: {key}')
            first = trace['requests'][0]
            if first['step'] != 0:
                raise RuntimeError(f'Pierwsze wywołanie nie dotyczy kroku 0: {arm}')
            request_hash = first['request_hash']
            cache_path = pilot / 'campaign/cache_12/live' / f'{request_hash}.json'
            cached = read_json(cache_path)
            request = cached['request']
            if hashlib.sha256(canonical_bytes(request)).hexdigest() != request_hash:
                raise RuntimeError(f'Hash zapisanego żądania nie pasuje do śladu: {arm}')
            payload = request['payload']
            if (trace['status'] != 'complete' or cached['status'] != 'completed'
                    or payload['model'] != identity['model'] or cached['requested_model'] != identity['model']
                    or cached['model'] != first['returned_model'] or request['slot'] != first['slot']):
                raise RuntimeError(f'Niezgodny model lub slot zapisanego wywołania: {arm}')
            config = manifest['study']['arms'][arm]
            print(label, '| para:', identity['pair'], '| request hash:', request_hash)
            print('Model żądany:', payload['model'], '| zwrócony:', cached['model'],
                  '| liczba przykładów:', manifest['study']['few_shot'],
                  '| temperatura akcji:', manifest['study']['policy_temperature'])
            print('Temperatura tekstu:', payload.get('temperature', 'nie ustawiono'),
                  '| reasoning:', payload.get('reasoning'), '| max_output_tokens:', payload['max_output_tokens'],
                  '| format:', payload['text']['format'], '| liczba zapisanych wywołań w śladzie:',
                  len(trace['requests']))
            print('Zamrożony profil modelu:', config['model_id'], '| źródło:', cache_path)
            messages = json.dumps(payload['input'], ensure_ascii=False, indent=2)
            display(HTML(f'<details><summary>Pełne zapisane request.payload.input — {escape(label)}</summary>'
                         '<pre style="white-space:pre-wrap;max-height:36em;overflow:auto">'
                         + escape(messages) + '</pre></details>'))
        """),
        markdown("""
        ## Kontrola śladów i pochodzenie

        Niezależne odtworzenie sprawdza zapisane akcje i stany w symulatorze;
        samo w sobie nie oznacza ponownego uruchomienia sieci. Osobna kontrola
        CLLP-32 porównuje wybrane decyzje publicznego interfejsu na CPU
        z zapisanymi decyzjami GPU. Nie oznacza to sprawdzenia wszystkich
        decyzji ani identyczności pełnych wektorów wyników.
        """),
        code("""
        audit = read_json(run / 'independent_verification.json')
        print('Niezależnie sprawdzone pary:', audit['pair_count'])
        for model, item in audit['models'].items():
            print(model, ': próby =', item['checked_trials'], '| akcje =', item['checked_actions'],
                  '| bez zmiany stanu =', item['no_ops'], '| powody zatrzymania =', item['stop_reasons'])
        legacy = audit['legacy_cllp8']
        print('Wspólne wcześniejsze pary CLLP-8:', legacy['matched_and_replayed'],
              '| zgodne akcje:', legacy['equal_actions'], '| zgodne wyniki:', legacy['equal_outcomes'])
        parity_path = run / 'cllp32_public_cpu_parity.json'
        if parity_path.is_file():
            parity = read_json(parity_path)
            print('CLLP-32, wybrane decyzje CPU/GPU:', parity['matches'], '/', parity['checked'],
                  '| sprawdzony zakres:', parity['scope'])
        else:
            print('Kontrola wybranych decyzji CLLP-32 nie jest dostępna.')
        for folder in ('cllp8', 'cllp32'):
            info = read_json(run / folder / 'provenance.json')
            status = read_json(run / folder / 'status.json')
            print('\\n', info['model'], '| checkpoint:', info['checkpoint'])
            print('Urządzenie:', info['device'], info.get('gpu_name'), '| typ:', info['dtype'],
                  '| batch:', info['batch_size'], '| czas uruchomienia:', round(status['seconds'], 2), 's')
            print('Hashe modelu:')
            for path, digest in info['sha256'].items():
                if path.endswith(('config.json', 'pytorch_model.bin')):
                    print(' ', path, digest)
        print('Pełne pochodzenie i hashe:', run / 'figure/provenance.json')
        """),
        markdown("""
        ## Odtworzenie

        Następna komenda przebudowuje i wykonuje tylko ten notebook na podstawie
        zapisanych wyników. Dłuższe eksperymenty, użycie GPU i API nie są
        potrzebne. Surowe ślady checkpointów oraz indywidualne pary pozostają
        w katalogu badania; źródłowe wyniki Luny i Terry znajdują się w
        `analysis/icl_reachability/figure1_proposal_hard_recovery_v5/`.
        """),
        code("""
        print('Z katalogu CaRL:')
        print('uv run --offline --no-sync python -m scripts.build_figure1_cllp_report',
              '--run-dir', run, '--output', run / 'figure1_cllp_report.ipynb')
        print('Próby checkpointów:', run / 'cllp8/trials.jsonl', 'i', run / 'cllp32/trials.jsonl')
        print('Niezależny audyt:', run / 'independent_verification.json')
        """),
    ]
    return nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="Write cells without executing them")
    args = parser.parse_args()
    output = args.output or args.run_dir / "figure1_cllp_report.ipynb"
    document = notebook(args.run_dir)
    if not args.prepare_only:
        document = execute_notebook(document)
    nbformat.validate(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        nbformat.write(document, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Wrote {'unexecuted' if args.prepare_only else 'executed'} notebook: {output}")


if __name__ == "__main__":
    main()
