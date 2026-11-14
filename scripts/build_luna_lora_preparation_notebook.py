#!/usr/bin/env python3
"""Build a local, model-free notebook for the prepared Luna/LoRA study."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "id": hashlib.sha256(text.encode()).hexdigest()[:12], "metadata": {},
            "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "id": hashlib.sha256(text.encode()).hexdigest()[:12], "execution_count": None,
            "metadata": {}, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


def notebook() -> dict:
    cells = [
        markdown("""
# Luna i LoRA: przygotowanie eksperymentu

Ten notebook pokazuje protokół, dane wejściowe i dostępne wyniki. Status
przyszłego biegu jest odczytywany z jego plików, jeśli istnieją. Żadna komórka nie wysyła
zapytań API, nie trenuje modelu i nie uruchamia ewaluacji kontrolera.

Pytamy, czy żądanie względnej skali `u` zmienia propozycje generatora, gdy
oba adaptery Qwena są uczone na tych samych pomiarach zamrożonej Luny.
Najpierw trzeba zebrać pomiary; poniższe intencje są sprawdzonymi przykładami
z demonstracji, a nie wynikami Luny.
"""),
        markdown("""
## Projekt i interpretacja

- Luna dostaje bieżącą planszę, cel, intencję i historię własnych działań.
  Jej wagi, prompt, parser, pamięć i limit 16 akcji pozostają stałe.
- W każdym stanie przygotowano cztery różne intencje. Zaplanowano po jednej
  próbie każdej z nich przy dekodowaniu temperature=0. `k` oznacza wyłącznie
  offset demonstracji. Taka pojedyncza próba nie mierzy zmienności ani
  stabilności skuteczności polityki. Wszystkie osiągnięte kandydaty mają
  wtedy obserwowane `p=1` i jednakową wagę w swoim kontekście.
- Dopiero komplet pomiarów całego kontekstu pozwoli wyliczyć ważony percentyl
  `u`; nieudane próby i koszty zostają w danych. Brak pomiaru to nie porażka.
- Dwie LoRA mają korzystać z identycznych pomierzonych przykładów, wag,
  kolejności, rozmiaru i budżetu. Jedna otrzymuje `u`, druga znacznik braku
  skali. Kontroler nie będzie trenowany.
- Test wykorzystuje te same konteksty dla metod. Osobno policzymy poprawność
  odpowiedzi, skuteczność Luny, czas, nieuruchomione i błędy infrastruktury.
  Niektóre plansze były już oglądane w poprzednich analizach, więc to pilot
  eksploracyjny, a nie dziewiczy test generalizacji.

Jeśli zmierzymy zbyt mało różnych udanych czasów w tym samym stanie,
warunkowanie `u` pozostanie nierozstrzygnięte. Nie zastąpimy go offsetem `k`.
"""),
        code("""
import html
import json
import os
from collections import Counter
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import HTML, Image, display

def show_figure(fig):
    image = BytesIO()
    fig.savefig(image, format='png', dpi=150, bbox_inches='tight')
    display(Image(data=image.getvalue()))
    plt.close(fig)

def find_root(path):
    for candidate in (path, *path.parents):
        if (candidate / 'configs/icl/luna_lora_prepared.json').is_file():
            return candidate
    raise FileNotFoundError('Uruchom notebook wewnątrz checkoutu CaRL.')

root = find_root(Path.cwd().resolve())
prepared = root / 'analysis/luna_lora_preparation_20261114'
run_dir = Path(os.environ.get('CARL_LUNA_LORA_RUN', str(Path.cwd()))).resolve()
data_root = run_dir / 'inputs' if (run_dir / 'inputs/config.json').is_file() else prepared
config_path = data_root / 'config.json' if data_root != prepared else root / 'configs/icl/luna_lora_prepared.json'
configuration = json.loads(config_path.read_text())
data_manifest = json.loads((data_root / 'data/manifest.json').read_text())
test_manifest = json.loads((data_root / 'evaluation/manifest.json').read_text())
records = [json.loads(line) for line in (data_root / 'data/calibration_candidates.jsonl').read_text().splitlines()]
status_path = run_dir / 'study_status.json'
supervisor_path = run_dir / 'run_manifest.json'
if supervisor_path.is_file():
    status = json.loads(supervisor_path.read_text())
    print('Status supervisora w chwili wykonania notebooka:', status['status'])
if status_path.is_file():
    recorded_status = json.loads(status_path.read_text())
    print('Ostatni zapisany status badania:', recorded_status)
    if supervisor_path.is_file() and status['status'] != 'active' and recorded_status['status'] == 'running':
        print('Proces sterujący przerwano; ostatni status badania może być nieaktualny.')
elif not supervisor_path.is_file():
    print('Status: przygotowanie, żadne próby modelu nie zostały uruchomione.')
"""),
        markdown("""
## Dane i kontrola rozdziału

Źródłem kandydatów są wcześniej zamrożone trajektorie 12×12×10 z podziałem
według układu planszy i jego symetrii. Każdy fragment odtworzono w środowisku,
a warunki intencji sprawdzono na obserwacjach. Test wybiera po jednym stanie
na planszę bez korzystania z wyników modeli. Warianty ×4 i ×10 pochodzą
z różnych źródeł, więc nie izolują wpływu liczby skrzynek.
"""),
        code("""
rows = []
for split, group in data_manifest['counts'].items():
    rows.append({'podział': split, 'plansze': group['boards'],
                 'konteksty': group['contexts'], 'intencje': group['records']})
print(pd.DataFrame(rows).to_string(index=False))
print('Rodziny testowe:', test_manifest['counts_by_family'])
print('Planowany górny limit prób kalibracji:',
      len(records) * configuration['calibration']['attempts_per_candidate'])
fig, ax = plt.subplots(figsize=(6, 3))
counts = Counter(r['effect_type'] for r in records)
ax.bar(counts.keys(), counts.values(), color='#58779b')
ax.set_ylabel('Liczba intencji')
ax.set_title('Przygotowane intencje — nie wyniki modelu')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
show_figure(fig)
"""),
        markdown("""
## Przykład danych

Pokazujemy bieżącą i późniejszą planszę z zapisanego fragmentu. Późniejsza
plansza uzasadnia adnotację, ale nie będzie przekazywana generatorowi ani
Lunie. To nie jest stan osiągnięty przez kontroler w eksperymencie.
"""),
        code("""
example = records[0]
print('Intencja:', json.dumps(example['intent'], indent=2))
print('Początek:\\n' + '\\n'.join(example['start_observation']))
print('Późniejsza obserwacja demonstracji:\\n' + '\\n'.join(example['later_observation']))
"""),
        markdown("""
## Budżet i bezpieczeństwo uruchomienia

Nowy bieg ma własny, wspólny ledger z limitem **2 USD** obejmującym wszystkie
jego wywołania Luny. Stara kampania zachowuje dwie nierozliczone rezerwacje;
nowy limit ich nie zwalnia ani nie nazywa wydatkiem. W razie niepewnej
odpowiedzi transport zatrzymuje się i nie ponawia zapytania automatycznie.

Przed treningiem trzeba zmierzyć pamięć GPU i czas pełnego kroku. Etapy
mają działać pod supervisorem z oddzielnym watchdogiem. Limit biegu to
4 godziny 30 minut, z zatrzymaniem treningu po 4 godzinach i czasem na raport.
Jeśli pierwszy pomiar zasobów się nie powiedzie, przewidziano jedną próbę
z mniejszą LoRA (rank 2 zamiast 4). Nie gwarantuje to zmieszczenia modelu
w pamięci. Oba warianty porównania otrzymają te same ustawienia.
W samym przygotowaniu nie uruchomiono supervisora ani procesu eksperymentu.
Próba zasobów odbędzie się przed pierwszym płatnym zapytaniem. W tym teście
`u=0,5` będzie wyłącznie sztucznym wejściem diagnostycznym, bez tworzenia
etykiet Luny. Potem rozmiary tokenów zostaną sprawdzone ponownie na danych
z faktycznych pomiarów.
"""),
        code("""
print(json.dumps(configuration['controller'], indent=2))
print('Terminy i ustawienia:', {k: configuration[k] for k in
      ('deadline_seconds', 'expensive_start_cutoff_seconds', 'training_stop_seconds')})
"""),
        markdown("""
## Pełne prompty i wyniki po uruchomieniu

Poniżej wyświetlamy zapisany prompt kontrolera i bazowy prompt generatora.
Poszczególne komunikaty i surowe odpowiedzi przyszłego biegu mają zostać
zapisane w jego katalogu. Pomiar `u`, krzywe treningu i wykresy skuteczności
można narysować dopiero, gdy odpowiednie pliki rzeczywiście powstaną.
"""),
        code("""
from carl.icl.luna_intent_rollout import POLICY_PROMPT
from carl.icl.policy_relative_lora import Arm, generator_messages

print('PROMPT KONTROLERA:\\n', POLICY_PROMPT)
print('PRZYKŁADOWY PEŁNY PROMPT GENERATORA ZE SKALĄ:\\n',
      json.dumps(generator_messages(records[0], Arm.SCALED, 0.5), indent=2))
print('PRZYKŁADOWY PEŁNY PROMPT GENERATORA BEZ SKALI:\\n',
      json.dumps(generator_messages(records[0], Arm.UNSCALED), indent=2))
"""),
        code("""
def complete_jsonl(path):
    if not path.is_file():
        return []
    items = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            print('Niepełna linia w zapisanym pliku:', path.name, number)
            break
    return items

prediction_files = sorted((run_dir / 'predictions').glob('*.jsonl'))
if prediction_files:
    for path in prediction_files:
        saved = complete_jsonl(path)
        print(path.name, 'zapisane:', len(saved), 'statusy:', dict(Counter(row['status'] for row in saved)))
        for row in saved:
            if row.get('prompt'):
                title = html.escape(path.name + ' / ' + row['record_id'])
                prompt = html.escape(json.dumps(row['prompt'], ensure_ascii=False, indent=2))
                display(HTML('<details><summary>Prompt generatora: ' + title + '</summary><pre>'
                             + prompt + '</pre></details>'))
else:
    print('Brak zapisanych promptów i predykcji nowego generatora.')

for stage in ('calibration', 'controller_test'):
    events = complete_jsonl(run_dir / stage / 'events.jsonl')
    requests = [event for event in events if event.get('type') == 'request']
    if requests:
        print(stage, 'zapisane prompty kontrolera:', len(requests))
    for event in requests:
        title = html.escape(stage + ' / ' + event['record_id'] + ' / krok ' + str(event['index']))
        prompt = html.escape(event['prompt'])
        display(HTML('<details><summary>Prompt kontrolera: ' + title + '</summary><pre>'
                     + prompt + '</pre></details>'))
"""),
        markdown("""
## Zapisane pomiary i wykresy

Puste pola oznaczają brak pomiaru, a nie wynik zero. Poniższe komórki
pokazują pliki jedynie wtedy, gdy rzeczywisty bieg je zapisał.
"""),
        code("""
calibration_path = run_dir / 'calibration/calibrated.jsonl'
if calibration_path.is_file():
    measurements = [json.loads(line) for line in calibration_path.read_text().splitlines()]
    statuses = Counter(item['policy_measurements']['status'] for item in measurements)
    complete = [item for item in measurements if item['policy_measurements']['complete']]
    reached = [item for item in complete if item['policy_measurements']['mean_success_actions'] is not None]
    print('Statusy kandydatów:', dict(statuses))
    print('Kandydaci w kompletnych kontekstach:', len(complete), 'z czego z sukcesem:', len(reached))
    if reached:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.scatter([item['policy_measurements']['u'] for item in reached],
                   [item['policy_measurements']['mean_success_actions'] for item in reached], alpha=0.7)
        ax.set_xlabel('Zmierzona skala u w kontekście')
        ax.set_ylabel('Średnia liczba działań przy sukcesie')
        plt.tight_layout()
        show_figure(fig)
else:
    print('Brak pomiarów kalibracyjnych Luny.')
"""),
        code("""
summary_path = run_dir / 'assessment/summary.csv'
if summary_path.is_file():
    print(pd.read_csv(summary_path).to_string(index=False))
else:
    print('Brak wyników oceny na planszach testowych.')

fig, ax = plt.subplots(figsize=(6, 3))
shown = False
for arm in ('unscaled', 'scaled'):
    curve = run_dir / 'adapters' / arm / 'training_curve.jsonl'
    if curve.is_file():
        measurements = [json.loads(line) for line in curve.read_text().splitlines()]
        if measurements:
            ax.plot([item['step'] for item in measurements], [item['train_loss'] for item in measurements], label=arm)
            shown = True
if shown:
    ax.set_xlabel('Krok treningu')
    ax.set_ylabel('Loss treningowy')
    ax.legend()
    plt.tight_layout()
    show_figure(fig)
else:
    plt.close(fig)
    print('Brak krzywych treningu LoRA.')
"""),
        code("""
figure_path = run_dir / 'assessment/intent_achievement.png'
if figure_path.is_file():
    display(Image(filename=str(figure_path)))
else:
    print('Brak wykresu osiągania intencji.')
"""),
        markdown("""
## Odtwarzanie

Notebook odczytuje wyłącznie dane i konfigurację na dysku. Dane są lokalnymi
artefaktami, bez wag modeli ani kluczy. Samo otwarcie ani wykonanie tego
notebooka niczego nie uruchamia w API.

Kontrola przygotowania bez żadnych prób modeli:

    uv run --offline --no-sync python -m scripts.run_luna_lora_experiment

Poniższe polecenie jest **wyłącznie na później**: uruchomi watchdog, pomiar
GPU, płatne próby Luny, trening i ewaluację. Katalog wyjściowy musi być nowy:

    uv run --offline --no-sync python -m scripts.run_luna_lora_experiment \\
        --run-dir analysis/luna_lora_run_NEW --execute

Trzeba uruchomić je w środowisku mającym dostęp do GPU i sieci API. Limit
nowych wywołań wynosi 2 USD. Przy przerwaniu lub błędzie zobacz plik
`study_status.json`, manifest supervisora, notebook i ledger w katalogu biegu.
Launcher nie ma automatycznego wznawiania po przerwaniu. Nie wznawiaj
niepewnych zapytań jako nowych prób ani nie restartuj zegara tego biegu.
Alias `gpt-6-luna` nie gwarantuje stałego snapshotu po stronie dostawcy.
"""),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                          "name": "python3"},
                                            "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(notebook(), ensure_ascii=False, indent=2) + "\n"
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Prepared notebook: {args.output}")


if __name__ == "__main__":
    main()
