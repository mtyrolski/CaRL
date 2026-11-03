# Ewaluacja i prototypowanie wykresów

Całość działa offline na zapisanych wynikach. Wykresy używają wyłącznie
`plotly.graph_objects` i `make_subplots`, bez Plotly Express. Gotowy raport:
`analysis/luna_icl/evaluation_mock/index.html`.

## Szybki start

Z głównego katalogu CaRL, w środowisku `uv sync --frozen --extra icl --group lint`:

```bash
# Powtórzenie krótkiej macierzy mock + zbudowanie raportu; bez API.
# Przy pierwszym uruchomieniu przygotuj dane: make icl-prepare
make icl-eval-demo

# Tylko przeliczenie raportu z istniejących wyników:
make icl-eval

# Wyniki wybranego eksperymentu z prawdziwą Luną:
uv run --extra icl python -m scripts.evaluate_icl \
  --config configs/icl/evaluation/live.json \
  --runs analysis/luna_icl/runs/TWOJ_KATALOG_MACIERZY \
  --output analysis/luna_icl/evaluation_live

# Opcjonalne statyczne figury do publikacji:
uv run --extra icl python -m scripts.evaluate_icl --formats png svg
```

Placeholder w `live.json` wymaga podmiany albo `--runs`; nie wybieramy automatycznie
„najnowszego” eksperymentu. `--runs` przyjmuje kilka katalogów macierzy/warunków.
Niezgodne ustawienia dostają osobne kohorty. Powtórzenia tego samego logicznego
przebiegu w różnych katalogach są odrzucane; nakładające się ścieżki wejściowe
nie podwajają danych.

## Co powstaje

- `index.html`: lokalny dashboard; `plotly.min.js` obok niego umożliwia pracę bez CDN.
- Osobne pliki `*.html` i `*.plotly.json`: każdy wykres można otworzyć lub zmieniać niezależnie.
- `summary`, `curves`, `pairs`, `points`, `attempts`, `runs` w JSON i CSV.
- `calibration_targets.csv` oraz `calibration_costs.csv`: kalibracja jest liczona osobno od searcha.
- `README.md`, rozwiązana konfiguracja, lista brakujących kalibracji i manifest SHA-256 wejść.
- Opcjonalne PNG/SVG/PDF przez Kaleido; błąd renderera przerywa eksport zamiast pomijać plik.

Zachowaj `index.html` i `plotly.min.js` w tym samym katalogu przy przenoszeniu
raportu. Same figury JSON można wczytać przez `plotly.io.read_json(...)`.

## Wykresy

Jeśli wejścia zawierają wyniki oddzielnych uruchomień dla wielu limitów, raport
dodaje wykres solve rate przy rzeczywiście uruchomionych budżetach i tabelę
`budget_sweep.csv/json`. Kompletne konfiguracje oraz instrukcje uruchomienia są
w `configs/icl/budget_sweeps/README.md`.

1. Skuteczność z bootstrapowym przedziałem 95%, z planszą jako jednostką losowania.
2. Obserwowane ukończenia względem łącznych wykonanych ruchów searcha, także nieudanych prób.
3. Rozkłady kosztu ruchów i czasu dla wszystkich przebiegów; długość planu tylko dla rozwiązanych.
4. Logiczne i fizyczne użycie tokenów oraz liczba wywołań generatora, policy, value i embeddingów.
5. Przyczyny zakończenia, w tym błędy API i wyczerpanie budżetów.
6. Trafienia w subgoale, ukończenia zadania przed trafieniem oraz próby nieudane, według skali.
7. Sparowane koszty i różnica skuteczności na wspólnych parach `(plansza, seed)`.
8. Zmierzone czasy trafienia `d`, etykiety percentylowe `u` i offsety trajektorii `k` z kalibracji.
9. Podgląd plansz z suwakiem: stan początkowy próby, żądany subgoal i rzeczywisty endpoint.

Domyślne porównanie to metody `relative` i `fixed`. Gdy w kohorcie są dwa warianty
tej samej metody, porównanie nie jest zgadywane. Ustaw np.:

```json
"comparison_configs": [["relative_12x12x4", "fixed_shared_cap_12x12x4"]]
```

## Notebook i Python

Otwórz `notebooks/luna_icl_evaluation.ipynb` z interpreterem `.venv/bin/python`.
Notebook zawiera tabelę wyników, wybór kohorty, wszystkie wykresy, podgląd plansz
i przykład modyfikowania oraz zapisywania pojedynczej figury. Sekcja eksportu jest
domyślnie wyłączona (`EXPORT = False`). Ścieżki do wyników live ustaw przez `RUN_DIRS`.
Notebook nie wysyła zapytań do API.

Bez notebooka:

```python
from carl.icl.evaluation.data import load_evaluation_config, load_runs
from carl.icl.evaluation.report import build_tables
from carl.icl.evaluation.statistics import groups
from carl.icl.evaluation.figures import cohort_figures

config = load_evaluation_config("configs/icl/evaluation/mock.json")
runs = load_runs(config)
tables = build_tables(runs, config)
cohort, matched = next(iter(groups(runs).items()))
selected = {name: [r for r in rows if r["cohort"] == cohort] for name, rows in tables.items()}
figures = cohort_figures(matched, selected)
figures["completion"].update_layout(title="Moje porównanie")
figures["completion"].show()
```

## Interpretacja

Mock nie mierzy skuteczności Luny. Mock, live i replay oraz łatwe fixture’y i pełne
stany nie są mieszane. Skuteczność jest średnią po planszach, po uśrednieniu seedów
w każdej planszy. Brakujące zaplanowane przebiegi są pokazane osobno. Jedna plansza
nie daje przedziału ufności; małe lub jednorodne próbki mogą dawać przedziały zerowej szerokości.

Krzywa budżetu opisuje obserwowane wyniki zapisanych uruchomień, nie zastępuje
eksperymentu uruchomionego ponownie z innym limitem. Krótsza nieudana próba nie
oznacza lepszej metody. Czas mock/replay/cache nie jest estymacją opóźnienia live API.
Tokeny wejściowe logiczne są lokalnym oszacowaniem, fizyczne pochodzą z odpowiedzi
providera; nie przeliczamy ich na dolary według domyślnych cen.

Statystyki subgoali obejmują zapisane zakończone próby; przerwana próba API może
nie mieć eventu. Raport nie potrzebuje rozwiązań kontrolnych testowych plansz.
