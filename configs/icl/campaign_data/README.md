# Dane kampanii: publiczne stany i prywatni świadkowie

Ta konfiguracja dotyczy wyłącznie przygotowania danych offline; **nie wybiera
budżetu akcji i nie uruchamia eksperymentu live**. Limit całej kolejki, brak
osłony kosztowej i dalsze kroki opisuje `analysis/luna_icl/campaign_100/campaign.md`.

## Splity i izolacja

Publiczne manifesty znajdują się w `data/icl/campaign_dataset_v2/`:

- `demonstrations.json`: 8 realnych pełnych stanów, współdzielonych przez
  demonstracje oraz kalibrację; `training.json` zawiera dozwolone przykłady
  treningowe i indeksy kandydatów kalibracji.
- `<family>/component_validation.json`: po 4 realne i 4 syntetyczne plansze.
- `<family>/scale_validation.json`: osobne 4 + 4 plansze.
- `<family>/pilot100.json`: te same 50 + 50 pełnych plansz z poprzedniego pilota.
- `<family>/holdout1000.json`: osobne 500 + 500 plansz; przygotowane, lecz nie
  użyte do inferencji ani wyszukiwania.

Każdy publiczny rekord zawiera początkowy `state`, `board_id`, `family`,
`trajectory_id`, ścieżkę i hash źródła, indeks/seedy, wersję generatora,
`state_hash`, starszy `level_key` oraz `layout_id`. `layout_id` jest hashem
układu ścian i celów kanonizowanego względem czterech obrotów i odbić;
pozycje gracza i skrzynek nie tworzą nowych layoutów. `state_hash` obejmuje
pełny stan wraz z graczem i skrzynkami, bez kanonizacji orientacji. Easy
fixtures i stany pośrednie nie są uwzględniane jako instancje.

Prywatne świadki są w osobnym katalogu
`data/icl/campaign_dataset_v2_private/<split>/<family>.json`, poza drzewem
publicznych manifestów. Loader inferencji waliduje publiczny schemat, odrzuca
pola świadka/przyszłego stanu, sprawdza hash wybranego manifestu i zwraca tylko
`id`, `state`, `level_key`. Wczytanie pilota nie otwiera manifestu 1k, plików
prywatnych ani shardów. To rozdział interfejsu i ścieżek programu, nie bariera
systemu operacyjnego dla dowolnego kodu działającego pod tym samym użytkownikiem.

Dla **wyłącznie walidacyjnych diagnostyk** trzeba świadomie podać osobny
`--private-data-dir`. Komenda poniżej używa mocka, nie otwiera testowych splitów
i nie wywołuje API:

```bash
uv run --extra icl python scripts/evaluate_trajectory_components.py \
  --config configs/icl/campaign_data/component_validation_real.json \
  --component policy --split component_validation --indices 0 --horizons 4 \
  --positions start --max-requests 8 --mock \
  --private-data-dir data/icl/campaign_dataset_v2_private \
  --cache-dir /tmp/carl-campaign-validation-mock-cache \
  --output /tmp/carl-campaign-validation-mock
```

Konfiguracje `pilot_real.json` i `pilot_generated.json` umożliwiają wczytanie
po 50 pełnych plansz. Ich odziedziczony budżet jest **niewalidowany**; nie
uruchamiać ich live bez wyboru budżetu, zestawu u/d/fixed i wspólnego guardu $.

## Źródła i powtarzalność

Realne trajektorie pochodzą ze wszystkich 112 shardów
`rl-data/validation/sokoban/offline/12-12-4/`; wybór holdoutu wykorzystał
111 shardów. Indeks `data/icl/.private_campaign_shards.sqlite` zapisuje same
identyfikatory początkowego stanu, źródło i hash, bez rozwiązań. Indeks
wykrył 5 567 kanonicznych layoutów obecnych w więcej niż jednym shardzie.
Historyczne manifesty, notebooki i zapisane zapytania porównano za pomocą
`analysis/luna_icl/campaign_100/history_snapshot.json`; jawnie wykluczono także
layouty `trajectory_199`, `trajectory_434`, `trajectory_732` i ich symetrie.

Syntetyczny generator to `carl.icl.datasets.reverse_board-v1`, a hash kodu i
seedy są w `campaign_dataset_manifest.json`. Jest to syntetyczna dystrybucja
reverse-pull, a długość świadka nie oznacza optymalnej długości rozwiązania.
Losowy wybór realnej walidacji skal i realnego holdoutu ma seed `20261112`.

Weryfikacja istniejącego przygotowania, bez wyświetlania rozwiązań:

```bash
uv run --extra icl python scripts/index_campaign_shards.py --workers 4
uv run --extra icl python scripts/audit_campaign_history.py
uv run --extra icl python scripts/prepare_campaign_real.py
uv run --extra icl python scripts/assemble_campaign_dataset.py --check
uv run --extra icl python scripts/verify_campaign_dataset.py --workers 10
uv run --extra icl python scripts/report_campaign_data.py
```

Powtórzenie generatora w osobnych katalogach i porównanie hashy (obliczeniowo
kosztowne, nadal offline):

```bash
uv run --extra icl python scripts/prepare_campaign_synthetic.py --seed-base 2026120200 --count 500 --output /tmp/carl_campaign_synthetic_repro_20261102 --workers 8
uv run --extra icl python scripts/prepare_campaign_synthetic.py --seed-base 2026110200 --count 50 --output /tmp/carl_campaign_pilot_repro_20261102 --workers 5
uv run --extra icl python scripts/prepare_campaign_synthetic.py --seed-base 2026111500 --count 4 --output /tmp/carl_campaign_scale_repro_20261102 --workers 2
uv run --extra icl python scripts/prepare_campaign_synthetic.py --seed-base 20761029 --count 4 --output /tmp/carl_campaign_component_repro_20261102 --workers 2
uv run --extra icl python scripts/check_campaign_reproducibility.py
```

Tworzenie od zera wymaga też prywatnych stagingów syntetycznych:

```bash
uv run --extra icl python scripts/prepare_campaign_synthetic.py --seed-base 2026111500 --count 4 --output data/icl/.private_campaign_scale_staging --workers 2
uv run --extra icl python scripts/prepare_campaign_synthetic.py --seed-base 2026120200 --count 500 --output data/icl/.private_campaign_holdout_staging --workers 10
```

Skrypt składania bez `--check` nigdy nie nadpisuje istniejącego zbioru.
Odtwarzanie starej kohorty pilota wymaga też źródła `component_study_100_clean`
i skryptu `analysis/luna_icl/campaign_100/prepare_pilot.py` opisanego w kampanii.
Wyniki audytu agregatowego, bez pojedynczych rozwiązań:
`analysis/luna_icl/campaign_100/dataset_report.md`, `dataset_statistics.csv`,
`split_intersections.csv` i `data_integrity.json`.
