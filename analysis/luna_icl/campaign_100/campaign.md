# Kampania 100: stan i miejsce wznowienia

## Korekta budżetu — 2026-11-03

Użytkownik wyjaśnił, że węzły high-level są osiąganymi stanami ze ścieżki
low-level i nie należy ich dodawać drugi raz. Jedyny budżet searcha to liczba
wykonanych kroków low-level, B=500 domyślnie. Liczone są także no-op, nieudane
próby i primitive fallback. `max_expansions=null`: liczba ekspansji high-level
jest osobną statystyką, nie dodatkowym limitem ani składnikiem B. Limit CLLP L=8
oraz limity API/czasu/kosztu pozostają ograniczeniami wykonania, nie osią wykresu.
Oryginalny CaRL podobnie raportuje low-level nodes osobno; historyczny solver
mógł przekroczyć `max_nodes` przy dłuższym rolloutcie, tutaj B jest ścisłe.

Nowe entrypointy: `scripts.launch100 --budget 500 --plan/--prepare` i osobne
`--budget 128`, `--budget 256`. Każde B to niezależny run; raport łączy jedynie
zgodne zamrożone konfiguracje, nie wycina niższego B z większego śladu. Torch i
verbal używają tej samej jednostki B; Torch przyjmuje `--budget`.
Plan 500 ekspansji + 4000 akcji z poprzedniego wpisu jest nieaktualny i nie był
wykonywany. Jego manifesty zachowano w lokalnym `analysis/` jako historię.

Jeden wcześniejszy 32-krokowy probe kosztował $0.022443483; liniowa ilustracja
200 runów × 500 kroków przy stałym koszcie to około $70.14, z mieszanym cache
na jednej planszy. To nie jest wiarygodna górna granica ani wynik pilota. Limit
całej kampanii pozostaje $40. Nowych płatnych wywołań w tej korekcie nie było.


## Aktualizacja: launch100 i porządkowanie repo — 2026-11-03

Bieżące polecenie: przygotować uruchomienie 100 pełnych plansz i uporządkować
kod/commity. **Nie uruchomiono nowych płatnych eksperymentów ani inferencji Torch.**
Instrukcja: `LAUNCH_100.md`. Nowy protokół nie wznawia starego pilota na innym kodzie.

- Kohorta: 100 różnych real_12x12x4 z `campaign_dataset_12x12_v1`, publiczne
  stany początkowe w `configs/icl/checkpoints/launch100/assets/`. Bez 32×32 solvów.
- Budżet: 500 ekspansji kolejki, osobno najwyżej 4000 rzeczywistych akcji.
  Seed 11; CLLP L=8. To nie natywny historyczny licznik `max_nodes` CaRL.
- Core: relative_u vs absolute_d, 200 runów; extended: dodatkowo fixed_k oraz
  verbal_relative, 400 runów. Przygotowane manifesty w
  `analysis/luna_icl/launch100_nodes500/` i `analysis/luna_icl/launch100_extended/`.
  Wszystkie komórki są nadal missing. Extended jest alternatywnym pełnym presetem;
  nie importuje completion rekordów core z innego freeze, lecz korzysta ze wspólnego cache.
- Zamrożone wejścia obejmują 26 kalibrowanych przykładów + 6 demonstracji
  odkładania skrzyń. Dla u/d kolejność i plansze są wspólne. Skale .85/.5 i 8/4
  przeniesiono z wcześniejszej walidacji; nie strojono ich po obejrzeniu nowych testów.
- Osobny `scripts.trained_reference.run` planuje 100 tych samych plansz,
  lokalne BART/BERT, wspólny search, skale 8/4, beam4, jedna propozycja.
  To nie historyczny natywny k841 (91/100 na innej kohorcie).
- Prognozy z jednego mieszanego-cache runu walidacyjnego: około $249 przy
  stałym koszcie na ekspansję albo $561 przy stałym koszcie na akcję dla core.
  Nie są zmierzonym kosztem pilota ani obietnicą. Globalny limit nadal $40,
  dostępne środki według ledger $34.748949284; $1.7392584 rezerwacji nieznanych
  pozostaje oddzielone od potwierdzonych opłat. Limit/stage może zatrzymać macierz.
- Default plan, prepare, status i report są offline. Live wymaga `--execute
  --usd-cap 40`; 40 oznacza absolutny limit istniejącej kampanii, nie nowe środki.
  Wznowienie sprawdza response/action journal oraz replay sukcesów. Retry tylko
  jawne HTTP429/5xx, jedna dodatkowa próba, osobny cache slot i rezerwacja.
- Top-level `scripts/`: 66→15 plików Python. 34 historyczne runnery są w
  `scripts/archive/campaign_20261102/`, narzędzia danych i treningu w osobnych
  pakietach. README archiwum mapuje stare nazwy; historyczne wpisy poniżej i
  zapisane źródła wyników zachowują oryginalne nazwy/hashes.
- Dane, cache, ledger, prywatni świadkowie i wagi pozostają lokalnie. Małe
  publiczne wejścia i prompty są wersjonowane. Notebooki mają czyste źródła;
  wcześniejsze wykonane kopie zachowano w `analysis/notebook_executions/`.

Kontrola i dokładne manifesty: `analysis/luna_icl/launch100_checks/`.
Eksport CSV/HTML/SVG/PDF: `uv run --extra icl python -m scripts.launch100.report --static`.
Powtórny eksport zachowuje agregaty; NULL oznacza brak obserwacji.


## Aktualizacja: demonstracje odkładania skrzyń i cele słowne — 2026-11-03

Zakończono mały `goal_progress_probe`; [raport i CSV](goal_progress_probe/report/README.md).
Wszystkie nowe opłaty: **$0.081064333**, 159 rozliczonych prób, bez nowych
nieznanych rezerwacji. Poprzednia rezerwacja $0.5791896 pozostaje osobna.
Dużego pilota, 50 pełnych solvów ani holdoutu nie uruchamiano.

- Sześć zweryfikowanych demonstracji z sześciu treningowych trajektorii
  pokazuje wzrost liczby skrzyń na celach. Mają długości 1, 2, 4, 8, 3 i 6.
  Nie dopisano im fikcyjnego u/d; 26 wcześniejszych skalibrowanych przykładów
  i dokładna CLLP pozostają niezmienione. Prompt + dodatkowe przykłady to
  wspólna interwencja, nie osobno zmierzone efekty.
- Sześć innych stanów walidacyjnych wybrano celowo blisko znanego odkładania
  skrzyni. Stary prompt: exact hit 4/6, trafiony cel z postępem zadania 0/6.
  Nowy: exact hit 5/6, trafiony cel z dodatkową skrzynią na celu **4/6**.
  To diagnostyka komponentów, nie sześć pełnych rozwiązań.
- Jeden dodatkowy pełny start walidacyjny: B=32, 9 ekspansji, 32 wykonane
  akcje, zachowana gałąź z **2/4 skrzyń na celach** po siedmiu ruchach ścieżki.
  Cała plansza pozostała nierozwiązana. Replay wszystkich 32 akcji poprawny.
  Historyczny B=128 to osobny run; nie jest równym budżetowo kontrastem.
- Wariant słowny ma własne demonstracje policy/judge i własną kalibrację:
  trzy konteksty treningowe × dwa cele, R=1, L=8, 6/6 deklarowanych trafień.
  Czasy par: (1,2), (3,4), (5,8); midpointy u=(.25,.75). To bardzo mała,
  zależna od modelowego sędziego kalibracja. Oba cele z kontekstu są zachowane.
- Cztery zapytania słowne: 2/4 osiągniętych celów i 2/4 przypadków z dodatkową
  skrzynią na celu. Oddzielny post-hoc test dwóch porażek z lepszym opisem
  współrzędnych odzyskał 2/2, po 3 i 4 akcjach; nie jest nowym testem czterech
  plansz. Policy i judge pozostały niezmienione.
- Audyt po fakcie: początkowe 46/46 ocen sędziego zgodne z literalnymi celami;
  w dwóch retry 6/7, jeden false negative opóźnił stop o ruch. Brak false
  positives w tej próbce. Audyt nie dostarcza modelom ruchów ani nie zmienia
  ich decyzji. Brakujące `progress` w raw oznaczono w audycie NULL, a stare
  zera w zapisie kroków opisano jako placeholder dekodera.

Kod/config: `scripts/goal_probe/`, `configs/icl/prompts/generator_goal_progress.json`,
`scripts/verbal_icl/probe.py`, `configs/icl/verbal/tiny.json`,
`configs/icl/verbal/grounded_generator.json`. Sześć dodatkowych przykładów
pełnych plansz wstawia `TaskProgressComponents`; sam ogólny CLI z plikiem
promptu nie wstawi tych przykładów. Używać zmierzonego runnera `goal_probe`.
Jawne checkpointy/payloady: `goal_progress_probe/explicit/openai/manifest.json`,
`goal_progress_probe/verbal/3907728c160d892bf5763a7054b9729c05244562491854905d5b984ff6a2091a/checkpoint.json`
i `goal_progress_probe/verbal_grounded_retry/7bfd933812e61e587cf83225d94519513c2de4cc1ce16c9f761e837f7bc709d4/checkpoint.json`.

Bez API: `uv run --extra icl python -m scripts.goal_probe.report` odbudowuje
CSV oraz HTML/SVG/PDF; `notebooks/luna_goal_progress.ipynb` pokazuje przykłady,
cele słowne, akcje i powody decyzji sędziego. `scripts.goal_probe.full_probe`
bez flag pokazuje plan. Wszystkie ukończone wyniki i raw calls zachowano.

## Aktualny zakres — 2026-11-03

Poniższa historia zachowuje poprzednie decyzje. Obowiązuje późniejsza dyspozycja:
**mała walidacja komponentów; duży pilot i holdout pozostają zatrzymane**.
Nie wznawiać historycznej macierzy na zmienionym kodzie. Ostatnie wyniki,
pełne odpowiedzi, koszt i ograniczenia: [generator_probe_v2/report/README.md](generator_probe_v2/report/README.md).

- Generator: 30 kandydatów z 10 treningowych trajektorii `rl-data`, po trzy
  różne cele na kontekst; 26 zachowanych przykładów po kalibracji R=2, L=8.
  Zapisany checkpoint: `generator_probe_v2/checkpoint_balanced/generator.json`.
  Sąsiadujące `README.md`, `config.json` i `generator_inputs.json` zawierają
  jawne plansze, etykiety, kolejność demonstracji i pełne wiadomości.
- Walidacja: 30 stanów na **10 innych layoutach**, po trzy stany na layout;
  to nie 30 niezależnych plansz ani pełnych rozwiązań. Pojedyncza próba
  generatora z 4 i 26 przykładami dała po **23/30 exact hits** CLLP.
  Każdy wariant zawierał jedną dosłowną kopię źródła; odrzucono je.
- Jawna druga próba generatora odzyskała **7/7** statycznie odrzuconych
  przypadków wariantu 26-shot. Pierwsze odpowiedzi odtworzono z cache:
  wynik 30/30 dotyczy tej samej obejrzanej kohorty z maksymalnie dwiema
  próbami, nie niezależnego retestu. Feedback zawiera wyłącznie przyczynę
  statycznego odrzucenia. Symulator nie dostarcza generatorowi ruchów.
- Większość osiągniętych subgoali dzieli od źródła 1–2 ruchy. Wykonalność
  endpointów poprawiła się, ale sterowanie odległością przez u pozostaje
  niepotwierdzone. Więcej przykładów samo w sobie nie poprawiło 23/30.
- Osobny, pojedynczy pełny solve walidacyjny B=128 zapisuje trwały ślad w
  `generator_probe_v2/solver_validation/`. O zakończeniu świadczy `case.json`,
  nie sam cache ani trace. Przed uruchomieniem sprawdzić proces; runner
  odmawia ponownego wykonania przerwanego śladu. Końcowe statystyki raportuje
  powyższy raport. Konfiguracja: `configs/icl/experiments/luna_checkpoint26_retry2.json`.
  **Zakończony:** 0/1 rozwiązań, stop `action_budget`, 128 akcji, 38 ekspansji,
  26/36 osiągniętych subgoali, 0 kopii źródła, 7 statycznie błędnych propozycji.
  Replay potwierdził cały ślad, nie ścieżkę rozwiązania (tej nie znaleziono).
  139 nowych żądań providera, 56 lokalnych trafień cache, koszt $0.094439696.
  Zweryfikowane 57 wejść i źródeł zachowano w `solver_validation/frozen_inputs.zip`
  przed dalszymi poprawkami stylu; stary manifest nie opisuje zmienionego kodu.
- Natywny trenowany AdaSubS k=8,4,1: losowe 100 layoutów wybrano przed
  odczytem wyników; odzyskano **91/100 historycznych sukcesów** i odtworzono
  wszystkie 91 ścieżek w CaRL. To nie nowe wykonanie checkpointów ani
  sparowany kontrast z Luną. Szczegóły i ograniczenia historycznego capu:
  `analysis/trained_reference/random100_k841/report.md`.
- Koszty nowych prób dzieli globalny ledger, etap `reserve` ma limit $1.50.
  Koszt po przerwanym połączeniu pozostaje nieznany i zarezerwowany
  ($0.5791896); nie przedstawiać go jako potwierdzonej opłaty.
  Aktualne rozliczenie per faza: `generator_probe_v2/report/phase_costs.csv`.
  Terra nie była wywoływana; wszystkie te pomiary modelu dotyczą Luny.
  Finalne nowe opłaty całej tej serii: **$0.197832468**, 347 rozliczonych prób.
  Jedna nieznana rezerwacja jest osobna; brak aktywnych rezerwacji po zakończeniu.

Bezpieczne polecenia offline z katalogu repo:

```bash
uv run --extra icl python -m scripts.generator_probe.solver_probe
uv run --extra icl python -m scripts.generator_probe.evaluate
uv run --extra icl python -m scripts.trained_reference.freeze
```

Pierwsze polecenie pokazuje plan bez API, drugie odtwarza raport ze zapisów,
trzecie weryfikuje historyczny zbiór referencyjny. Notebook
`notebooks/luna_checkpoint_inspection.ipynb` pozwala obejrzeć checkpoint,
konkretne plansze demonstracyjne, wiadomości i zapisane wyniki bez API.

Po refaktorze sprawdzono ponowne wykonanie całego solva przez główny CLI:
128 akcji, 38 ekspansji, 195 lokalnych trafień cache, **0 wywołań providera**.
Polecenie odtworzenia (offline, brakujący cache kończy się błędem):

```bash
uv run --extra icl python -m scripts.run_icl_search run --config configs/icl/experiments/luna_checkpoint26_retry2.json --replay --output analysis/luna_icl/campaign_100/generator_probe_v2/post_refactor_replay
```

Kontrola końcowa: 428 testów; Ruff, Pyright (w tym 6 notebooków), mypy;
Pylint 10.00/10 bez ostrzeżeń. `carl/icl` ma 48 plików Python, maksymalnie
341 wierszy na plik, bez `from __future__ import annotations`. Wszystkie
60 zapisanych payloadów generatora i fingerprint CLLP są niezmienione:
`generator_probe_v2/source_refactor_verification.json`. Logi: `final_lint.log`,
`final_tests.log`, `final_pylint.log` w katalogu `generator_probe_v2`.

## Historia przygotowania i poprzednich przebiegów

Stan 2026-11-02, inwentaryzacja i przygotowanie danych offline. Checkout `/home/mtyrolski/phd/2026/CaRL`,
branch `feature/openai-icl-search`, HEAD `4fefc81a810a6f469c985a542c426ad65c1e7b20`.
Drzewo jest mocno niezacommitowane (m.in. `carl/icl`, konfiguracje, notebooki,
skrypty, `analysis/`); zachować wszystkie zmiany, cache i `.env`. Nie znaleziono
`AGENTS.md` na ścieżce do repo ani w `analysis/luna_icl`. Snapshot identyfikujący
kohortę: `pilot_spec.json` oraz `data/icl/campaign_100_pilot/campaign_manifest.json`.

## Pytanie i zasady

Czy generator Luny otrzymujący lokalny percentyl `u` rozwiązuje więcej pełnych
plansz w zamrożonym budżecie akcji niż ten sam generator otrzymujący średni
skuteczny czas pierwszego trafienia `d`? To **explicit-state ICL** z §4.3
propozycji, nie trenowany encoder ani latentny generator/MSE. Generator, CLLP
oraz V: `gpt-6-luna` przez istniejący klient; rzeczywiste akcje wykonuje CaRL.
Bez ukrytych osobnych prób w symulatorze do weryfikacji generatora/CLLP; kontrola
struktury planszy i normalne roll-outy wyszukiwania są jawne. Mock nie jest wynikiem Luny.

Główny kontrast to **relative u − absolute d**, na tych samych planszach,
kandydatach kalibracji, ważeniu/przykładach, zamrożonych CLLP/V, limicie L,
seedzie i budżetach. `fixed-k` pozostaje trzecim, osobno opisanym baseline'em
AdaSubS-style: jego cap=k zmienia również regułę wykonania, więc nie jest samym
przeetykietowaniem u→d. Wynik ujemny lub brak rozwiązania jest poprawnym wynikiem;
raportować błędy, brakujące przypadki i fizyczne koszty, bez retuszowania prób.
Poziomy skali i **główny budżet akcji nie są jeszcze wybrane**. Wybrać je tylko
na rozłącznej walidacji i zamrozić manifest przed pierwszym testem pilota.

## Odzyskane wyniki i aktualne entrypointy

- `trajectory_tuning_20261102/explore.py` nie istnieje w pierwotnym miejscu;
  skrypt jest w `trajectory_tuning_20261102/archive/explore.py`. Nie uruchamiać
  go ponownie: używa historycznych dodatkowych prób symulatora i może płacić.
  `exploration.json` zapisano 14:57:44 CET; żadnego procesu explore obecnie
  nie ma. Historia procesów/exit code nie jest dostępna, więc nie dowodzi to
  sposobu zakończenia. Wszystkie 40/40 komórek mają wiersze: policy 12, generator
  8, value 20; 36 bez błędu i cztery błędy value. Validation: 2 plansze,
  `gpt-6-luna` / OpenAI (źródło skryptu i cache; raport nie ma pola provider).
  `low_reasoning` CLLP 3/4, `coordinates` 0/4, `simulator_successors` 0/4;
  generator baseline 0/4 i low-reasoning 1/4 strukturalnie poprawnych;
  value baseline 6/10 poprawnych, trajectory-examples 10/10. To mała eksploracja,
  nie pomiar solve-rate. Wariant successor/audit nie jest obecną metodą.
- To wywołanie: 71 fizycznych żądań, 5 trafień lokalnego cache; 144 282 input,
  8 166 output, 2 512 cached input i 133 611 cache-write tokenów. Ze starej
  tabeli cen w `cost_summary.json` ≈ **$0.0216254**, nie faktura. Raport
  całego tuning `cost_summary.json`: 519 prób, ≈ $0.192656, zawiera te 71;
  nie dodawać ich drugi raz. Osobny `debug_live_20261102/README.md`: 31 prób,
  ≈ $0.00621 (bez wcześniejszego trzywywołaniowego smoke). Faktyczna faktura,
  aktualna cena, błędy z nieznanym kosztem i niższy limit konta są offline
  nieznane. Historyczne $0.198866 nie jest wynikiem nowej kolejki.
- `debug_live_20261102/full_solver.json`: OpenAI, jedna plansza, nie rozwiązano,
  limit czterech akcji, jeden odrzucony cel; to diagnostyka, nie benchmark.
  `final_simple/`: na pięciu nowych planszach CLLP 4/5 dla k=4, 3/5 dla k=8;
  generator poprawny 5/5, rzeczywiście trafiony przez CLLP 1/5; value MAE
  względem świadków ≈0.0683. Próbka mała, świadek nie musi być optymalny.
- Zamiast archiwum używać `scripts/evaluate_trajectory_components.py` (jawne
  diagnostyki), `scripts/evaluate_component_study.py` i
  `notebooks/luna_component_study.ipynb` (0/2/4 examples), a dla solvera
  `scripts/run_icl_search.py`, `scripts/run_budget_sweep.py`,
  `carl/icl/{calibration,luna,search}.py`. Historyczny cache `trajectory_tuning_20261102/cache`
  oraz `debug_live_20261102/cache` zachować; nowszy klient używa pełnego payloadu
  jako klucza. Wyniki i starsze odpowiedzi można odzyskać przez dokładny replay,
  ale zmienione prompty/schema/streaming dają cache miss; stary zapis może nie
  zawierać pełnego obiektu odpowiedzi. Nie mieszać wpisów mock i openai.

## Dane, budżet i zabezpieczenia

Pilot: `data/icl/campaign_dataset_v2`, split `pilot100`, 50 `real_12x12x4` + 50
`generated_32x32x10`, seed solvera **11**, pełne plansze. Pierwsze 50 rzeczywistych
z `component_study_100_clean` (wyklucza trening, walidację, stary test i trzy
poprzednie holdouty); 50 syntetycznych z `reverse_board` seedami
2026110200…2026110249, rozłącznych z wcześniejszymi wygenerowanymi layoutami.
To nowa syntetyczna dystrybucja, nie odtworzenie starego generatora; świadkowie
są poprawni, ale nie dowodzą optymalności. W 553 lokalnych plikach cache pod
`analysis/luna_icl/**/openai` nie znaleziono zapytania live z żadnym ze 100
layoutów pilota. To nie dowodzi braku użycia poza tymi zapisami.
Oryginalny manifest kohorty `data/icl/campaign_100_pilot/campaign_manifest.json`
ma SHA-256 `b80ecd9ece212ba67632069273f925e4a00638e2deea8d2ac2e973e8f53711d3`;
nowe publiczne manifesty zawierają te same 100 pełnych stanów, z dodatkowymi
identyfikatorami i kanonizacją symetrii. Ich hashe zawiera `pilot_spec.json`.
Generowanie i walidacja świadków były lokalne, nie przeprowadzono inferencji.
Loader zweryfikował obie rodziny po 50 przykładów, kształty 12×12/32×32,
liczbę skrzynek 4/10, 100 różnych hashy layoutów i wszystkie 100 świadków.

**1k:** zgodnie z późniejszym zleceniem przygotowano oddzielny split
`holdout1000`: 500 realnych i 500 syntetycznych pełnych stanów; **nie wykonano
na nim inferencji ani nie uruchomiono solvera**. Selekcja realnych używa seeda
20261112 i obejmuje 111 shardów; syntetyczne używają seedów 2026120200…2026120699.
Wykluczono wcześniejsze zaobserwowane layouty, symetrie, zarezerwowane manifesty,
trening, obie walidacje i pilota. Wszystkie 112 shardów przeindeksowano:
112 000 stanów początkowych, 106 426 kanonicznych layoutów, 5 567 layoutów
powtórzonych między shardami. Osobny raport liczebności i długości świadków:
`dataset_report.md`, pełna tabela przecięć: `split_intersections.csv`.

Publiczne manifesty: `data/icl/campaign_dataset_v2`; prywatne świadki:
`data/icl/campaign_dataset_v2_private` (osobny katalog). Demonstracje i
kalibracja współdzielą 8 kontekstów; walidacja komponentów i walidacja skal są
osobnymi splitami po 4 realne i 4 syntetyczne plansze. Wszystkie 36 par grup
ma zerowe przecięcie kanonicznych layoutów. Znane długości świadków nie są
minimalnymi długościami rozwiązań. Szczegóły i offline odtworzenie:
`configs/icl/campaign_data/README.md`.
Ponowne przygotowanie odtworzyło te same 9 publicznych i 9 prywatnych artefaktów;
niezależnie odtworzono syntetyczne stany i hashe: 500/500 holdout, 50/50 pilot,
4/4 walidacja skal i 4/4 walidacja komponentów (`reproducibility.json`).
Stare numery `trajectory_199`, `trajectory_434`, `trajectory_732` oraz wszystkie
ich obroty/odbicia są nieobecne w obu testowych kohortach. Zmiany loadera
zapewniają odrzucenie prywatnych pól manifestu i brak odczytu 1k/świadków przez
ścieżkę inferencji pilota. Przygotowanie danych nie rozwiązuje blokady kosztowej 02.
Niezależny końcowy audyt odtworzył 1124 świadków i 405651 akcji: wszystkie
przejścia poprawne, wszystkie końce spełniają `is_solved`, a stany rzeczywiste
zgadzają się z pełnymi stanami początkowymi z indeksowanych shardów; sumy
kontrolne prywatnych plików znajdują się w `witness_verification.json`.

Kolejka 01–09: **maksymalnie $40 nowych łącznie**, również testy live,
kalibracja, retry, wszystkie komponenty i przerwane próby; obowiązuje niższy
limit konta lub właściwy limit konfiguracji. Krok 01: **$0 nowych opłat**.
Nie znamy limitu konta ani rzeczywistego rachunku offline. Repo ma ponadto
`configs/icl/sokoban_openai_smoke.json` z $0.20 na osobny legacy smoke
`gpt-5.4-mini`; nie podwyższać go ani nie stosować jego cen do Luny. Od kroku walidacji policy klient używa trwałego wspólnego ledgeru i atomowych
rezerwacji; szczegóły oraz bieżący status podano w końcowej sekcji tego pliku.
Limit konta nie jest dostępny przez klienta: jawnie niższy limit można przekazać
przez `CARL_CAMPAIGN_MAX_USD`, a usługa może odmówić próby.

## Kolejka 01–09 i wznowienie

Polecenia poniżej uruchamiać z katalogu CaRL. `--plan`, `preview`, odczyt i testy
są offline; historyczne plany poniżej pozostają punktami odniesienia. Aktualny
status osłony i walidacji policy znajduje się w końcowej sekcji. Gdy wymagana konfiguracja jeszcze nie istnieje,
wskazano istniejące polecenie inspekcyjne, nie udajemy że uruchamia docelową kampanię.

| Krok | Stan | Dalsze działanie / konkretne polecenie wznowienia |
|---|---|---|
| 01 | gotowe offline | Sprawdź `cat analysis/luna_icl/campaign_100/campaign.md`; w razie przerwania przygotowania przed powstaniem docelowego katalogu: `uv run --extra icl python analysis/luna_icl/campaign_100/prepare_pilot.py` (wykorzysta staging; nigdy nie nadpisuje istniejącego celu). |
| 02 | **w toku: guard i walidacja policy** | Zacznij od `uv run --extra icl python -m scripts.run_budget_sweep --config configs/icl/budget_sweeps/tuned_quick.json --plan`. To tylko inspekcja starej pary relative/fixed; następnie zaimplementuj wspólny guard $/ledger, u-vs-d konfiguracje, testy i mapowanie raportów. |
| 03 | oczekuje 02 | Przygotuj dopasowaną kalibrację policy przed porównaniem; inspekcja bez API: `uv run --extra icl python -m scripts.run_icl_search preview --config configs/icl/experiments/tuned_relative.json`. Obecne dwie kalibracje są `mock`, nie wolno ich użyć jako live. Płatna komenda i config mają powstać w 02 pod osłoną guardu. |
| 04 | oczekuje 03 | Na walidacji wybierz/zamroź L, skale i budżet z równą liczbą prób; inspekcja istniejących dostępnych budżetów: `uv run --extra icl python -m scripts.run_budget_sweep --config configs/icl/budget_sweeps/tuned_quick.json --plan`. Uwaga: istniejący tuner wybiera tylko relative/fixed, nie u/d. |
| 05 | oczekuje 04 | Sparowany pilot u-vs-d na zamrożonych 50+50, seed 11, następnie osobny fixed-k; rozpocząć dopiero przez guard i nową konfigurację wyprodukowaną w 02. Offline kontrola wejścia bez ujawniania plansz: `uv run --extra icl python scripts/verify_campaign_dataset.py --identity-only`. |
| 06 | oczekuje 05 | Policz solve-rate i parowane różnice u−d osobno dla rodzin, wraz z błędami/coverage. Istniejący reporter porównuje tylko relative−fixed; najpierw 02 musi dodać u−d. Wejście/plan: `cat analysis/luna_icl/campaign_100/pilot_spec.json`. |
| 07 | oczekuje 05 | Raportuj fixed-k oddzielnie, w tym różnicę limitu cap=k vs L; inspekcja jego profilu: `cat configs/icl/experiments/tuned_fixed.json`. Nie mieszać go z kontrastem label-only. |
| 08 | oczekuje 06–07 | Offline kontrola integralności artefaktów i regresji: `make test`; dodatkowo sprawdzić rozwiązania, paired keys, tokeny, cache i limit kolejki. |
| 09 | oczekuje 08 | Zamknij raport z ujemnym lub dodatnim wynikiem, stanem budżetu i tym co niewykonane; wznowienie dokumentacji: `cat analysis/luna_icl/campaign_100/campaign.md`. 1k pozostaje osobno. |

### Konkretny problem dla 02

`carl/icl/budget_sweep.py:load_sweep` dopuszcza tylko parę `relative`+`fixed`,
`_manifest` uwzględnia calibration hash tylko dla `relative` (a `absolute`
także jej wymaga), `carl/icl/experiment.py:tune` tunuje tylko relative/fixed,
a `carl/icl/reporting.py` liczy tylko relative−fixed. Potrzebne są sparowane
warunki u/d/fixed, walidacyjny wybór budżetu, wspólny kosztowy stop oraz testy
proweniencji i raportowania; sam istniejący `--plan` nie rozwiązuje tych braków.

## Walidacja policy i wspólny licznik — aktualizacja 2026-11-02

**Ukończono porównanie CLLP live.** Dawne cztery realne plansze
`component_validation` były już wykorzystane w debugowaniu, więc utworzono
rozłączne uzupełnienie `data/icl/campaign_policy_validation_v1`: 4 nowe
realne layouty z 48 niewykorzystanych, uprzednio odtworzonych kandydatów.
Wybór uwzględnia historyczne layouty, trening, obie walidacje, pilot i 1k.
Świadkowie pozostają w `data/icl/campaign_policy_validation_v1_private`.
Dla każdej planszy wybrano jeden wspólny kontekst i cztery różne cele
`k=1,2,4,8`; fragmenty obejmują chodzenie, pchanie i co najmniej jedno
obejście nieruchomej skrzyni (odległość przejścia ze skrzyniami jako
przeszkodami jest większa od odległości bez nich). To nie jest dowód,
że każda możliwa trasa Sokobana musi mieć ten kształt. Odległość świadka
nie oznacza optymalnej drogi.

Produkcja: `search.rollout`, limit **L=8**, `gpt-6-luna`, cztery te same
przykłady treningowe, kategoryczne próbkowanie i sparowany seed dla pary.
Wszystkie warianty miały `policy_tokens=2048`, aby low reasoning mógł
użyć tokenów rozumowania; baseline i coordinates poza tym odpowiadają
`fixed.json` i `fixed_policy_coordinates.json`. Porównanie na czterech
planszach, po jednym rolloutcie na parę (błędy: 0/48):

| Wariant | k=1 | k=2 | k=4 | k=8 | Średnia z plansz i offsetów |
|---|---:|---:|---:|---:|---:|
| baseline | 3/4 | 1/4 | 0/4 | 0/4 | 4/16 = 25% |
| coordinates | 4/4 | 2/4 | 0/4 | 0/4 | 6/16 = 37,5% |
| coordinates + low | 4/4 | 4/4 | 4/4 | 2/4 | 14/16 = 87,5% |

Zamrożono `coordinates_low`: `configs/icl/campaign_data/frozen_policy_fixed.json`,
oraz warianty `frozen_policy_relative.json` i `frozen_policy_absolute.json`.
Te ostatnie zawierają **prowizoryczne skale**, a nie wybrany budżet pilota.
Wszystkie trzy mają ten sam wymuszany fingerprint policy i L; zmiana modelu,
promptu, demonstracji, temperatury, formatu lub cap skutkuje błędem przed
inferencją. Pełna tabela, przykład porażki, wykres i kompletne ślady:
`policy_validation/{report.md,exact_hit.html,report.json,selection.json}`.
Licznik akcji obejmuje no-op; first hit i wcześniejsze rozwiązanie zadania
są odrębne. Historyczna eksploracja jest opisana wyżej i nie została
ponownie uruchomiona.

Dodatkowy wariant `simulator_successors` używa **czterech dodatkowych przejść
symulatora na decyzję** i nie uczestniczy w wyborze policy. Wyniki odpowiednio
4/4, 2/4, 2/4, 0/4; szczegółowe liczniki prób w
`policy_validation/simulator_successors.json`. W obecnej diagnostyce
wzbogacono tylko bieżące zapytanie; historyczny explore wzbogacał również
pytania demonstracyjne, więc nie jest to identyczny prompt.

Przed live callami wdrożono `carl/icl/campaign_budget.py` i wspólny
`analysis/luna_icl/campaign_100/cost_ledger.sqlite3`. Ceny/model/limity są w
`configs/icl/campaign_budget.json`; ceny wg [OpenAI](https://developers.openai.com/api/docs/models/gpt-6-luna),
2026-11-02, z konserwatywnym 10% regionalnym narzutem. SQLite transakcyjnie
rezerwuje koszt przed fizycznym wysłaniem; usage, odczyt i zapis cache są
rozliczane osobno. Gdy usage jest niedostępny, rezerwacja pozostaje. Cena
rzeczywista konta i dostępny limit konta nie są dostępne z tego ledgeru;
kwoty to konserwatywny szacunek, nie faktura. Opcjonalny niższy limit całej
kampanii `CARL_CAMPAIGN_MAX_USD` jest zapamiętywany trwale, a każdy proces
musi wybrać `CARL_CAMPAIGN_STAGE`; domyślny `unassigned` ma limit $0.
Sublimity: policy $2, calibration $4, scale $4, trzy części pilota $12/$12/$4,
reporting $0,50 i rezerwa $1,50; suma $40. Nieznane, przerwane i równoczesne
próby również zużywają przydział.

Stan po walidacji: 330 logicznych zapytań, 165 trafień lokalnego cache,
164 zakończone płatne próby z usage i jedna nieznana; 285162 input, 13083 output,
253316 cache-write oraz 0 cache-read tokenów w usage zakończonych prób.
Rozliczony konserwatywnie koszt tych 164 prób: **$0,045529694**. Pierwszy
HTTP 401 wynikał z odziedziczonego klucza różnego od `.env`; jego koszt
nie jest znany i zachowano **$0,579189600 rezerwacji**. Razem zajęte
**$0,624719294**, bez aktywnego requestu. Zmiana na repozytoryjny `.env`
przez `env -u OPENAI_API_KEY` rozwiązała 401; nie modyfikowano kluczy.
Poprzednie koszty z historycznej inwentaryzacji nie są nowymi kosztami tej
kolejki. Status na bieżąco:

```bash
uv run --extra icl python -m scripts.campaign_budget_status
```

Wznowienie walidacji, jeżeli raport zostanie przerwany (już ukończone pary
są pomijane; może płacić):

```bash
env -u OPENAI_API_KEY uv run --extra icl python -m scripts.campaign_policy_validation --provider openai
env -u OPENAI_API_KEY uv run --extra icl python -m scripts.campaign_policy_successors --provider openai
uv run --extra icl python -m scripts.freeze_campaign_policy
uv run --extra icl python -m scripts.report_campaign_policy
```

**Następny krok 03:** przygotować i wykonać kalibrację na treningowych
kontekstach, zachowując fingerprint i jawnie wskazując stage. Punkt wejścia,
gdy kalibracja zostanie świadomie uruchomiona (powoduje koszty):

```bash
env -u OPENAI_API_KEY CARL_CAMPAIGN_STAGE=calibration uv run --extra icl python -m scripts.run_icl_search calibrate --config configs/icl/campaign_data/frozen_policy_relative.json --execute
```

Kalibracji i wyboru skal/budżetu pilota jeszcze nie wykonano, a u-vs-d sweep
wymaga brakujących zmian opisanych w kolejce wyżej. Ani pilot, ani 1k nie
zostały uruchomione. Testy regresji, Ruff i Pyright dla zmienionych
modułów przeszły; mock jest wyłącznie testem infrastruktury.

## Generator i V: walidacja przed kalibracją — 2026-11-02

Wykonano wyłącznie inferencję na komponentowych splitach walidacyjnych;
pilot ani holdout nie były użyte. `scripts/campaign_components_validation.py`
i `scripts/campaign_value_current.py` zapisują osobny plik dla każdej pary
oraz hashują konfigurację, kod i źródła. W razie przerwania wznawiają od
brakujących pozycji; zapisane błędy nie są automatycznie powtarzane. Żądania
rozliczono w istniejącym subbudżecie `scale_validation` (tagi generator/value),
więc ta kwota zmniejsza środki pozostałe na wybór skali. Modele miały dostęp
wyłącznie do przykładów z train i aktualnego stanu. Świadkowie walidacji
służyli do konstrukcji wejść i metryk offline.

Generator porównano na tych samych 8 żądaniach (4 nowe realne plansze,
k=4 i k=8, po maksymalnie dwie propozycje). Baseline to tekstowy kontrakt
bez reasoning, a `full_rows` zmienia kontrakt na JSON schema z kompletnymi
wierszami i wzorcem zachowującym ściany i cele. Obie wersje otrzymały cztery
przykłady treningowe: dla k=4 trzy zmieniają skrzynie, wszystkie zmieniają
gracza; dla k=8 wszystkie zmieniają zarówno gracza, jak i skrzynie. Jedna
próba generatora na żądanie; bez filtrowania mianownika, cap zamrożonej CLLP
L=8. **Baseline 0/8** żądań z osiągniętym, odmiennym subgoalem, `full_rows`
**4/8** (k=4: 3/4; k=8: 1/4). Baseline zwrócił 5 kopii źródła i jedną
statycznie niepoprawną propozycję; `full_rows` trzy kopie, jedną niepoprawną
i cztery odmienne. Wszystkie cztery odmienne cele zostały osiągnięte przez
CLLP i mają świadków BFS w L=8. Bounded BFS miał limit 10000 stanów i zwraca
trzy odrębne stany: `reachable`, `unreachable_within_cap`, `node_limit`.
Jest diagnostyką offline; jego świadkowie nie trafiają do żadnego promptu.
Wynik 4/8 i pojedyncze plansze nie przesądzają o solve rate. Wybór:
**full_rows**, mimo pozostałych kopii i błędnej planszy.

V: osiem plansz, cztery realne 12×12×4 i cztery syntetyczne 32×32×10; dla
każdej odtworzono pełnego świadka i oceniono początek, 1/4, 1/2, 3/4 i
koniec. `initial_examples` i `trajectory_examples` mają ten sam prompt
bezpośredniego zwrotu i różnią się jedynie przykładami z train. Dodatkowo
sprawdzono istniejący `current_frozen`: `remaining_actions`, cechy planszy,
JSON schema i przykłady trajektoryjne. Ta trzecia konfiguracja różni się
wieloma elementami, więc porównanie z nią nie jest izolowaną ablacją.
Referencja to `gamma ** remaining` na danej poprawnej kontynuacji, która nie
musi być najkrótsza. Wybór oparto na średniej po planszach poprawnych par
nieterminalnych, przy czym remisy i brakujące odpowiedzi pozostają w
mianowniku; MAE rozstrzyga remis. Jedna próbka modelu na stan i wariant.

| V | poprawne / 48 par nieterminalnych | remisy | brakujące pary | Spearman nieterminalny: średnia (plansze) | płaskie nieterminalne | MAE | poprawne odpowiedzi |
|---|---:|---:|---:|---:|---:|---:|---:|
| initial_examples | 12/48 | 13 | 13 | 0,086 (7/8) | 1 | 0,185 | 30/40 |
| trajectory_examples | 15/48 | 11 | 8 | 0,149 (8/8) | 0 | 0,266 | 37/40 |
| current_frozen | 44/48 | 0 | 0 | 0,900 (8/8) | 0 | 0,141 | 40/40 |

Niektóre tekstowe odpowiedzi miały uszkodzony JSON lub przerwany strumień;
pozostały błędami, bez retry pod poprawę wyniku. Sam stan terminalny nie
uczestniczy w podstawowej metryce; pełne liczniki obejmujące także terminal
są w raporcie maszynowym. Wynik jest dla ośmiu trajektorii, nie dowodzi
jakości na pilocie, a syntetyczne kontynuacje są znacznie dłuższe od części
przykładów treningowych.

Wspólne zamrożone konfiguracje wszystkich metod to teraz
`configs/icl/campaign_data/frozen_components_{fixed,relative,absolute}.json`;
**zastępują** wcześniejsze `frozen_policy_*` dla następnych etapów. Ten sam
fingerprint policy nadal jest egzekwowany, a `frozen_components.json`
zawiera hashe konfiguracji, źródeł, demonstracji i raportu. Relative i
absolute mają wyłącznie prowizoryczne skale; nie wybrano budżetu pilota.
Wybór generatora dotyczy fixed-k; transfer kontraktu na u/d wymaga osobnej
walidacji i nie jest potwierdzony tym pomiarem.
Pełne odpowiedzi, etapy walidacji, BFS, CLLP i faktyczne endpointy są w
`components_validation/openai/*.json`; agregaty w `summary.json`, tabela V
w `value_predictions.csv`, wykresy w `generator_reached.html` i
`value_trajectories.html`, opis w `report.md`.

Ta walidacja użyła 142 prób z usage, 143 logicznych żądań i jednego
lokalnego trafienia cache. Ledger rozlicza je konserwatywnie jako
**$0,047472540**; nie było nowych nieznanych rezerwacji. Razem dotąd
$0,093002234 rozliczone oraz $0,579189600 poprzedniej nieznanej rezerwacji,
czyli **$0,672191834 zajęte z $40**. To nie jest rachunek; stawki zawierają
przyjęty bufor regionalny. 1k nadal pozostaje zamknięty.

Następny krok: kalibracja na kontekstach train w ramach subbudżetu
`calibration`, a następnie wybór skal i budżetu na `scale_validation`.
Nie wykonywano tu kalibracji ani pilota. Polecenie wznowienia kalibracji
(wykonuje płatne żądania; uruchamiać tylko jako kolejny etap):

```bash
env -u OPENAI_API_KEY CARL_CAMPAIGN_STAGE=calibration uv run --extra icl python -m scripts.run_icl_search calibrate --config configs/icl/campaign_data/frozen_components_relative.json --execute
```

Kontrola po zamrożeniu: wszystkie trzy konfiguracje przechodzą wymuszenie
fingerprintu policy, dziewięć hashy `frozen_components.json` jest zgodnych,
a `make test` zakończył się wynikiem **264 passed**. Ruff i Pyright dla
nowych skryptów przeszły bez błędów.

## Preflight kalibracji v2 — wykonany offline przed live

Zweryfikowano lokalne źródło `rl-data/validation/sokoban/offline/12-12-4/sokoban_12_12_4_trajectories_part_1.pkl`: jego SHA-256 zgadza się z proweniencją `training.json`. Odtworzono w środowisku pełne udane kontynuacje **8/8** kontekstów i potwierdzono końcowe `is_solved`; 8 layoutów jest różnych także modulo D4. Wszystkie **128** zapisanych przyszłych stanów dla `k≤16` i ich prefiksy akcji zgadzają się z odtworzonym źródłem. Po wykluczeniu 1 powtórzonego stanu i 2 powrotów do źródła pozostaje **125** różnych kwalifikowanych stanów w obrębie kontekstów. Długości świadków nie dowodzą optymalności.

Przed live przyjęto **8 kontekstów × 4 różne cele losowane bez zwracania × R=3, L=8 i D_ref=16**, seed 20261029. Górna liczba to 96 rolloutów i 768 logicznych decyzji policy. Wobec poprzednich 57 realnych odpowiedzi zamrożonej policy, scenariusz wszystkich 768 kroków jako płatnych, z zapisem 30m dla całego wejścia i 10% buforem regionalnym, daje około **$0,26980** przy średnim usage albo **$0,38419** przy p95. Lokalny proxy przy 8000 input i 2048 output na każde żądanie, z droższą stawką 12h, to około **$2,55468**. Limit input jest lokalną estymacją tokenizera, więc proxy nie jest gwarancją faktury; twarde atomowe rezerwacje ledgeru nadal egzekwują sublimit kalibracji **$4** i limit całej kolejki $40, również dla nieznanych kosztów. Bieżącego niższego limitu konta nie da się tu odczytać.

Maszynowe wyniki, wybór losowy, hashe, szczegóły źródła i założenia kosztu: `analysis/luna_icl/campaign_100/calibration_v2_preflight/{preflight.json,preflight.md}`. Preflight nie wywołał API, nie odczytywał plansz ani świadków pilota lub 1k.

### Kalibracja v2: wznowienie i integralność

Ustalony kontrakt: 8 kontekstów treningowych, 4 różne cele na kontekst,
`D_ref=16`, `R=3`, `L=8`, seed 20261029. Konfiguracje warunków w
`configs/icl/campaign_data/{relative_u,absolute_d,ratio_d_over_ac,normalized_offset,fixed_k}.json`
dziedziczą `calibration_live_v2.json`; skale są nadal prowizoryczne.
Wszystkie cztery kalibrowane warunki pobierają tę samą zapisaną, raz
wylosowaną listę ważonych ID demonstracji. `normalized_offset=k/16` przy
konfigurowalnym stałym mianowniku; na planszy zapytania nie oblicza się
referencyjnych trajektorii, median ani rang. Fingerprint kalibracji obejmuje
pełne dane treningowe, model i kontrakt policy, `R`, `L`, `D_ref`, dobór
celów, RNG i seed; osobny skrót wykrywa zmianę zapisanych etykiet i wag.
CLLP korzysta z zapisanych rozkładów dla identycznych wejść; osobne seedy
losują akcje niezależnie warunkowo na tych zrealizowanych rozkładach.
Sam niezasiany losowy output modelu można dokładnie odtworzyć tylko z cache.

Jedna odpowiedź policy podczas realnej kalibracji była niepoprawnym JSON-em.
Jej surowy rekord zachowano w `rejected_policy_responses`; zapytanie modelu
można ponowić z tym samym payloadem. Przerwany rollout zaczyna się ponownie
od źródła z tym samym seedem i odtwarza prefix przez cache; nie stanowi
nowej niezależnej obserwacji. Akcje częściowej próby oraz ponowienie są
jawnie rozliczane. Wznowienie jest ograniczone liczbą prób naprawy i wspólnym
ledgerem. Nie uruchamiać drugiej kopii tego samego procesu.

```bash
uv run --extra icl python -m scripts.calibration_progress
env -u OPENAI_API_KEY uv run --extra icl python -m scripts.resume_calibration_v2
uv run --extra icl python -m scripts.audit_calibration_v2
uv run --extra icl python -m scripts.report_calibration_v2
```

Po zakończeniu audyt przelicza mediany z całkowitych liczników trafień i
weryfikuje etykiety z zachowanych prób bez wywołań API. Główne porównanie w
`budget_sweep` to `relative_u − absolute_d`; historyczny `relative − fixed`
pozostaje osobnym trybem. W module Plotly ewaluacji należy przekazać jawnie
`comparison_methods=("relative_u", "absolute_d")`, ponieważ jego stary
domyślny wybór nadal dotyczy `relative`/`fixed`.

### Wynik kalibracji v2

Rzeczywista kalibracja `gpt-6-luna` zakończona: **96/96 rolloutów i 32/32
celów**, z 33 dokładnymi trafieniami. Pozostawiono **11/32 celów** z hitami;
21 celów miało zero hitów. Spośród 8 kontekstów 7 ma przynajmniej jeden
trafiony cel, a jeden jest wykluczony. Cztery z ośmiu kontekstów mają tylko
jeden odróżnialny czas i wszystkie ich etykiety `u=0.5`; jedynie trzy
konteksty rozróżniają co najmniej dwa poziomy czasu. To ograniczenie wyniku,
nie przesłanka do ponownego strojenia.

Wszystkie obserwowane wartości `p_hat` są równe 0 albo 1; dla 30/32 celów
wszystkie trzy zapisane sekwencje akcji były identyczne. Spośród 151 różnych
wejść policy w aktywnym cache 149 miało rozkład punktowy, a dwa więcej niż
jedną akcję z dodatnią masą. Seedy akcji są niezależne, ale trzy obserwacje
są warunkowe na tych samych raz otrzymanych i cache’owanych rozkładach;
to nie są niezależne próbki odpowiedzi LLM. R=3 daje też bardzo małą
rozdzielczość oszacowania i nie należy interpretować tej kalibracji jako
stabilnego pomiaru niepewności.

Odnotowano dwa przerwane rollouty z niepoprawnym JSON-em, których odpowiedzi
zachowano w `rejected_policy_responses`; tę samą próbę wznowiono z tym samym
seedem. Przerwane fragmenty nie tworzą nowych niezależnych obserwacji.
Całkowity koszt tego etapu wg ledgeru to **$0,063213442** z 153 fizycznych
prób, 270733 tokenami input i 48083 output; cache-write 266753, cache-read
3521. Jest to konserwatywny szacunek, nie rachunek. Brak nowych nieznanych
rezerwacji. Wszystkie 96 prób przeliczono niezależnym audytem median, wag,
CDF i etykiet bez dodatkowych wywołań: zero poprawek; sprawdzono też
rachunek osiągnięcia celu i długości zapisanych prób. Weryfikacja czterech
trybów potwierdziła identyczne cztery pierwsze ID demonstracji, ich wagi,
stany i cele oraz jedną sygnaturę kalibracji; zapytanie zawiera tylko stan,
cel zadania, stałą skalę i liczbę subgoali.

Raport z rozkładem po offsetach, wszystkie seedy, próby, cele i koszty:
`analysis/luna_icl/campaign_100/calibration_v2/{report.md,report.json,calibration_targets.csv,calibration_trials.csv,demo_equivalence.json,coverage.html}`.
Pliki źródłowe JSON: `data/icl/campaign_dataset_v2/calibration/65ecc5c170ab42b35f28eaac5f8431b092ee4c1b44a7c9561cb0901afc65af70/`.
Łączny ledger kampanii: **$0,156215676 rozliczone + $0,579189600 wcześniejszej
nieznanej rezerwacji = $0,735405276 zajęte z $40**. Następny etap to
walidacyjny wybór skali/budżetu na rozłącznych planszach, nie pilot ani 1k.

Końcowa weryfikacja implementacji kalibracji i nowych trybów: `make test`
**304 passed**, Ruff i Pyright dla zmienionych modułów bez błędów.
Uwaga: konfiguracje `relative_u`/`absolute_d` są obecnie ustawieniami
walidacyjnymi; pilotowe 50+50 plansz, seed 11 oraz główny budżet wymagają
osobnego zamrożenia po walidacji.

### 06 — walidacja skal i budżetu (w toku)

Przed live zapisano protokół `scale_tuning/protocol.md`, siatkę
`configs/icl/campaign_data/scale_tuning/design.json` i niezmienny manifest
`scale_tuning/solver/manifest.json`. Dwie pełne plansze walidacyjne, po jednej
z rodziny, i seed 11 służą wszystkim sześciu warunkom przy B=128. Rozmiar tej
próby nie pozwala wiarygodnie estymować solve rate. Rozstrzygnięcie remisu,
również samych zer, to pierwszy indeks pary; nie wybierać pary z niepełnej
macierzy. Przy błędzie API/parsera, replay albo systemicznym zatrzymaniu przez
limity nie wolno ogłaszać gotowości pilota.

Primitive-only z tym samym V wykonał po 128 akcji na każdej z dwóch plansz,
żadnej nie rozwiązując. Oba pełne drzewa przeszły replay; przypadki i manifest
są w `scale_tuning/primitive_v2/`. Direct goal-conditioned nie ma uczciwego
wejścia w obecnej policy: wymaga pełnej docelowej planszy, której nie znamy.

Proces strojenia z dwoma workerami zakończył się statusem `api_error` po
czterech przypadkach; nie wybieramy pary z niepełnej macierzy. Powrót do
tego samego manifestu nie naprawia zapisanego przypadku, a zmiana kodu lub
kontraktu wymaga nowego manifestu. Przed każdym wznowieniem koniecznie
sprawdzić, czy żaden proces nie działa; nie uruchamiać drugiej kopii:

```bash
pgrep -af 'python -m scripts.tune_campaign_scales'
env -u OPENAI_API_KEY uv run --extra icl python -m scripts.tune_campaign_scales --provider openai --workers 2
uv run --extra icl python -m scripts.audit_scale_io
uv run --extra icl python -m scripts.estimate_campaign_pilot
uv run --extra icl python -m scripts.campaign_scale_stress --plan
```

Nie ma jeszcze `freeze.json`. Niezależny stress B=1024 wolno rozpocząć po
kompletnym, poprawnym stroju, jego `--plan` i `--prepare`, tylko jeśli
pozostały sublimit na to pozwala; `--execute` jest płatny. Presety do 256 nie
obejmują żadnej z czterech znanych walidacyjnych ścieżek 32×32 długości
572–735; nie są to długości minimalne. Kandydat `[256,1024,2048]` pozostaje
hipotezą do sprawdzenia, a nie zatwierdzonym wyborem. Trzy zakończone bez
błędu przypadki wykorzystały całe B=128 bez rozwiązania; wszystkie cztery
zapisane drzewa przeszły replay. Wstępny scenariusz wyłącznie dla relative-u
(`scale_tuning/pilot_cost_projection/preliminary.md`) to ok. $280 dla 100
plansz i trzech budżetów przy liniowej ekstrapolacji logicznych tokenów oraz
cenie zapisu cache dla całego inputu. To nie jest prognoza ani dowód kosztu:
rzeczywisty cache, długość runów i inne plansze zmienią wynik. Mimo to wynik
nie uzasadnia założenia, że całość zmieści się w $40. Pełny estymator czeka
na kwalifikujący się wybór z kompletnej macierzy.

#### Dodatkowy prompt generatora — eksploracja na prośbę użytkownika

Oddzielny, wcześniej zapisany projekt `configs/icl/campaign_data/generator_prompt_v1.json`
porównuje aktualny kontrakt z pełniejszym opisem legalnych ruchów, zachowania
celów i liczby obiektów oraz bezskalową demonstracją 32×32 wygenerowaną
osobno dla treningu. Cztery oryginalne skalibrowane przykłady pozostają bez
zmian; nowy przykład nie ma fikcyjnej etykiety u. Porównanie użyło jednej
planszy z każdej rodziny splitu component_validation, dwóch żądań u i
zamrożonej CLLP. Nie przeglądano wyników pilota.

Live `gpt-6-luna`: baseline **0/4** zapytań z osiągniętym odmiennym celem,
nowy wariant **0/4**. Baseline zwrócił dwie odrębne statycznie poprawne
propozycje, dwie kopie źródła i trzy błędne propozycje; nowy wariant nie
zwrócił żadnej poprawnej odmiennej propozycji, za to trzy puste odpowiedzi
i jedną błędną propozycję. Brak błędów API/parsera w tym teście. To bardzo
mała próba, lecz nie uzasadnia zastąpienia zamrożonego promptu. Wszystkie
surowe odpowiedzi, rollouty CLLP, endpointy i mianowniki są w
`generator_prompt_v1/openai/`; 24 fizyczne próby kosztowały według ledgeru
**$0,013113635**. Komenda wznowienia, po sprawdzeniu aktywnego procesu:

```bash
env -u OPENAI_API_KEY uv run --extra icl python -m scripts.campaign_generator_prompt --provider openai --execute
```

W bieżącym stroju `absolute_d[0]` na 12×12 policy oddała uszkodzony JSON mimo
statusu `completed`: brakowało `]`, a limit wyjścia nie został osiągnięty.
Przypadek przerwano po 50 akcjach, zachowano surową odpowiedź i drzewo, które
przeszło replay. Ten błąd blokuje wybór z macierzy i freeze. Nie zastępować
oryginału nieudokumentowanym restartem: powtórzenie prefixu oznaczałoby
dodatkowe faktyczne akcje. Plan naprawy pojedynczego wywołania i ograniczonego
retry opisuje `scale_tuning/policy_json_recovery.md`; zmiana kontraktu wymaga
nowego manifestu i rewalidacji lub ponownej kalibracji. Skrypt
`scripts.recover_scale_case --check` działa offline, zaś `--prepare` archiwizuje
materiał dopiero po zakończeniu tunera; nie modyfikuje cache.
Po zatrzymaniu tunera archiwizację wykonano. 26 plików wejściowych oryginalnego
manifestu zweryfikowano i zapisano jako `scale_tuning/original_manifest_inputs.zip`.
Oryginalny wadliwy wpis cache pozostał nietknięty.

Weryfikacja: `make test` 331 passed, po zmianie testu kalibracji dodatkowo 5
testów tej grupy passed. Ruff, Pyright kodu i testów oraz Pyright czterech
notebooków są czyste. `make lint` kończy się błędem mypy: 26 usterek w 12
plikach, w tym starszych modułach repo; nie należy raportować go jako passed.

#### Wznowienie po naprawie JSON

Dodano jawny `json_max_retries=1`: tylko niepoprawny składniowo JSON lub
odpowiedź niebędąca obiektem jest powtarzana wewnątrz wywołania, przed
następną akcją symulatora. Oba pełne wyniki zostają w osobnych slotach cache,
z tym samym payloadem, a retry przechodzi przez wspólne limity i ledger.
Błąd transportu lub nieznany koszt nie jest automatycznie ponawiany.
Domyślne 0 zachowuje stary fingerprint. Nowy fingerprint wynosi
`3e422d383fe00a6272a0791b8b3d3b58c360f010467a1c0dce86d39e60599fd6`;
stare wyniki nie stają się nowym porównaniem. Opis implementacji i stare
skróty: `scale_tuning/json_retry_implementation.md`.

Nowa kalibracja `configs/icl/campaign_data/calibration_json_retry_v1.json`
zakończyła 96/96 rolloutów, 11 zachowanych celów i 21 wykluczonych, z 648
trafieniami lokalnego cache i bez nowych płatnych wywołań. Niezależny audyt
nie zmienił żadnych wag ani etykiet. Dane:
`data/icl/campaign_dataset_v2/calibration/08e8317905e28ebc90073b94aad3ffea004ef265f5ad2d09065e44c2137d80ab/`.
Nowy manifest eksperymentalnego strojenia, zapisany po zaobserwowanym błędzie,
jest w `scale_tuning/solver_json_retry_v1/`; ta sama siatka, plansze, seed i
limity dotyczą wszystkich metod. To ponowna eksploracja, nie niezależny test
potwierdzający. Proces jest uruchomiony; przed ponownym wykonaniem sprawdzić
czy nie działa, potem wznowić bez drugiej kopii:

```bash
pgrep -af 'python -m scripts.tune_campaign_scales'
env -u OPENAI_API_KEY uv run --extra icl python -m scripts.tune_campaign_scales --design configs/icl/campaign_data/scale_tuning/design_json_retry_v1.json --output analysis/luna_icl/campaign_100/scale_tuning/solver_json_retry_v1 --provider openai --workers 2
uv run --extra icl python -m scripts.estimate_campaign_pilot --results-dir analysis/luna_icl/campaign_100/scale_tuning/solver_json_retry_v1 --output analysis/luna_icl/campaign_100/scale_tuning/pilot_cost_retry_v1
uv run --extra icl python -m scripts.campaign_scale_stress --tuning-dir analysis/luna_icl/campaign_100/scale_tuning/solver_json_retry_v1 --output analysis/luna_icl/campaign_100/scale_tuning/stress_json_retry_v1 --plan
```

Strojenie retry zakończone: **12/12** przypadków przeszło replay, każdy
wyczerpał `B=128`, żaden nie rozwiązał planszy. Pary w obu rodzinach i dla
wszystkich trzech metod mają po 0/2 rozwiązań, więc zgodnie ze z góry
zapisaną regułą remisu wybrane są pierwsze pary: `relative_u=(0.85,0.5)`,
`absolute_d=(8,4)`, `fixed_k=(8,4)`. To wybór operacyjny przy samych zerach,
nie dowód przewagi lub stabilna estymata solve rate. Wszystkie zakończenia
to limit akcji, nie requestów ani tokenów. Wyniki i pełne drzewa są w
`scale_tuning/solver_json_retry_v1/`.

Stress na pierwszej planszy walidacyjnej 32×32 dla `relative_u` i B=1024
trwa; ma trwały trace każdej akcji w `scale_tuning/stress_json_retry_v1/`.
Przed uruchomieniem oszacowano ok. $1.063 bez lokalnego replay plus $0.581
rezerwacji; stage miał wówczas $0.838 rozliczone z $4. Limit pozostaje
egzekwowany przez ledger. Proces uruchomiono raz, po `--prepare`; po przerwaniu
nie uruchamiać go ponownie bez zbadania istniejącego częściowego trace’u.

Kwalifikujący się estymator `scale_tuning/pilot_cost_retry_v1/estimate.md`
w scenariuszu liniowym, bez lokalnego replay i wcześniejszych zakończeń,
przewiduje ok. $939 dla trzech metod i trzech niezależnych budżetów
256/1024/2048 na 100 planszach. Cena to proxy: po jednym walidacyjnym
przypadku na rodzinę/metodę, bez uwzględnienia oszczędności cache; nie jest
prognozą rachunku. Scenariusz jednak znacznie przekracza $40 całej kampanii
i osobne sublimity $12/$12/$4. Nie zamrażać pilota ani `freeze.json` bez
rozwiązania tego ograniczenia i dokończenia oceny B*.

Po wdrożeniu retry ponowne `make test` dało **344 passed**. Przy zmianie
źródeł zachowano zweryfikowane archiwa wejść obu wcześniejszych manifestów.

#### Zmiana zakresu: pełne solvy tylko na małych planszach

Na prośbę użytkownika przerwano stress 32×32 sygnałem `SIGINT` po zapisaniu
200 akcji i 113 ekspansji. Nie ma końcowego `case.json`. Zdarzenie `finish`
w częściowym trace zawiera domyślny `frontier_exhausted`, ponieważ zostało
zapisane w `finally` podczas przerwania; **nie jest wynikiem solvera**.
Żądanie aktywne podczas przerwania oznaczono w ledgerze jako `unknown`,
pozostawiając pełną rezerwację $0,5808792. Nie uruchamiać go ponownie.

Użytkownik wykluczył pełne solvy 32×32; dopuszcza jedynie oddzielną
diagnostykę losowych ruchów i deterministycznego BFS do pobliskich stanów.
Nowy pilot musi używać plansz 10×10×4 lub 12×12×4 oraz osobnych manifestu,
walidacji i `freeze.json`. Wcześniejszy skład 50 małych + 50 32×32 oraz
poprzednie B* nie są zatwierdzonym planem nowego pilota. Holdout nie jest
uruchomiony. Żadnych komórek pilota nie wykonano w tym kroku.

Oddzielna diagnostyka 32×32 (`random_walk_bfs_32/`) użyła dwóch plansz
walidacyjnych i losowych spacerów o długościach 1/2/4/8; BFS ograniczono do
głębokości 8 i 2500 stanów. Wszystkie 8 endpointów miało świadka, w tym jeden
był równy stanowi startowemu; żaden nie wymagał przemieszczenia skrzyni.
Cztery różne cele o dystansach BFS 2, 4, 2, 2 osiągnęła live CLLP 4/4,
L=8, koszt 10 fizycznych prób $0,004259203. Wejście policy obejmowało stan,
cel i współrzędne wyliczone z obu plansz; nie zawierało wyników BFS ani
świadków. Dystans BFS jest odrębny od d jako średniego first-hit policy i u
jako ważonego percentyla z propozycji. Dane te nie są pełnym solvem 32×32.

Nowa kohorta pilota `data/icl/campaign_dataset_12x12_v1/` zawiera 100
różnych realnych layoutów 12×12×4; 50 zachowano z poprzedniego pilota, a 50
dobrano deterministycznie z innych shardów. Każdy pełny stan ma prywatnego,
odtworzonego świadka poza publicznymi manifestami; brak przecięć z treningiem,
walidacją, historycznymi zapytaniami i zachowanym holdoutem sprawdzono
programowo. Ponowne przygotowanie daje identyczne hashe. Raport integralności:
`small_boards_v1/integrity.json`. Nowego holdoutu 1000 dla małych plansz nie
przygotowano ani nie uruchomiono; dotychczasowy realny holdout 500 zachowano.

Trwa niezależna walidacja skal na dwóch pełnych planszach 12×12 z B=128,
seedem 11, nowym fingerprintem policy i trzema metodami:
`scale_tuning/real12_v1/solver/`. Nie uruchamiać drugiego procesu. Przyszłe
niezależne progi `[128,256,512]`, w tym B*=256, pozostają kandydatami do
czasu sprawdzenia stressu i audytu. Próba wznowienia małego shardu z dwoma
workerami na dwóch pełnych planszach walidacyjnych przeszła offline w trybie
mock: 4/4 uruchomień, jedna wcześniej zakończona komórka pominięta,
0 wywołań providera (`small_boards_v1/resume_smoke_workers2_final/`).

Nowy runner wymaga audytowanego freeze i jawnego `--pilot-freeze`. Używa
osobnych checkpointów dla `(freeze_hash, board_id, method, seed, B)`,
przeplata dwie metody, wykonuje całą fazę B* przed innymi progami, rozlicza
żądania atomowo w ledgerze i zapisuje niepełne oraz ocenzurowane przypadki.
Domyślne dwa workery są ograniczone do zakresu 1–3; backoff obejmuje tylko
jawne 429/5xx, z osobną rezerwacją i zachowaniem nieznanego kosztu oryginału.
Wyniki i mianowniki parowane będą oddzielone od nieukończonych komórek.
Przekroczona prognoza nie blokuje pilota: zgodnie z nowszym poleceniem
użytkownika etapy wykonuje się w kolejności, aż zadziała rzeczywisty limit.
Na tym etapie freeze ani żaden wynik pilota jeszcze nie istnieją.

Walidacja real12_v1 zakończona: 12/12 runów i replayów, każdy wykorzystał
128 akcji, 0 rozwiązań. Zgodnie z wcześniejszą regułą remisu wybrano
`relative_u=(0.85,0.5)`, `absolute_d=(8,4)` i `fixed_k=(8,4)`. Na dwóch
planszach to eksploracyjny wynik, nie wiarygodny solve rate. Trwa pojedynczy
stress B=256 na 12×12 (`scale_tuning/real12_v1/stress_256/`); sprawdzić proces
przed jakąkolwiek próbą uruchomienia ponownie. Osobny estymator
`scale_tuning/real12_v1/pilot_cost/estimate.md` daje scenariusz średnio $120
dla dwóch metod i budżetów [128,256,512], zakładając liniowy koszt bez
lokalnego replay; nie jest to rachunek. Zgodnie z nowszą instrukcją scenariusz
nie blokuje działania w granicach rzeczywistych limitów.

Przedstawiony offline smoke z dwoma workerami i scheduler przeszły testy
przerwania/wznowienia. Runner rozdziela etapy per instancja API i obejmuje
bounded retry jawnych 429/5xx; niezależnie sprawdzono wymuszone przeplatanie,
barierę B*, auth stop i rezerwacje. Repo: `make test` 380 passed; Ruff i
Pyright wraz z notebookami bez błędów.

Stress 12×12 zakończył się po pełnych 256 akcjach, bez rozwiązania i bez
wcześniejszego limitu API; niezależny replay potwierdził 256 akcji i 146
krawędzi drzewa. Utworzono `scale_tuning/real12_v1/freeze.json` i wykonano
rzeczywisty `--plan`: `scale_tuning/real12_v1/pilot_plan.json` zawiera 100
różnych ID, jedną rodzinę 12×12, obie metody, seed 11 i niezależne budżety
w kolejności 256 → 128 → 512. Core to 600 runów, w tym 200 przy B*=256.
Na początku pozostało $37,585585353 globalnie i po $12 w etapach obu metod;
maksymalny dodatkowy przydział core to $24. Scenariusz $120 pozostaje tylko
estymacją, więc zgodnie z instrukcją uruchomiono kolejkę w granicach ledgeru.

Pilot `analysis/luna_icl/campaign_100/pilot_small_v1/` jest uruchomiony z
dwoma workerami. Przed wznowieniem sprawdzić, czy proces już działa. Po
zatrzymaniu i inspekcji ewentualnych przerwanych/failed komórek wznowienie:

```bash
pgrep -af 'scripts.run_budget_sweep.*pilot-freeze'
env -u OPENAI_API_KEY uv run --extra icl python -m scripts.run_budget_sweep --pilot-freeze analysis/luna_icl/campaign_100/scale_tuning/real12_v1/freeze.json --execute --resume --workers 2
uv run --extra icl python -m scripts.plot_small_pilot_progress
```

Trwałe checkpointy, ślady, `progress.json` i `summary.csv` stanowią źródło
faktycznych wyników. Opcjonalne baseline'y uruchamia się dopiero po wykonaniu
core i zabezpieczeniu kosztu; holdout 1000 pozostaje niewykonany.

### Zmiana zakresu: małe testy zamiast dużego pilota

Na późniejszą prośbę użytkownika przerwano pilot bez rozpoczynania kolejnych
przypadków. Stan po zatrzymaniu: 22 ukończone runy, 11 pełnych par B=256,
0/11 rozwiązań dla obu metod, dwa jawnie przerwane przypadki oznaczone jako
failed i 576 brakujących. Przerwanie nie jest błędem autoryzacji ani
wynikiem pełnego eksperymentu. Wszystkie bieżące żądania rozliczono, bez
nowej nieznanej rezerwacji. Rozliczony koszt etapów pilota wg ledgeru:
`pilot_relative` $1,012103 oraz `pilot_absolute` $0,966446; dwie starsze
nieznane rezerwacje innych etapów pozostają zabezpieczone.

Przed refaktoryzacją całą zawartość 55 ścieżek związanych z freeze
zweryfikowano po SHA-256 i zapisano do
`scale_tuning/real12_v1/pilot_freeze_inputs.zip`. Zmiana kodu zmieni hashe
freeze: nie wolno bezpiecznie wznowić pilota na zmienionym kodzie pod starym
manifestem. Odtworzenie wersji historycznej wymaga izolowanego checkoutu z
archiwum i osobnej inspekcji dwóch przerwanych przypadków. W tej kolejce nie
uruchamiać dużego pilota ani holdoutu bez nowej dyspozycji użytkownika.
