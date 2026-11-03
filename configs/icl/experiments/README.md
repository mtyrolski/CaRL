# Konfiguracje eksperymentów Luna ICL

`base.json` definiuje model wszystkich trzech komponentów, few-shoty, budżety,
kalibrację i siatkę strojenia. Pozostałe warunki dziedziczą z niego przez `extends`.
Loader odrzuca nieznane pola, cykle dziedziczenia i niepoprawne skale.
Schemat pojedynczej konfiguracji jest w `../experiment.schema.json`.

| Macierz | Zastosowanie |
|---|---|
| `matrix_smoke.json` | Dwie metody, po dwie plansze jeden ruch od rozwiązania; sprawdzenie interfejsu |
| `matrix_pilot.json` | Relative vs fixed AdaSubS; pełne stany 12×12×4 i 32×32×10; 48 przebiegów |
| `matrix_transfer.json` | Wszystkie przygotowane rodziny plansz; 144 przebiegi |
| `matrix_embeddings.json` | Obie metody z retrievalem przykładów przez embeddingi |
| `matrix_ablation.json` | Dodatkowo skala absolutnego czasu trafienia i fixed z takim samym limitem rolloutów |

Z katalogu głównego repozytorium, po `uv sync --frozen --extra icl --group lint`:

```bash
make icl-check
make icl-smoke
# Po uzupełnieniu OPENAI_API_KEY w .env:
make icl-calibrate
uv run --extra icl python -m scripts.run_icl_search matrix --execute \
  --matrix configs/icl/experiments/matrix_pilot.json
```

Domyślne pary skal są oznaczone jako `pilot_unvalidated`. Dobór na walidacji:

```bash
uv run --extra icl python -m scripts.run_icl_search tune --execute \
  --config configs/icl/experiments/relative.json \
  --output analysis/luna_icl/selected
uv run --extra icl python -m scripts.run_icl_search matrix --execute \
  --matrix analysis/luna_icl/selected/matrix_selected.json
```

Sekcja `tuning` w `base.json` zawiera `relative_scales` i `fixed_scales`.
Obie listy muszą być niepuste i mieć tę samą długość. Każda para ma kolejność
`[większa, mniejsza]`; duplikaty są odrzucane. Horyzonty fixed nie mogą przekroczyć
limitu wykonania policy. Ustawienia nie są już zaszyte w runnerze.

`policy_coordinate_features` w `base.json` jest domyślnie `false`. Opcjonalne
`fixed_policy_coordinates.json` i `relative_policy_coordinates.json` dodają do
pytań i przykładów policy współrzędne gracza, skrzyń i celów odczytane z tych
samych plansz. Generator i value dostają swoje dotychczasowe wejścia. To osobna
wersja policy, więc wariant relative wymaga **nowej kalibracji**. W małym
debugowaniu live ten wariant poprawił część pojedynczych ruchów, ale nie
wykazał niezawodności całego searcha; nie jest automatycznie nowym baseline'em.

`tuned_fixed.json` i `tuned_relative.json` to osobny, eksperymentalny profil
po testach na trajektoriach: CLLP używa współrzędnych i `reasoning=low`, generator
zwraca pełne plansze w schemacie zależnym od układu, a value przewiduje liczbę
pozostałych ruchów, którą kod przelicza na `discount ** n`. Prompty i parametry
można zmieniać przez dziedziczone konfiguracje. Generator może maksymalnie trzy
razy odpowiedzieć po informacji o błędnej strukturze, ale nie korzysta z
symulatora ani wyszukiwania osiągalności. Value nie jest nadpisywane dla stanu
końcowego; cechy planszy, w tym liczba skrzyń na celach i dolne ograniczenie
Manhattan, stanowią jawne wejście modelu.

Przed uruchomieniem profilu przygotuj lokalnie 32 przykłady value z istniejących,
zweryfikowanych trajektorii treningowych (bez wywołań API):

```bash
uv run python -m scripts.run_icl_search prepare-value --config configs/icl/experiments/tuned_fixed.json
```

To nadal profil pilotażowy. Relative wymaga nowej kalibracji dla tego samego
CLLP. Żeby zmierzyć prawdziwe przejścia bez osobnego verifiera generatora, użyj
`scripts.evaluate_trajectory_components` oraz notebooka
`notebooks/luna_trajectory_diagnostics.ipynb`. Cele offline są podawane policy
tylko jako plansze, bez akcji z rozwiązania.

`--mock` nie korzysta z Luny i nie mierzy jej skuteczności. Live API wymaga jawnego
`--execute`; kalibracje i cache mock/live pozostają oddzielne. Szczegóły danych,
kosztów i zgodności z propozycją: [ICL_SEARCH.md](../../../ICL_SEARCH.md).
