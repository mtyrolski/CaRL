#!/usr/bin/env python3
"""Build an offline notebook for the Sokoban intent-representation study."""

import argparse
import os
import tempfile
from pathlib import Path
from textwrap import dedent

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "analysis/intent_representations_20261114T075324Z"


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbformat.v4.new_code_cell(dedent(source).strip())


def notebook(run: Path):
    relative = str(run.relative_to(ROOT))
    cells = [
        markdown("""
        # Sokoban: reprezentacje intencji i ICL

        Porównujemy sposoby przedstawienia tej samej rodziny intencji w
        odpowiedzi małego modelu Qwen oraz wpływ przykładów w kontekście.
        Notebook korzysta ze źródeł zapisanych lokalnie: jego ponowne
        wykonanie nie uruchamia treningu, GPU ani API. Przypadki brakujące
        lub przerwane są pokazane osobno od błędnych odpowiedzi modelu.
        """),
        code(f'''
        from pathlib import Path
        from datetime import datetime, timezone
        import hashlib
        import json
        import pandas as pd
        from IPython.display import Code, Image, Markdown, display

        here = Path.cwd().resolve()
        ROOT = next((p for p in (here, *here.parents)
                     if (p / "pyproject.toml").exists() and (p / "carl").is_dir()), None)
        if ROOT is None:
            raise RuntimeError("Uruchom notebook w checkoutcie CaRL")
        RUN = ROOT / {relative!r}
        ORIGINAL = ROOT / "analysis/intent_generator_20261114T032232Z"
        def read_json(path, default=None):
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
        def read_jsonl(path):
            if not path.exists():
                return []
            rows = []
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    print("Niepełny wiersz:", path, number)
            return rows
        def sha(path):
            if not path.exists():
                return None
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            return digest.hexdigest()
        def show_json(path):
            obj = read_json(path)
            if obj is not None:
                print("Plik:", path, "SHA-256:", sha(path),
                      "czas zapisu UTC:", datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat())
                display(Code(json.dumps(obj, ensure_ascii=False, indent=2), language="json"))
            return obj
        state = read_json(RUN / "run_manifest.json", {{}}) or {{}}
        study = read_json(RUN / "study_plan.json", {{}}) or {{}}
        finish_request = read_json(RUN / "finish_request.json")
        campaign = read_json(RUN / "campaign_state.json")
        print("Katalog:", RUN)
        print("Stan nadzoru:", state.get("status", "brak"))
        if state.get("status") == "active" and finish_request:
            print("Zakończenie zgłoszone; watchdog oczekuje na wyjście zarejestrowanych procesów.")
            print("Żądanie:", json.dumps(finish_request, ensure_ascii=False))
        if campaign is not None:
            print("Stan etapów eksperymentu:", json.dumps(campaign, ensure_ascii=False))
        print("Początek i termin UTC:", state.get("clock", {{}}).get("start_utc"),
              state.get("clock", {{}}).get("deadline_utc"))
        show_json(RUN / "study_plan.json")
        '''),
        markdown("""
        ## Pytanie i interpretacja

        Generatory otrzymują tylko bieżącą planszę i cel zadania. Przykłady ICL
        pochodzą ze sprawdzonych plansz treningowych; późniejsza plansza,
        ruchy demonstracji, offset i wynik sprawdzania nie trafiają do zapytania.
        Formaty JSON, zwarty formalny oraz kontrolowany język angielski są
        tłumaczone na tę samą strukturę: tablice warunków intencji i warunków
        zachowania. Współrzędne są zerowane, w kolejności wiersz, kolumna.

        Główna miara wymaga ścisłego odczytu formatu i wspólnego schematu oraz deterministycznej
        walidacji nietrywialnej intencji na bieżącej planszy. Druga miara
        usuwa z tego zbioru żądania obejmujące cały cel planszy. Poprawna
        intencja nie jest jeszcze wykonaną intencją ani rozwiązaniem Sokobana.
        Przedziały i różnice liczymy z bootstrapu **plansz**, nie ich fragmentów.

        Zestawy przykładów ICL wybrano bez wyników bieżącej oceny, ale część
        ich konstrukcji uwzględniała obserwacje z wcześniejszego eksperymentu.
        Jest to więc analiza eksploracyjna; wybranie najlepszego wariantu na
        tych wynikach wymagałoby nowego, niezależnego testu.

        Porównujemy całe prompty wraz z formatem odpowiedzi, a nie wyłącznie
        kodowanie tej samej treści przy identycznych tokenach wejściowych.
        `zero` oznacza brak kompletnych demonstracji z planszą; polecenia
        formalne i angielskie nadal zawierają krótkie przykłady składni, a
        przykłady te różnią się między promptami.
        Nowy prompt JSON różni się od promptu poprzedniego eksperymentu,
        dlatego ich wyników nie łączymy jako tej samej metody.
        Historyczny benchmark i trener adapterów r2/r4/r8 sprawdzały długości
        etykiet testowych ×10; etykiety nie trafiły do gradientów ani wyboru
        checkpointu. Nowe treningi nie czytają etykiet testowych ani ich długości.
        Niezależny audyt dopasowanych treningów rekonstruuje kolejność ID
        przykładów na podstawie danych i seeda; rzeczywiste ID na każdym kroku
        nie były zapisywane, więc odtworzona kolejność nie jest dziennikiem kroków.
        Audyt jest migawką z czasu zapisu pliku, a nie automatyczną gwarancją
        statusu późniejszych replikacji; bieżące statusy adapterów podano osobno.
        """),
        code('''
        print("Zapisany plan i konfiguracje w katalogu badania:")
        for name in ("study_plan.json", "contrasts_plan.json", "support_manifest.json", "runtime_environment.json",
                     "policy_calibration_amendment.json",
                     "experiment_plan.json", "matched_training_audit.json", "validation_checkpoint_audit.json",
                     "verification.json"):
            if (RUN / name).exists():
                show_json(RUN / name)
        frozen_hashes = read_json(RUN / "model_and_protocol_hashes.json", {}) or {}
        print("Porównanie zapisanych przed generacją hashy modelu i kodu z bieżącymi plikami:")
        if frozen_hashes:
            current = []
            for relative, expected in frozen_hashes.items():
                actual = sha(ROOT / relative)
                current.append({"plik": relative, "zapisany_sha256": expected,
                                "bieżący_sha256": actual, "zgodne": actual == expected})
            display(pd.DataFrame(current))
            print("Pliki zmienione lub niedostępne:", sum(not item["zgodne"] for item in current))
        else:
            print("Brak zapisanego snapshotu hashy.")
        '''),
        markdown("""
        ## Dane i pochodzenie plansz

        Jednostką podziału jest plansza. Dla każdej rodziny raportujemy
        liczbę kontekstów i niezależnych identyfikatorów układu oraz zachowane
        manifesty, hashe i ścieżki źródeł. Różne rodziny plansz mogą mieć różne
        pochodzenie, więc ich różnic nie można przypisywać wyłącznie liczbie
        skrzyń. Wcześniej oceniane układy nie są nowym zaślepionym testem.
        Flaga braku układu w poprzedniej macierzy LoRA nie dowodzi, że model
        nigdy wcześniej nie widział tej planszy ani jej odpowiednika.
        """),
        code('''
        data_path = RUN / "data/dataset.jsonl"
        ids_path = RUN / "data/test_ids.json"
        records = read_jsonl(data_path)
        frozen = read_json(ids_path, {}) or {}
        print("Dane:", data_path, "SHA-256:", sha(data_path))
        print("Manifest ID:", ids_path, "SHA-256:", sha(ids_path))
        print("Hash danych w manifeście:", frozen.get("dataset_sha256"))
        print("Liczba zamrożonych ID:", len(frozen.get("record_ids", [])))
        if records:
            frame = pd.DataFrame(records)
            if "evaluation_family" not in frame:
                frame["evaluation_family"] = frame.get("family", "brak")
            display(frame.groupby(["split", "evaluation_family"], dropna=False).agg(
                konteksty=("record_id", "size"), plansze=("layout_id", "nunique")))
            layouts = {split: {r["layout_id"] for r in records if r.get("split") == split}
                       for split in ("train", "validation", "test")}
            print("Przecięcia plansz train/test i validation/test:",
                  len(layouts["train"] & layouts["test"]), len(layouts["validation"] & layouts["test"]))
            if "previously_evaluated_in_lora_matrix" in frame:
                display(frame.loc[frame["split"] == "test"].groupby(
                    ["evaluation_family", "previously_evaluated_in_lora_matrix"], dropna=False).agg(
                        konteksty=("record_id", "size"), plansze=("layout_id", "nunique")))
        else:
            print("Brak danych w katalogu przebiegu — etap mógł się jeszcze nie zakończyć.")
        for path in sorted((RUN / "data").glob("*.json")) if (RUN / "data").exists() else []:
            if path != ids_path:
                show_json(path)
        '''),
        markdown("""
        ## Przykłady ICL

        Lista przykładów, ich identyfikatory i reguła wyboru są zamrożone
        w manifeście. Pokazujemy planszę i intencję każdej demonstracji, a nie
        późniejszy stan ani demonstracyjne ruchy. Różna długość promptu jest
        kosztem danej metody i jest raportowana razem z wynikami.
        Zestaw `balanced0` współdzieli jeden przykład z `legacy`, dlatego
        wyników tych dwóch zestawów nie interpretujemy jako niezależnych prób.
        """),
        code('''
        support_manifest = read_json(RUN / "support_manifest.json", {}) or {}
        examples = {r.get("record_id"): r for r in read_jsonl(ORIGINAL / "dataset.jsonl")}
        for name, selected in support_manifest.get("support_sets", {}).items():
            display(Markdown("### Zestaw `" + name + "`"))
            if not selected:
                print("Brak przykładów (zero-shot).")
            for item in selected:
                row = examples.get(item.get("record_id"))
                print("Rekord:", item, "odnaleziony:", row is not None)
                if row:
                    print("Podział:", row.get("split"), "typ:", row.get("effect_type"))
                    print("\\n".join(row.get("start_observation", [])))
                    print("Cel:", row.get("task_goal"))
                    print("Intencja:", json.dumps(row.get("intent"), ensure_ascii=False))
        '''),
        markdown("""
        ## Wyniki generatorów

        `syntax_valid` obejmuje ścisłą gramatykę danego formatu **i wspólny
        schemat** intencji; nie jest samą poprawnością dowolnej składni JSON.
        Dla reprezentacji JSON osobno zapisujemy `raw_json_valid`.
        `semantic_valid` oznacza ponadto poprawną nietrywialną intencję na
        bieżącej planszy. `valid_not_whole_goal` wyklucza warunek żądający
        całego celu. Podane liczby opisują ocenione odpowiedzi; kolumny
        `planned`, `missing` i `infrastructure_errors` pozwalają odróżnić
        niezrealizowane próby od błędów modelu.

        Pomocnicza analiza usuwa najwyżej jedną kompletną zewnętrzną otoczkę
        Markdown i stosuje tę samą regułę do wszystkich trzech reprezentacji.
        Nie zastępuje ona głównego wyniku ścisłego.

        Osobne katalogi `assessment_*` mogą być kolejnymi zapisami częściowej
        analizy lub jej odtworzeniem z tej samej kopii wejść; nie są dodatkowymi
        niezależnymi eksperymentami i ich mianowników nie wolno sumować.

        Zapisana odpowiedź to tekst po dekodowaniu z pominięciem tokenów
        specjalnych i usunięciu białych znaków z brzegów. Pozostawiono
        otoczki Markdown, ale nie zapisano oryginalnych ID tokenów.

        Jeśli uruchomiono ograniczone BFS, informuje ono tylko o znalezieniu
        ciągu ruchów w zadanych granicach. Limit oznacza wynik nierozstrzygnięty.
        BFS nie mierzy działania Qwena, Luny ani wytrenowanego kontrolera.
        """),
        code('''
        assessment_directories = [p.parent for p in RUN.glob("assessment*/summary.json")]
        def coverage(directory):
            item = read_json(directory / "summary.json", {}) or {}
            rows = item.get("results", [])
            return (sum(row.get("outputs", 0) for row in rows), len(rows),
                    bool((item.get("config") or {}).get("bfs")), directory.name)
        assessment_directories.sort(key=coverage)
        if not assessment_directories:
            print("Brak ukończonego zapisu oceny; nie jest to wynik zerowy.")
        for directory in assessment_directories:
            summary = read_json(directory / "summary.json", {}) or {}
            display(Markdown("### " + directory.name))
            print("SHA-256 podsumowania:", sha(directory / "summary.json"))
            if (directory / "reproduction_check.json").exists():
                show_json(directory / "reproduction_check.json")
            if (directory / "input_snapshot/manifest.json").exists():
                print("Snapshot dokładnych bajtów predykcji:", directory / "input_snapshot/manifest.json",
                      "SHA-256:", sha(directory / "input_snapshot/manifest.json"))
            for title, key in (("Wszystkie rodziny", "results"), ("Osobne rodziny", "family_results"),
                               ("Podział względem poprzedniej macierzy LoRA", "exposure_results")):
                display(Markdown("#### " + title))
                rows = summary.get(key, [])
                if rows:
                    display(pd.DataFrame(rows))
                else:
                    print("Brak danych.")
            print("Wspólne ustawienia, hashe i jawne niezgodności:")
            display(Code(json.dumps(summary.get("config", {}), ensure_ascii=False, indent=2), language="json"))
            for image in sorted(directory.glob("validity_*.png")):
                display(Image(filename=str(image)))
            core_figure = directory / "core_representations.png"
            if core_figure.exists():
                display(Image(filename=str(core_figure)))
            for image in sorted(directory.glob("bounded_bfs_*.png")):
                display(Image(filename=str(image)))
        '''),
        markdown("""
        ## Sparowane porównania

        Każda para porównuje te same konteksty. Średnia różnica przypisuje
        jednakową wagę każdej planszy, a przedział 95% powstaje przez
        powtórne losowanie plansz. Przy zbyt małej liczbie plansz pokazujemy
        wyłącznie wynik opisowy. Niezrealizowane pary nie są dopisywane jako
        sukces ani porażka.

        Główne i dodatkowe porównania zapisano przed generowaniem odpowiedzi
        w `contrasts_plan.json`. Przedziały są eksploracyjne i nie zostały
        skorygowane za wielokrotne porównania.
        """),
        code('''
        for directory in assessment_directories:
            summary = read_json(directory / "summary.json", {}) or {}
            pairs = summary.get("paired", [])
            print(directory.name, "liczba zapisanych kontrastów:", len(pairs))
            if pairs:
                display(pd.DataFrame(pairs))
            figure = directory / "paired_overall.png"
            if figure.exists():
                display(Image(filename=str(figure)))
        '''),
        markdown("""
        ## Replikacje treningu LoRA

        Dla formatów JSON, formalnego i kontrolowanego angielskiego plan obejmuje
        osobne adaptery z seedami treningu 1729, 2718 i 3141. Każdy seed jest
        pokazany osobno, wraz z liczbą ocenionych przypadków. O ile pozwolił
        czas, opisujemy średnią i zakres tylko dla ukończonych seedów; brak
        replikacji nie jest zerem. Rozrzut między seedami jest pomiarem
        zmienności treningu, a nie przedziałem bootstrapowym po planszach.
        """),
        code('''
        training_status = []
        for representation in ("json", "formal", "natural"):
            for seed in (1729, 2718, 3141):
                directory = RUN / f"adapter_repr_{representation}_seed{seed}"
                status = read_json(directory / "status.json", {}) or {}
                training_status.append({"reprezentacja": representation, "seed": seed,
                                        "status": status.get("status", "brak"),
                                        "kroki": status.get("completed_steps"),
                                        "plik": str(directory / "status.json")})
        display(pd.DataFrame(training_status))
        for directory in assessment_directories:
            summary = read_json(directory / "summary.json", {}) or {}
            seeds, seed_stats = summary.get("training_seed_points", []), summary.get("training_seed_summary", [])
            if seeds:
                display(Markdown("### " + directory.name))
                display(pd.DataFrame(seeds))
                display(pd.DataFrame(seed_stats))
            figure = directory / "training_seed_variation.png"
            if figure.exists():
                display(Image(filename=str(figure)))
        '''),
        markdown("""
        ## Przykłady odpowiedzi

        Dla pierwszego w kolejności zapisanego kontekstu z każdej rodziny
        pokazujemy bieżącą planszę i odpowiedzi wszystkich dostępnych metod.
        Przykłady nie są wybierane według sukcesu. Plansza późniejsza z
        demonstracji nie jest wynikiem modelu; stan osiągnięty przez BFS,
        jeśli istnieje, jest podpisany osobno.
        """),
        code('''
        for directory in assessment_directories:
            rows = read_jsonl(directory / "assessment.jsonl")
            selected = {}
            for row in rows:
                if row.get("status") == "evaluated" and row.get("family") not in selected:
                    selected[row.get("family")] = row.get("record_id")
            for family, identifier in selected.items():
                display(Markdown("### " + directory.name + " — " + str(family) + " — " + str(identifier)))
                chosen = [row for row in rows if row.get("record_id") == identifier and row.get("status") == "evaluated"]
                if not chosen:
                    continue
                print("Bieżąca plansza:\\n" + "\\n".join(chosen[0].get("start_observation", [])))
                for row in chosen:
                    print("\\nMetoda:", row.get("variant"), "format:", row.get("syntax_valid"),
                          "intencja poprawna:", row.get("semantic_valid"))
                    print("Surowa odpowiedź:", row.get("raw_output"))
                    print("Intencja:", json.dumps(row.get("intent"), ensure_ascii=False))
                    print("Uwagi sprawdzania:", row.get("semantic_errors"))
                    solver = row.get("solver") or {}
                    if solver.get("witness_found"):
                        print("Ciąg ruchów BFS (0=góra, 1=dół, 2=lewo, 3=prawo):", solver.get("witness_actions"))
                        print("Plansza osiągnięta przez BFS:\\n" + "\\n".join(solver.get("reached_observation") or []))
                    elif solver:
                        print("BFS — status:", solver.get("status"))
        '''),
        markdown("""
        ## Zamrożony kontroler

        Jeśli zapisano próby kontrolera, jego wejściem jest bieżąca plansza,
        intencja i historia własnych ruchów. Model kontrolera, prompt,
        dekodowanie, pamięć i budżet powinny być jednakowe dla porównywanych
        generatorów, a zablokowane ruchy liczą się do budżetu. Trening
        generatora lub zmiana reprezentacji nie oznacza trenowania polityki
        niskiego poziomu. Błędy infrastruktury i nieuruchomione próby są
        oddzielone od odpowiedzi oraz ruchów modelu.
        Odpowiedzi kontrolera są również zapisane po dekodowaniu, pominięciu
        tokenów specjalnych i usunięciu brzegowych białych znaków; pole
        `raw_policy_responses` zachowuje otoczkę przed jej normalizacją.

        W kalibracji zapisano hashe wag oraz `tokenizer.json`; konfiguracje
        `config.json`, `generation_config.json` i `tokenizer_config.json`
        związano hashem przed ewaluacją, ale nie ma pełnego dowodu ich
        wstecznej identyczności podczas wcześniejszej kalibracji.

        Osobny audyt kalibracji odtworzył 153 ruchy i 155 wywołań wybranego
        kontrolera. Spośród 12 kandydatów jeden odniósł sukces po jednym ruchu,
        co daje w jego kontekście `u = 0.5`; pozostałe mają `u = null`. To
        pojedyncza obserwacja, zbyt mało do uczenia skali. Generator w tym
        badaniu nie jest warunkowany skalą; offset demonstracji nie zastępuje
        liczby ruchów kontrolera.
        """),
        code('''
        policy_files = sorted(set(RUN.glob("policy*.json")) | set(RUN.glob("policy*/manifest.json"))
                              | set(RUN.glob("controller*.json")) | set(RUN.glob("controller*/manifest.json")))
        for path in policy_files:
            show_json(path)
        policy_results = sorted(RUN.glob("policy*.jsonl")) + sorted(RUN.glob("policy*/results.jsonl"))
        policy_results += sorted(RUN.glob("controller*.jsonl")) + sorted(RUN.glob("controller*/results.jsonl"))
        for path in policy_results:
            rows = read_jsonl(path)
            print("Plik prób:", path, "SHA-256:", sha(path), "wiersze:", len(rows))
            if rows and all("result" in row for row in rows):
                display(pd.DataFrame([{"record_id": row.get("record_id"),
                                       "status": (row.get("result") or {}).get("status"),
                                       "osiągnięto": (row.get("result") or {}).get("achieved"),
                                       "akcje": len((row.get("result") or {}).get("actions", []))}
                                      for row in rows]))
                for row in rows:
                    result = row.get("result") or {}
                    for step in result.get("steps", []):
                        print("\\nPróba", row.get("record_id"), "krok", step.get("index"))
                        print("Pełny prompt polityki:\\n", step.get("prompt"))
                        print("Odpowiedź:", step.get("raw_response"), "akcja:", step.get("action"),
                              "no-op:", step.get("no_op"))
                        print("Stan po ruchu:\\n" + "\\n".join(step.get("after") or []))
        if not policy_results:
            print("Brak zapisanych prób kontrolera.")
        for path in sorted(RUN.glob("controller_assessment*/summary.json")):
            outcome = read_json(path, {}) or {}
            display(Markdown("### " + path.parent.name))
            print("Podsumowanie:", path, "SHA-256:", sha(path))
            print("Mianowniki:", (outcome.get("config") or {}).get("denominators"))
            if outcome.get("results"):
                display(pd.DataFrame(outcome["results"]))
            if outcome.get("paired"):
                display(pd.DataFrame(outcome["paired"]))
            if outcome.get("curves"):
                display(pd.DataFrame(outcome["curves"]))
            for figure in sorted(path.parent.glob("controller_budget_*.png")):
                display(Image(filename=str(figure)))
        '''),
        markdown("""
        ## Pełne prompty generatorów

        Poniżej znajdują się kompletne wiadomości i wyrenderowane prompty
        zapisane w tym przebiegu, z liczbą tokenów i hashem każdego pliku.
        Brak pliku oznacza brak zapisanego promptu, nie wynik modelu.
        """),
        code('''
        prompt_files = sorted(path for path in RUN.rglob("*.prompts.jsonl") if "input_snapshot" not in path.parts)
        print("Liczba plików promptów:", len(prompt_files))
        for path in prompt_files:
            rows = read_jsonl(path)
            print("\\nPlik:", path, "SHA-256:", sha(path), "prompty:", len(rows))
            for row in rows:
                print("\\nKontekst:", row.get("record_id"), "tokeny:", row.get("prompt_tokens"))
                print("Wiadomości:", json.dumps(row.get("messages"), ensure_ascii=False, indent=2))
                print("Wyrenderowany prompt:\\n", row.get("rendered_prompt"))
        '''),
        markdown("""
        ## Czas, koszt i odtworzenie

        Sam notebook nie wywołuje API. Rzeczywiste uruchomienia, ich polecenia,
        limity i statusy pochodzą z manifestu supervisora; brak pomiaru pamięci
        lub czasu nie jest zerem. Stanowisko zgłoszone do badania: Dell XPS 15
        9520, i9-12900HK, 64 GB RAM, RTX 3050 Laptop 4 GB w WSL. Wyniki i
        wykresy zapisano także jako JSON/CSV i PNG/PDF/SVG, gdy etap oceny
        został ukończony.

        Komenda do odtworzenia notebooka z zapisanych plików, bez GPU i API:

        ```bash
        uv run --offline --no-sync python scripts/build_representation_report.py
        ```
        """),
        code('''
        stages = [{key: item.get(key) for key in ("name", "status", "exit_code", "started_utc", "finished_utc",
                                                    "timeout_seconds", "command", "output")}
                  for item in state.get("stages", [])]
        display(pd.DataFrame(stages) if stages else "Brak zapisanych etapów.")
        print("Polityka kosztów w planie:", study.get("limits"))
        for path in sorted(RUN.glob("*benchmark*.json")) + sorted(RUN.glob("*resource*.json")):
            show_json(path)
        decisions = RUN / "decisions.log"
        if decisions.exists():
            print("Dziennik decyzji:\\n", decisions.read_text(encoding="utf-8"))
        '''),
        markdown("""
        ## Tekst do rozważenia w proposal

        Akapit poniżej jest opisowy i powstaje tylko dla dostępnych wyników.
        Nie jest automatycznie wstawiany do proposal ani interpretowany jako
        wynik wykonania intencji przez kontroler.
        """),
        code('''
        latest = assessment_directories[-1] if assessment_directories else None
        summary = read_json(latest / "summary.json", {}) if latest else {}
        rows = (summary or {}).get("results", [])
        completed = [row for row in rows if row.get("outputs") == row.get("planned") and row.get("outputs", 0) > 0]
        if completed and min(row.get("boards_observed", 0) for row in completed) >= 5:
            numbers = "; ".join(f"{row['variant']}: {row['semantic_valid']}/{row['outputs']}" for row in completed)
            paragraph = ("In an exploratory Sokoban evaluation, alternative intent representations were compared "
                         "on the same recorded layouts using a frozen generator. Under strict representation parsing "
                         "and current-state validity checks, the observed counts of valid, non-trivial intents were "
                         + numbers + ". These measurements do not establish execution by a controller or transfer "
                         "to independent environments.")
            calibration = read_json(RUN / "policy_calibration_relative_u.json", {}) or {}
            labelled = sum(candidate.get("u") is not None for context in calibration.get("contexts", [])
                           for candidate in context.get("candidates", []))
            if labelled == 1:
                paragraph += (" A separate calibration yielded one policy-relative scale label, which was not "
                              "used to condition generator training. Neither scale conditioning nor complete "
                              "hierarchical search was evaluated.")
            print(paragraph)
        else:
            print("Brak kompletnych wyników dla co najmniej pięciu plansz; akapit nie powstał.")
        '''),
    ]
    result = nbformat.v4.new_notebook(cells=cells)
    result.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    result.metadata["language_info"] = {"name": "python", "version": "3.11"}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run.resolve()
    document = notebook(run)
    shell = InteractiveShell.instance()
    old_directory = Path.cwd()
    try:
        os.chdir(ROOT)
        count = 0
        for cell in document.cells:
            if cell.cell_type != "code":
                continue
            count += 1
            with capture_output() as captured:
                result = shell.run_cell(cell.source)
            error = result.error_before_exec or result.error_in_exec
            if error is not None:
                raise error
            cell.execution_count = count
            cell.outputs = []
            if captured.stdout:
                cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
            if captured.stderr:
                cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
            cell.outputs.extend(nbformat.v4.new_output("display_data", data=o.data, metadata=o.metadata)
                                for o in captured.outputs)
    finally:
        os.chdir(old_directory)
    run.mkdir(parents=True, exist_ok=True)
    destination = run / "representation_report.ipynb"
    descriptor, temporary = tempfile.mkstemp(prefix=".representation-report-", suffix=".ipynb", dir=run)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            nbformat.write(document, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(destination)


if __name__ == "__main__":
    main()
