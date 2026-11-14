#!/usr/bin/env python3
"""Build and execute an offline report for the Sokoban LoRA matrix."""

import argparse
import os
import tempfile
from pathlib import Path
from textwrap import dedent

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "analysis/lora_matrix_20261114T065804Z"
DEFAULT_ORIGINAL = ROOT / "analysis/intent_generator_20261114T032232Z"


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbformat.v4.new_code_cell(dedent(source).strip())


def notebook(run: Path, original: Path):
    run_name = str(run.relative_to(ROOT))
    original_name = str(original.relative_to(ROOT))
    cells = [
        markdown("""
        # Sokoban: porównanie wariantów LoRA

        Porównujemy rangi adaptera LoRA dla tego samego małego modelu
        Qwen3-0.6B i tych samych danych treningowych. Notebook
        odczytuje wyłącznie lokalnie zapisane wyniki i nie uruchamia GPU ani API.
        Brakujące lub przerwane etapy są pokazywane jako brak danych.
        """),
        code(f'''
        from pathlib import Path
        import json
        import sys
        import pandas as pd
        from IPython.display import Code, Image, Markdown, display

        here = Path.cwd().resolve()
        ROOT = next((p for p in (here, *here.parents)
                     if (p / "pyproject.toml").is_file() and (p / "carl").is_dir()), None)
        if ROOT is None:
            raise RuntimeError("Uruchom notebook z katalogu checkoutu CaRL")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        RUN = ROOT / {run_name!r}
        ORIGINAL = ROOT / {original_name!r}
        def read_json(path, default=None):
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
        def read_jsonl(path):
            if not path.exists():
                return []
            rows = []
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    print("Pominięto niepełny wiersz:", path)
            return rows
        plan = read_json(RUN / "experiment_plan.json", {{}})
        supervisor = read_json(RUN / "run_manifest.json", {{}})
        print("Katalog wyników:", RUN)
        print("Stan nadzoru:", supervisor.get("status", "nieznany"))
        display(Code(json.dumps(plan, ensure_ascii=False, indent=2), language="json"))
        '''),
        markdown("""
        ## Dane, podział i ograniczenia

        Używamy tych samych bazowych wag, zapisanego szablonu rozmowy, zakresu
        kroków i zamrożonych wejść dla porównywanych wariantów. O ile plan nie
        mówi inaczej, LoRA trenuje na intencjach z dokładnie sprawdzonych
        fragmentów trajektorii. Do promptu generatora trafia tylko bieżąca plansza
        i cel zadania: późniejsza plansza, ruchy i offset służą wyłącznie audytowi.

        Pierwotny test to 100 kontekstów z 20 syntetycznych plansz 12×12×10.
        Był on już oceniony dla modelu bazowego, r4 i ICL przed decyzją o
        porównaniu rang r2/r8. Wyniki nowego porównania rang są więc eksploracyjne,
        a nie wynikiem świeżego, zaślepionego testu.
        Drugi test zawiera po 20 plansz 12×12×4 i 12×12×10. Jest eksploracyjny:
        wybór ustalono po treningu r8 i podczas treningu r2, ale przed obejrzeniem
        predykcji na tym zbiorze. Część plansz ×10
        występuje w obu testach, więc nie wolno sumować ich mianowników jako
        niezależnych przypadków. Plansze ×4 i ×10 różnią się także pochodzeniem;
        historia wcześniejszej ekspozycji może być niepełna. W wykonanym
        przebiegu zarówno benchmark, jak i `train()` sprawdziły długości
        tokenów etykiet testowych. Etykiety testowe nie trafiły do gradientów
        ani do wyboru checkpointu, ale pomiar długości dał pośredni wgląd w
        statystyki testu przy doborze limitu 512 tokenów. Aktualny kod pomija
        etykiety testowe podczas obu tych operacji.
        """),
        code('''
        datasets = {
            "pierwotny": (ORIGINAL / "dataset.jsonl", ORIGINAL / "frozen_evaluation_contexts.json"),
            "×4 i ×10": (RUN / "eval_transfer/dataset.jsonl", RUN / "eval_transfer/test_ids.json"),
        }
        selected = {}
        rows = []
        splits = []
        for label, (data_path, ids_path) in datasets.items():
            all_rows = read_jsonl(data_path)
            for row in all_rows:
                splits.append({"zbiór": label, "podział": row.get("split"),
                               "plansza": row.get("layout_id"),
                               "adnotacja": (row.get("annotation_provenance") or {}).get("kind")})
            ids = (read_json(ids_path, {}) or {}).get("record_ids", [])
            by_id = {r["record_id"]: r for r in all_rows if r.get("split") == "test"}
            records = [by_id[i] for i in ids if i in by_id]
            selected[label] = records
            for r in records:
                family = r.get("evaluation_family") or r.get("family") or "syntetyczne 12×12×10"
                rows.append({"zbiór": label, "rodzina": family, "plansza": r["layout_id"], "kontekst": r["record_id"]})
            print(label, "zamrożone ID:", len(ids), "odnalezione rekordy:", len(records))
        frame = pd.DataFrame(rows)
        if not frame.empty:
            display(frame.groupby(["zbiór", "rodzina"]).agg(konteksty=("kontekst", "size"), plansze=("plansza", "nunique")))
            by_group = {label: {r["layout_id"] for r in group} for label, group in selected.items()}
            print("Wspólne plansze obu testów:", len(by_group["pierwotny"] & by_group["×4 i ×10"]))
        split_frame = pd.DataFrame(splits)
        if not split_frame.empty:
            display(split_frame.groupby(["zbiór", "podział"], dropna=False).agg(
                rekordy=("plansza", "size"), plansze=("plansza", "nunique")))
            display(split_frame.groupby(["zbiór", "podział", "adnotacja"], dropna=False).size().rename("rekordy"))
        print("Manifest danych treningowych:")
        display(Code(json.dumps(read_json(RUN / "train_local_only.manifest.json", {}), ensure_ascii=False, indent=2), language="json"))
        '''),
        markdown("""
        ## Niezależny audyt danych i ciągów ruchów

        Osobny skrypt sprawdził predykcje trzech rang na teście transferowym
        oraz odtworzył znalezione ciągi w środowisku CaRL. Liczby i hashe niżej
        pochodzą z zapisanego audytu. W tej próbce odtwarzane intencje miały puste
        warunki zachowania, więc audyt nie dowodzi działania niepustego `preserve`.
        """),
        code('''
        import hashlib
        audit_path = RUN / "independent_transfer_audit.json"
        audit = read_json(audit_path)
        if audit:
            print("Ścieżka:", audit_path)
            print("SHA-256:", hashlib.sha256(audit_path.read_bytes()).hexdigest())
            display(Code(json.dumps(audit, ensure_ascii=False, indent=2), language="json"))
            checks = []
            for relative, expected in audit.get("input_sha256", {}).items():
                path = ROOT / relative
                actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
                checks.append({"plik": relative, "hash zgodny": actual == expected and actual is not None})
            display(pd.DataFrame(checks))
        else:
            print("Brak zapisanego niezależnego audytu transferu.")
        '''),
        markdown("""
        ## Warianty i trening

        Warianty z identyczną liczbą kroków różnią się podaną w planie rangą
        adaptera. Zaplanowano także ablację źródła etykiet `r4_local_100` i
        przygotowano jej dane, ale trening nie wystartował; wynik pozostaje
        nieoceniony. Późniejszy checkpoint wcześniejszej LoRA jest punktem
        odniesienia z innym budżetem treningu. Ewentualne `r4_20` i `r4_last300`
        są opisowymi przekrojami krzywej wybranymi po fakcie, a nie checkpointami
        wybranymi na walidacji. Główne porównanie rang dotyczy stałych 100 kroków.
        Krzywe opisują stratę odpowiedzi i walidację, ale strata nie
        jest główną miarą jakości intencji. Każdą faktycznie zapisaną wartość
        pokazujemy bez wygładzania.
        """),
        code('''
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "svg.fonttype": "none"})
        fig_dir = RUN / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        sources = [("wcześniejsza r4", ORIGINAL / "adapter")]
        sources += [(p.name.removeprefix("adapter_"), p) for p in sorted(RUN.glob("adapter_*")) if p.is_dir()]
        curves = []
        training_info = []
        for name, directory in sources:
            curve = read_jsonl(directory / "training_curve.jsonl")
            for item in curve:
                curves.append({"wariant": name, **item})
            training_info.append({"wariant": name, "kroki zapisane": max((r.get("step", 0) for r in curve), default=0),
                                  "best": read_json(directory / "best.json"),
                                  "status": read_json(directory / "status.json"),
                                  "config": read_json(directory / "configuration.json")})
        display(pd.DataFrame(training_info))
        curve_df = pd.DataFrame(curves)
        if not curve_df.empty:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
            for name, group in curve_df.groupby("wariant", sort=False):
                if "train_loss" in group:
                    train = group.dropna(subset=["step", "train_loss"])
                    axes[0].plot(train["step"], train["train_loss"], label=name, linewidth=1)
                if "validation_loss" in group:
                    val = group.dropna(subset=["step", "validation_loss"])
                    if not val.empty:
                        axes[1].plot(val["step"], val["validation_loss"], marker="o", label=name, linewidth=1)
            for ax, title in zip(axes, ("Trening", "Walidacja")):
                ax.set(title=title, xlabel="Krok", ylabel="Strata tokenów odpowiedzi")
                ax.grid(alpha=.2)
                if ax.lines:
                    ax.legend(fontsize=8)
            for extension in ("png", "pdf", "svg"):
                fig.savefig(fig_dir / f"training_curves.{extension}", dpi=180)
            plt.close(fig)
            display(Image(filename=str(fig_dir / "training_curves.png")))
            curve_df.to_csv(fig_dir / "training_curves.csv", index=False)
        else:
            print("Brak zapisanych krzywych treningu.")
        '''),
        markdown("""
        ## Sprzęt i rzeczywisty pomiar zasobów

        Stanowisko to Dell XPS 15 9520, Intel i9-12900HK, 64 GB RAM i RTX 3050
        Laptop 4 GB pod WSL. Poniżej pokazujemy zapisany benchmark r8 z pełnym
        krokiem optymalizacji i generowaniem oraz szczyty pamięci zakończonych
        treningów. Są to pomiary tego przebiegu, nie prognoza przyszłej prędkości.
        """),
        code('''
        benchmark = read_json(RUN / "benchmark_r8.json")
        display(Code(json.dumps(benchmark, ensure_ascii=False, indent=2), language="json")
                if benchmark else "Brak zapisanego benchmarku r8")
        peaks = []
        for rank in (2, 8):
            status = read_json(RUN / f"adapter_r{rank}_lr2e4/status.json", {})
            def gib(key):
                value = status.get(key)
                return round(value / 2**30, 3) if isinstance(value, (int, float)) else None
            peaks.append({"ranga": rank, "status": status.get("status"), "kroki": status.get("completed_steps"),
                          "GPU allocated [GiB]": gib("gpu_peak_allocated_bytes"),
                          "GPU reserved [GiB]": gib("gpu_peak_reserved_bytes"),
                          "RAM szczyt [GiB]": gib("host_peak_rss_bytes")})
        display(pd.DataFrame(peaks))
        '''),
        markdown("""
        ## Wyniki i mianowniki

        Ścisły JSON oraz pomocnicze usunięcie jednej kompletnej otoczki Markdown
        są liczone oddzielnie. Poprawna, nietrywialna intencja przechodzi
        deterministyczne sprawdzenie na bieżącej planszy. Podajemy też liczbę
        odrębnych intencji, typy warunków, liczbę warunków w koniunkcji oraz
        przypadki żądające od razu pełnego celu planszy. Przedziały są liczone
        przez bootstrap niezależnych plansz, nie pojedynczych fragmentów.

        `local_rule` to ręcznie zaprogramowana reguła, nie model językowy:
        szuka dostępnego dojścia i legalnego pchnięcia skrzyni, preferując pole
        celu. Korzysta tylko z bieżącej planszy, a nie z przyszłej demonstracji.
        To silny punkt odniesienia dla krótkich intencji; nie mierzy rozwiązania
        całego Sokobana ani zużycia tokenów. Brak pomiaru tokenów pozostaje null.

        Jeżeli uruchomiono BFS, wynik oznacza jedynie znalezienie ciągu ruchów
        w podanych granicach. Limit wyszukiwania to wynik nierozstrzygnięty;
        nie jest to pomiar działania Luny ani dowód niemożliwości. Pierwotne
        wyniki i analiza po usunięciu otoczki nie są tym samym protokołem.
        `bfs_attempted` liczy faktyczne uruchomienia przeszukiwania na poprawnych
        intencjach; `pipeline_evaluated_for_bfs` to liczba ocenionych odpowiedzi,
        także tych z niepoprawnym formatem, bez błędów weryfikacji. Ten drugi
        mianownik pokazuje działanie całego procesu, a nie skuteczność samego BFS.

        Porównanie ICL na drugim zbiorze nie zostało wykonane: manifest wejść
        nie przeszedł kontroli zgodności, a limit czasu nie pozwolił na
        powtórzenie. Wynik pozostaje brakujący, nie jest zerem.

        Oddzielny [notebook eksperymentu z Luną](../luna_intent_20261114T060730Z/expanded/luna_intent_report.ipynb)
        pokazuje wykonanie intencji pochodzących z demonstracji. Te próby nie
        oceniały intencji wygenerowanych przez porównywane tu adaptery Qwena.
        """),
        code('''
        assessments = {}
        for label, relative in (("Pierwotny test ×10", "assessment_main"),
                                ("Osobna ocena ×4 i ×10", "assessment_transfer")):
            directory = RUN / relative
            summary = read_json(directory / "summary.json")
            if not summary:
                directory = RUN / (relative + "_partial")
                summary = read_json(directory / "summary.json")
            assessments[label] = (directory, summary)
            display(Markdown("### " + label))
            if not summary:
                print("Brak zapisanego podsumowania; część ocen mogła nie wystartować lub trwać.")
                continue
            if directory.name.endswith("_partial"):
                print("Wczesny zapis częściowy — wyniki kolejnych procesów mogą jeszcze nie być ujęte.")
            display(pd.DataFrame(summary.get("results", [])))
            families = summary.get("family_results", [])
            if families:
                display(pd.DataFrame(families))
            print("Sparowane różnice poprawności, bootstrap po planszach:")
            display(pd.DataFrame(summary.get("paired_strict", [])))
            print("Porównywalność ustawień i hashe wejścia:")
            display(Code(json.dumps(summary.get("config", {}), ensure_ascii=False, indent=2), language="json"))
            for image in sorted(directory.glob("validity_*.png")):
                display(Image(filename=str(image)))
            for image in sorted(directory.glob("bounded_bfs_*.png")):
                display(Image(filename=str(image)))
        '''),
        markdown("""
        ## Akapit do rozważenia w proposal

        Poniższy opis po angielsku pojawia się tylko dla ukończonych wyników
        pierwotnego testu obu wskazanych adapterów. Nie zmienia proposal i nie
        interpretuje poprawności intencji jako wykonania przez kontroler.
        """),
        code('''
        main_summary = (assessments.get("Pierwotny test ×10") or (None, None))[1] or {}
        by_variant = {r["variant"]: r for r in main_summary.get("results", [])}
        reference, candidate = by_variant.get("r4_100"), by_variant.get("r8_100")
        if (reference and candidate and reference["outputs"] == reference["planned"]
                and candidate["outputs"] == candidate["planned"] and reference["outputs"] > 0
                and reference["boards_observed"] >= 5 and candidate["boards_observed"] >= 5):
            print(f"In an exploratory evaluation on {reference['boards_observed']} synthetic Sokoban layouts, "
                  f"the rank-four adapter produced {reference['strict_valid']}/{reference['outputs']} valid, "
                  f"non-trivial intents under strict JSON parsing, whilst the rank-eight adapter produced "
                  f"{candidate['strict_valid']}/{candidate['outputs']}. Intents were checked against the "
                  f"current board, and these measurements do not establish controller achievement or transfer.")
        else:
            print("Brak kompletnych porównywalnych wyników obu wariantów; akapit nie powstał.")
        '''),
        markdown("""
        ## Sparowane różnice rang

        Porównujemy r8−r4 i r2−r4 wyłącznie na tych samych ocenionych kontekstach,
        bez wybierania ich po wyniku. Średnią liczymy najpierw dla planszy, a
        przedział 95% pochodzi z bootstrapu plansz. Przy mniej niż pięciu planszach
        wynik jest wyłącznie opisowy. Szeroki przedział nie rozstrzyga przewagi.
        """),
        code('''
        for label, relative in (("Pierwotny test ×10", "paired_main"), ("Ocena ×4 i ×10", "paired_transfer")):
            directory = RUN / relative
            result = read_json(directory / "paired_ranks.json")
            display(Markdown("### " + label))
            if not result:
                print("Brak wyliczonej analizy sparowanych rang.")
                continue
            display(pd.DataFrame(result.get("results", [])))
            if (directory / "paired_ranks.png").exists():
                display(Image(filename=str(directory / "paired_ranks.png")))
        '''),
        markdown("""
        ## Konkretne przykłady

        Pokazujemy pierwsze w kolejności zamrożone przypadki każdej rodziny,
        bez wybierania ich po sukcesie. Plansza z demonstracji jest pokazana
        osobno; nie trafiała do modelu. Gdy jest dostępny wynik BFS, zapisany
        osiągnięty stan pochodzi z tego przeszukiwania, nie z Luny.
        """),
        code('''
        for label, (directory, summary) in assessments.items():
            records = read_jsonl(directory / "assessment.jsonl")
            if not records:
                continue
            selected_examples = {}
            for row in records:
                key = (row.get("family"), row.get("record_id"))
                if row.get("status") == "evaluated" and row.get("family") not in selected_examples:
                    selected_examples[row.get("family")] = row.get("record_id")
            for family, identifier in selected_examples.items():
                display(Markdown(f"### {label} — {family} — {identifier}"))
                examples = [r for r in records if r.get("record_id") == identifier and r.get("status") == "evaluated"]
                if not examples:
                    continue
                print("Aktualna plansza:\\n" + "\\n".join(examples[0].get("start_observation", [])))
                for row in examples:
                    print("\\nWariant:", row["variant"], "strict:", row["strict_valid"],
                          "po usunięciu otoczki:", row["fence_valid"])
                    print("Pełna odpowiedź:", row.get("raw_output"))
                    print("Intencja:", json.dumps(row.get("fence_intent"), ensure_ascii=False))
                    print("Uwagi walidacji:", row.get("strict_errors"))
                    status = (row.get("solver") or {}).get("status", "nie uruchomiono")
                    labels = {"witness_found": "znaleziono ciąg ruchów", "node_limit": "limit węzłów",
                              "depth_limit": "limit głębokości", "time_limit": "limit czasu",
                              "verification_error": "błąd weryfikacji", "exhausted": "wyczerpano przestrzeń"}
                    print("BFS:", labels.get(status, status))
                    solver = row.get("solver") or {}
                    if status == "witness_found":
                        names = ("góra", "dół", "lewo", "prawo")
                        actions = solver.get("witness_actions") or []
                        print("Ruchy BFS:", [names[action] for action in actions])
                        print("Plansza osiągnięta przez BFS:\\n" +
                              "\\n".join(solver.get("reached_observation") or []))
                print("\\nPóźniejsza plansza demonstracji — nie wynik modelu:\\n" +
                      "\\n".join(examples[0].get("demonstration_later_observation") or []))
        '''),
        markdown("""
        ## Pełne prompty

        Poniżej zapisujemy kompletny prompt bazowy dla każdego zamrożonego
        kontekstu z obu zbiorów. Wszystkie warianty LoRA używają tego samego
        szablonu; ewentualne osobne pliki promptów ICL wypisujemy później.
        `effects` to techniczna nazwa pola opisującego intencję.
        """),
        code('''
        import importlib.util
        snapshot = RUN / "code_snapshot/intent_generator_experiment.py"
        if snapshot.exists():
            specification = importlib.util.spec_from_file_location("lora_recorded_generator", snapshot)
            if specification is None or specification.loader is None:
                raise RuntimeError("Nie można wczytać snapshotu generatora")
            recorded = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(recorded)
            SYSTEM, prompt_messages = recorded.SYSTEM, recorded.prompt_messages
            print("Prompty odtwarzane ze snapshotu:", snapshot)
        else:
            from carl.icl.intent_generator_experiment import SYSTEM, prompt_messages
            print("Brak snapshotu generatora — użyto aktualnej wersji checkoutu.")
        print("Pełny prompt systemowy:")
        display(Code(SYSTEM, language="text"))
        for label, records in selected.items():
            print("\\nZbiór:", label, "liczba promptów:", len(records))
            for record in records:
                print("\\nKontekst:", record["record_id"])
                print(json.dumps(prompt_messages(record), ensure_ascii=False, indent=2))
        icl_prompt_files = sorted(RUN.rglob("*.prompts.jsonl"))
        original_icl = ORIGINAL / "icl_followup/predictions.prompts.jsonl"
        if original_icl.exists():
            icl_prompt_files.append(original_icl)
        for path in icl_prompt_files:
            print("\\nPełne zapisane prompty ICL:", path)
            for row in read_jsonl(path):
                print(json.dumps(row, ensure_ascii=False))
        '''),
        markdown("""
        ## Ograniczenia i odtworzenie

        Jeden pomiar na kontekst nie opisuje zmienności między losowaniami,
        a pojedyncze fragmenty tej samej planszy nie są niezależnymi planszami.
        Wybór checkpointu opiera się na małym zbiorze walidacyjnym. Nowe
        eksperymenty w tym katalogu wykorzystują lokalne obliczenia; zapisane
        rezerwacje wcześniejszych żądań API nie są nowym wydatkiem.

        Odświeżenie notebooka bez GPU i bez API, z katalogu CaRL:

        ```bash
        .venv/bin/python scripts/build_lora_matrix_notebook.py
        ```

        Dokładne polecenia etapów, ich statusy i limity są zapisane w
        `run_manifest.json`; odpowiedzi modeli pozostają w `eval_main/` i `eval_transfer/`,
        konfiguracje i metryki w katalogach `assessment_main` oraz
        `assessment_transfer`. Wykresy zapisano jako PNG, PDF, SVG i CSV.

        W dzienniku oryginalnego procesu interpreter może wskazywać tymczasowy
        cache uv, który po zakończeniu zniknie. Zamiast tej ścieżki użyj bieżącego
        środowiska i kodu zgodnego ze snapshotem. Przykład powtórzenia oceny
        istniejącego checkpointu r4 na zbiorze ×4/×10 (wynik w nowym katalogu):

        ```bash
        uv run --offline --no-sync --with peft==0.21.3 python -m carl.icl.intent_generator_experiment evaluate \\
          --data analysis/lora_matrix_20261114T065804Z/eval_transfer/dataset.jsonl \\
          --test-ids analysis/lora_matrix_20261114T065804Z/eval_transfer/test_ids.json \\
          --output /tmp/carl-r4-transfer-reproduction.jsonl \\
          --model analysis/intent_generator_20261114T032232Z/models/Qwen3-0.6B \\
          --revision c1899de289a04d12100db370d81485cdf75e47ca \\
          --max-length 512 --max-new-tokens 128 \\
          --adapter analysis/intent_generator_20261114T032232Z/adapter/checkpoint-100
        ```
        """),
        code('''
        stages = [{key: stage.get(key) for key in ("name", "status", "exit_code", "started_utc", "finished_utc", "command")}
                  for stage in supervisor.get("stages", [])]
        display(pd.DataFrame(stages) if stages else "Brak zapisanych etapów")
        print("Snapshot kodu użytego przy ewaluacji:")
        display(Code(json.dumps(read_json(RUN / "code_snapshot/manifest.json", {}), ensure_ascii=False, indent=2),
                     language="json"))
        print("Wcześniejszy notebook Qwen/LoRA/ICL:", ORIGINAL / "intent_generator_report.ipynb")
        '''),
    ]
    document = nbformat.v4.new_notebook(cells=cells)
    document.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    document.metadata["language_info"] = {"name": "python", "version": "3.11"}
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    args = parser.parse_args()
    run, original = args.run.resolve(), args.original.resolve()
    document = notebook(run, original)
    shell = InteractiveShell.instance()
    previous_directory = Path.cwd()
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
            for rich in captured.outputs:
                cell.outputs.append(nbformat.v4.new_output("display_data", data=rich.data, metadata=rich.metadata))
    finally:
        os.chdir(previous_directory)
    run.mkdir(parents=True, exist_ok=True)
    destination = run / "lora_matrix_report.ipynb"
    fd, name = tempfile.mkstemp(prefix=".lora-notebook-", suffix=".ipynb", dir=run)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            nbformat.write(document, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)
    print(destination)


if __name__ == "__main__":
    main()
