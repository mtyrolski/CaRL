#!/usr/bin/env python3
"""Build the local, data-backed Sokoban intent experiment notebook."""

import argparse
import os
import tempfile
from pathlib import Path
from textwrap import dedent

import nbformat
from nbclient import NotebookClient


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbformat.v4.new_code_cell(dedent(source).strip())


def notebook(run_relative: str, original_report: str):
    cells = [
        markdown("""
        # Sokoban: generator intencji — dane, LoRA i in-context learning

        Ten notebook jest raportem z małego eksperymentu lokalnego. Pokazuje,
        jakie dane dostał model, jak powstały etykiety, co zmienił trening LoRA,
        co oznacza punkt odniesienia `local_rule` i co rzeczywiście sprawdzono
        na planszach testowych. Wariant in-context learning (ICL) jest pokazywany
        tylko wtedy, gdy jego wyniki zostały zapisane na dysku.

        **Pytanie badawcze:** czy niewielki model może zaproponować poprawną,
        nietrywialną intencję na podstawie aktualnej planszy i zadania? To nie
        jest jeszcze dowód transferu, rozwiązania całej planszy ani skuteczności
        kontrolera wykonującego intencje.

        Komórki czytają lokalne, zapisane dane i nie pobierają modeli, nie
        prowadzą treningu ani nie wysyłają zapytań do zdalnych modeli. Wykresy
        i przykłady są obliczane z tych samych plików wynikowych co raport.
        """),
        code(f'''
        from pathlib import Path
        import json
        import sys
        from collections import Counter

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from IPython.display import Code, Markdown, display
        from matplotlib.colors import ListedColormap

        here = Path.cwd().resolve()
        ROOT = next((p for p in (here, *here.parents)
                     if (p / "pyproject.toml").exists() and (p / "carl").is_dir()), None)
        if ROOT is None:
            raise RuntimeError("Uruchom notebook w checkoutcie CaRL lub jego podkatalogu")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        RUN = ROOT / {run_relative!r}
        if not RUN.is_dir():
            raise FileNotFoundError(RUN)

        def read_json(path):
            return json.loads(path.read_text(encoding="utf-8"))

        def read_jsonl(path):
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

        plt.rcParams.update({{"font.family": "serif", "font.size": 10,
                             "pdf.fonttype": 42, "svg.fonttype": "none"}})
        dataset = read_jsonl(RUN / "dataset.jsonl")
        manifest = read_json(RUN / "dataset_manifest.json")
        strict = read_json(RUN / "assessment/summary.json")
        posthoc = read_json(RUN / "fence_feasibility/summary.json")
        benchmark = read_json(RUN / "qwen3_benchmark.json")
        config = read_json(RUN / "adapter/configuration.json")
        best = read_json(RUN / "adapter/best.json")
        assessment = read_jsonl(RUN / "assessment/assessment.jsonl")
        print("Checkout:", ROOT)
        print("Dane i wyniki:", RUN)
        '''),
        markdown("""
        ## Jak czytać plansze i intencje

        Współrzędne są zerowane i zapisane jako `(wiersz, kolumna)` od lewego
        górnego rogu. `#` to ściana, `-` podłoga, `.` cel, `$` skrzynia,
        `*` skrzynia na celu, `@` gracz, a `+` gracz na celu.

        Intencja jest obiektem JSON. Pole `effects` zawiera warunki, które mają
        być spełnione **jednocześnie**; każdy musi być fałszywy na początku.
        Dozwolone są: skrzynia na podanym polu `box_at`, gracz na podanym polu
        `player_at` oraz co najmniej podana liczba skrzyń na celach
        `boxes_on_goals_at_least`. `preserve` może być puste; jeśli nie jest,
        jego warunki sprawdza się na całym zapisanym fragmencie. `instruction`
        jest opcjonalnym tekstem: checker sprawdza jego typ, lecz nie sprawdza
        semantycznie, czy tekst odpowiada predykatom. Poprawność formalna nie
        potwierdza więc poprawności swobodnego opisu. Skrzynie nie mają trwałych
        identyfikatorów.

        Model przewiduje intencję, widząc tylko aktualną planszę i zadanie.
        Późniejsza plansza i zapisane ruchy służą do tworzenia i kontroli danych,
        ale **nie są wejściem** generatora dla ocenianego przypadku.
        """),
        markdown("""
        ## Źródła, podział i kontrola danych

        Źródłem są istniejące syntetyczne trajektorie `proposal_gamma2_12x12x10`:
        plansze 12 × 12, po 10 skrzyń i 10 pól docelowych. Nie jest to niezależny
        benchmark naturalny; wcześniejszego kontaktu z tą kohortą nie można
        wykluczyć. Plansze oraz powiązane symetrie rozdzielono **przed** wyborem
        fragmentów. Pięć kontekstów z jednej planszy nie stanowi pięciu niezależnych plansz.

        Kolejny manifest zawiera ścieżki i SHA-256 źródeł, długości trajektorii,
        podział oraz audyt wcześniejszej ekspozycji. Trajektorie odtworzono
        w środowisku, odróżniając pchnięcia od chodzenia; fragmenty po pierwszym
        stanie końcowym wyłączono. Kodowanie ruchów: `0` góra, `1` dół,
        `2` lewo, `3` prawo. Odstęp `k` pochodzi z demonstracji i nie oznacza
        zmierzonego czasu wykonania intencji przez kontroler.
        """),
        code('''
        counts = []
        for split, stats in manifest["counts"].items():
            counts.append({"Podział": split, "Plansze": stats["boards"],
                           "Fragmenty": stats["records"], "Rodzaje intencji": stats["effect_types"]})
        display(pd.DataFrame(counts).set_index("Podział"))
        provenance = Counter(row["annotation_provenance"]["kind"] for row in dataset)
        print("Pochodzenie etykiet:", dict(provenance))
        print("Odrzucone rekordy:", manifest["rejected_records"])
        print("Przykładowe źródła i długości trajektorii:")
        display(pd.DataFrame(manifest["trajectories"])[
            ["trajectory_id", "split", "source_path", "source_sha256", "primitive_actions", "push_actions"]
        ].head(6))
        frozen_ids = read_json(RUN / "frozen_evaluation_contexts.json")["record_ids"]
        by_id = {row["record_id"]: row for row in dataset}
        context_keys = {(tuple(by_id[record_id]["start_observation"]), by_id[record_id]["task_goal"])
                        for record_id in frozen_ids}
        print("Pierwotna ocena:", len(frozen_ids), "rekordów,", len(context_keys),
              "unikalnych wejść,", len({by_id[record_id]["layout_id"] for record_id in frozen_ids}), "plansz.")
        extended_manifest = RUN / "icl_followup/extended/manifest.json"
        if extended_manifest.exists():
            print("Manifest dodatkowych kontekstów:")
            display(Code(extended_manifest.read_text(encoding="utf-8"), language="json"))
        '''),
        markdown("""
        Większość etykiet wyznaczono lokalnie z dokładnych różnic zapisanych
        plansz. 15 historycznych adnotacji Terry pochodziło ze współdzielonego
        kontekstu; nie są to nowe niezależne zapytania. W tym przebiegu nie
        zaakceptowano żadnej świeżej adnotacji izolowanej. Odrzucone rekordy
        i powody znajdują się w `rejected.jsonl`; szczegóły w manifeście danych.

        Audyt pierwotnych 100 rekordów wykazał **99 unikalnych wejść** na 20
        planszach. Dodatkowe 59 kontekstów pochodzi z tych samych 20 plansz;
        jeżeli zostaną ocenione, jest to analiza eksploracyjna wybrana po
        obejrzeniu pierwotnych wyników, a nie nowy niezależny zbiór plansz.

        Podczas audytu długości tokenizer przetworzył kompletne przykłady
        wszystkich podziałów, w tym etykiety testowe, wyłącznie po to, aby
        policzyć tokeny. Etykiet testowych nie użyto w przebiegu modelu,
        obliczaniu straty, optymalizacji ani wyborze checkpointu. Przyszła
        plansza nie była wejściem generatora dla ocenianego przypadku.
        """),
        markdown("""
        ## Trzy rzeczywiste fragmenty danych

        Pierwsza plansza to wejście generatora. Druga jest późniejszą obserwacją
        z zapisanej trajektorii i pomaga sprawdzić etykietę: **nie jest** wynikiem
        kontrolera ani planszą podawaną modelowi podczas oceny. Przykłady wybrano
        z danych treningowych według pierwszego wystąpienia różnych rodzajów
        intencji, bez patrzenia na wyniki modeli. Na rysunkach `B` oznacza
        skrzynię, `P` gracza, a żółty kolor pokazuje pole docelowe. Numery
        osi są zerowane i odpowiadają współrzędnym w JSON-ie.
        """),
        code('''
        symbol_colors = {"#": "#29394b", "-": "#f6f2e9", ".": "#f2d77d", "$": "#b87944",
                         "*": "#d1933e", "@": "#6596c2", "+": "#85b2c9"}

        def draw_board(ax, board, title):
            colors = list(symbol_colors.values())
            labels = list(symbol_colors)
            values = np.array([[labels.index(ch) for ch in row] for row in board])
            ax.imshow(values, cmap=ListedColormap(colors), vmin=-0.5, vmax=len(colors)-0.5)
            for r, row in enumerate(board):
                for c, char in enumerate(row):
                    if char in "$*@+":
                        ax.text(c, r, "P" if char in "@+" else "B", ha="center", va="center", fontsize=6)
            ax.set_xticks(range(len(board[0])))
            ax.set_yticks(range(len(board)))
            ax.tick_params(axis="both", length=0, labelsize=5, pad=1)
            ax.xaxis.tick_top()
            ax.set_title(title, fontsize=9)

        chosen, seen = [], set()
        for row in dataset:
            kind = row.get("effect_type")
            if row["split"] == "train" and kind not in seen:
                chosen.append(row)
                seen.add(kind)
            if len(chosen) == 3:
                break
        fig, axes = plt.subplots(len(chosen), 2, figsize=(6.5, 3 * len(chosen)))
        for i, row in enumerate(chosen):
            draw_board(axes[i, 0], row["start_observation"], f"{i + 1}: przed")
            draw_board(axes[i, 1], row["later_observation"], f"{i + 1}: późniejsza demonstracja")
        fig.tight_layout()
        display(fig)
        plt.close(fig)
        for i, row in enumerate(chosen, 1):
            print(f"\\nPrzykład {i}: {row['record_id']} | train | {row['annotation_provenance']['kind']}")
            print("Intencja:", json.dumps(row["intent"], ensure_ascii=False))
            print("Zapisane ruchy demonstracji:", row["fragment_actions"], "| kontrola:", row["validation"])
        '''),
        markdown("""
        ## Pełne prompty i wejście modelu

        Poniżej pokazujemy pełne komunikaty system/user oraz pełny tekst po
        zastosowaniu lokalnego szablonu czatu Qwena. To rekonstrukcja z zapisanego
        kodu i tokenizera; pierwotny plik podsumowania modelu bazowego nie
        zawierał hasha promptu, więc nie twierdzimy, że osobno potwierdzono
        każdy jego bajt historycznie. W ocenie bazowego modelu i adaptera LoRA
        użyto tej samej struktury promptu. Cel treningowy jest pokazany oddzielnie:
        przyszła plansza, akcje, `k`, nazwa pliku i wyniki walidacji nie należą
        do wejścia generatora. Strata dotyczy tylko tokenów odpowiedzi.

        W metadanych `prompt_sha256` jest hashem samej instrukcji systemowej
        `SYSTEM`, a nie całego promptu. Dla ICL osobne `prompts_sha256`
        identyfikuje zapisany plik JSONL z pełnymi promptami. Historyczny wynik
        base nie zapisał ani typu `dtype`, ani hasha instrukcji systemowej.
        """),
        code('''
        from carl.icl.intent_generator_experiment import prompt_messages, target_text
        from transformers import AutoTokenizer

        train_example = chosen[0]
        test_example = next(row for row in dataset if row["split"] == "test")
        tokenizer = AutoTokenizer.from_pretrained(RUN / "models/Qwen3-0.6B", local_files_only=True)
        messages = prompt_messages(test_example)
        print("Pełne wiadomości dla przykładowej planszy testowej:")
        display(Code(json.dumps(messages, ensure_ascii=False, indent=2), language="json"))
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        print("Pełny tekst po zastosowaniu szablonu czatu Qwena:")
        display(Code(rendered, language="text"))
        print("Osobno — pełne wejście dla przykładu treningowego:")
        display(Code(tokenizer.apply_chat_template(prompt_messages(train_example), tokenize=False,
                     add_generation_prompt=True, enable_thinking=False), language="text"))
        print("Cel odpowiedzi treningowej, NIE część wejścia przy generowaniu:")
        display(Code(target_text(train_example), language="json"))
        '''),
        markdown("""
        Dla audytu pokazujemy także pełny tekst żądania wysłanego do CLI pilota
        adnotacyjnego. Nie mamy potwierdzenia pełnego zestawu instrukcji
        systemowych dodanych przez platformę Codex. Adnotator dostaje dwie
        obserwacje, ponieważ jego zadanie jest inne niż zadanie generatora:
        opisuje znaną zmianę w zarejestrowanym fragmencie. Pilot zakończył się
        błędem dostępu dla tej ścieżki logowania; nie jest zaakceptowaną nową
        etykietą Terry. Pokazany wcześniej render lokalnego Qwena jest kompletny
        dla użytego szablonu, z zastrzeżeniem rekonstrukcji historycznego base.
        """),
        code('''
        annotation_prompt = RUN / "terra/batch_0000_attempt_1/request.txt"
        annotation_schema = RUN / "terra/batch_0000_attempt_1/schema.json"
        if annotation_prompt.exists():
            print("Pełny zapisany tekst żądania przekazanego do CLI:")
            display(Code(annotation_prompt.read_text(encoding="utf-8"), language="text"))
            print("Pełny schemat wymaganej odpowiedzi:")
            display(Code(annotation_schema.read_text(encoding="utf-8"), language="json"))
        else:
            print("Brak zapisanego promptu pilota.")
        '''),
        markdown("""
        ## Model, zasoby i trening LoRA

        Model to `Qwen/Qwen3-0.6B`, przypięta rewizja
        `c1899de289a04d12100db370d81485cdf75e47ca`; generowanie jest zachłanne,
        a tryb myślenia wyłączony. Bazę zamrożono; trenowany był adapter LoRA
        rangi 4. Mikro-batch wynosił 1, akumulacja 4, maksymalna długość 512,
        bez obcinania przykładów. Wyboru checkpointu dokonano na walidacji.

        Sprzęt i wersje: i9-12900HK, RTX 3050 Ti Laptop GPU 4 GB, Python 3.11.4,
        Torch 2.10.0+cu128, Transformers 4.57.6, PEFT 0.21.3. Ponad 4 GB dostępnej
        VRAM nie zakładano. Poniższy wykres straty jest diagnostyką treningu,
        a nie główną miarą jakości ani dowodem generalizacji. Bez wygładzania;
        ten sam wykres jest zapisywany także jako `figures/lora_training_curve`
        w formatach PDF, SVG i PNG.
        """),
        code('''
        lengths = config["token_lengths"]
        print("Ukończone kroki: 300; najlepszy checkpoint:", best["step"],
              "| strata walidacyjna:", round(best["validation_loss"], 4))
        print("Pełne przykłady, zakres tokenów:", lengths["min"], "–", lengths["max"],
              "| limit:", lengths["limit"], "| obcięte:", lengths["truncated"])
        print("Pełny krok treningowy:", round(benchmark["complete_step_seconds"], 2), "s",
              "| generowanie:", round(benchmark["generation_tokens_per_second"], 1), "tokenów/s")
        print("Szczyt VRAM allocated/reserved:",
              round(benchmark["gpu_peak_allocated_bytes"] / 2**30, 2), "/",
              round(benchmark["gpu_peak_reserved_bytes"] / 2**30, 2), "GiB")
        curve = read_jsonl(RUN / "adapter/training_curve.jsonl")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot([x["step"] for x in curve], [x["train_loss"] for x in curve], lw=1, label="Trening")
        val = [x for x in curve if "validation_loss" in x]
        ax.plot([x["step"] for x in val], [x["validation_loss"] for x in val], "o", label="Walidacja (co 20 kroków)")
        ax.axvline(best["step"], color="0.4", ls="--", label=f"Wybrany krok {best['step']}")
        ax.set(xlabel="Krok optymalizatora", ylabel="Strata", title="Przebieg treningu LoRA")
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        figure_dir = RUN / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "svg", "png"):
            temporary = figure_dir / f".lora_training_curve.{ext}.tmp"
            fig.savefig(temporary, format=ext, dpi=220, bbox_inches="tight")
            temporary.replace(figure_dir / f"lora_training_curve.{ext}")
        display(fig)
        plt.close(fig)
        '''),
        markdown("""
        ### Czas generowania i pamięć

        Zapisany czas pętli nie obejmuje ładowania modelu. Przepustowość
        poniżej pochodzi z liczby wygenerowanych tokenów podzielonej przez
        zsumowane czasy generowania poszczególnych odpowiedzi. Różnice nie
        są czystym porównaniem szybkości sprzętu: prompty i liczby tokenów
        wyjściowych różniły się między metodami. Jeśli zapisano wznowienie,
        czas pętli może dotyczyć tylko ostatniej części przebiegu.
        """),
        code('''
        inference_files = {
            "Qwen bazowy": (RUN / "base_outputs.summary.json", RUN / "base_outputs.jsonl"),
            "Qwen + LoRA": (RUN / "adapted_outputs.summary.json", RUN / "adapted_outputs.jsonl"),
            "Qwen + ICL": (RUN / "icl_followup/predictions.summary.json", RUN / "icl_followup/predictions.jsonl"),
            "Qwen bazowy — dodatkowe, eksploracyjne": (RUN / "icl_followup/extended/base.summary.json",
                                                       RUN / "icl_followup/extended/base.jsonl"),
            "Qwen + LoRA — dodatkowe, eksploracyjne": (RUN / "icl_followup/extended/adapted.summary.json",
                                                      RUN / "icl_followup/extended/adapted.jsonl"),
            "Qwen + ICL — dodatkowe, eksploracyjne": (RUN / "icl_followup/extended/in_context.summary.json",
                                                     RUN / "icl_followup/extended/in_context.jsonl"),
        }
        inference_rows = []
        def gib(value):
            return round(value / 2**30, 2) if isinstance(value, (int, float)) else None

        for name, (summary_file, output_file) in inference_files.items():
            summary_data = read_json(summary_file) if summary_file.exists() else {}
            output_rows = read_jsonl(output_file) if output_file.exists() else []
            tokens = [r.get("generated_tokens") for r in output_rows]
            seconds = [r.get("generation_seconds") for r in output_rows]
            measurable = output_rows and all(isinstance(x, (int, float)) for x in tokens + seconds) and sum(seconds) > 0
            inference_rows.append({"Metoda": name, "Status": summary_data.get("status", "brak podsumowania"),
                                   "Odpowiedzi": len(output_rows), "Czas pętli [s]": summary_data.get("elapsed_seconds"),
                                   "Tokeny/s": round(sum(tokens) / sum(seconds), 2) if measurable else None,
                                   "VRAM allocated [GiB]": gib(summary_data.get("gpu_peak_allocated_bytes")),
                                   "VRAM reserved [GiB]": gib(summary_data.get("gpu_peak_reserved_bytes")),
                                   "Wznowione rekordy": summary_data.get("resumed_records")})
        display(pd.DataFrame(inference_rows).set_index("Metoda"))
        '''),
        markdown("""
        ## Co to jest `local_rule`?

        To **nie jest drugi model językowy**. To lokalny algorytm, który patrzy
        tylko na aktualną planszę. Najpierw wyznacza pola, na które gracz może
        dojść bez ruszania skrzyń. Potem szuka legalnego pchnięcia, preferując
        pole docelowe, mniejszą odległość Manhattan do celu i krótsze dojście.
        Proponuje intencję „skrzynia na tym polu”. Jeśli nie ma pchnięcia,
        może zaproponować osiągalne pole gracza.

        Jest to uprzywilejowany punkt odniesienia: sam oblicza lokalną
        osiągalność i legalne pchnięcia, często proponując prostą zmianę. Jego 100/100
        **nie** oznacza rozwiązania 100 plansz, a jego intencje nie muszą być
        równie trudne ani równie przydatne jak intencje Qwena. JSON tej reguły
        serializuje sam program, dlatego 100/100 poprawnego JSON-a nie mierzy
        stosowania się modelu do instrukcji. Reguła nie korzysta z późniejszej
        planszy demonstracji.
        """),
        markdown("""
        ## Porównanie na tych samych kontekstach testowych

        Główny wynik używa ścisłego parsera JSON. Osobny, późniejszy audyt
        usunął po jednej zewnętrznej ramce Markdown symetrycznie z odpowiedzi
        wszystkich porównywanych modeli; jest to analiza pomocnicza, a nie
        zmiana wyniku głównego.

        Bez wytrenowanego kontrolera sprawdzamy, czy ograniczone BFS znalazło
        ciąg do stanu spełniającego intencję. Limit to głębokość 24, 10 000
        rozwiniętych stanów i 1 s na intencję. Osiągnięcie limitu oznacza
        **nie rozstrzygnięto**, nie „niemożliwe”. Znaleziony ciąg odtworzono
        w środowisku. Przedziały niepewności pochodzą z bootstrapu całych plansz,
        a nie pojedynczych nakładających się fragmentów.

        **Uwaga na nazwę metryki:** „poprawna nietrywialna intencja” oznacza
        formalną poprawność i to, że każdy wymagany warunek jest fałszywy na
        planszy początkowej. Nie dowodzi, że propozycja jest sensownym etapem
        pośrednim: powtórzenie całego zadania może spełnić ten warunek.
        """),
        code('''
        icl_path = RUN / "icl_followup/assessment/summary.json"
        icl = read_json(icl_path) if icl_path.exists() else None

        def comparison(summary, icl_summary=None, include_icl=False):
            rows = list(summary["results"])
            if icl_summary:
                candidates = icl_summary.get("results", [])
                candidate = next((r for r in candidates if r.get("method") in ("in_context", "icl", "few_shot")), None)
                if candidate:
                    rows.append({**candidate, "method": "in_context"})
            if include_icl and not any(row.get("method") == "in_context" for row in rows):
                rows.append({"method": "in_context", "outputs": None, "valid_json": None,
                             "valid_nontrivial": None, "witness_found": None})
            return rows

        def display_comparison(summary, title, icl_summary=None, include_icl=False):
            rows = comparison(summary, icl_summary, include_icl)
            names = {"base": "Qwen bazowy", "adapted": "Qwen + LoRA", "local_rule": "Local rule", "in_context": "Qwen + ICL"}
            colors = {"base": "#8297a8", "adapted": "#4e8a71", "local_rule": "#c18b53", "in_context": "#76539a"}
            def ratio(value, count):
                return f"{value}/{count}" if value is not None and count is not None else "brak danych"
            table = []
            for r in rows:
                n = r.get("outputs")
                table.append({"Metoda": names.get(r["method"], r["method"]), "Konteksty": n,
                              "Plansze": r.get("boards_with_outputs"),
                              "Poprawny JSON": ratio(r.get("valid_json"), n),
                              "Poprawna nietrywialna intencja": ratio(r.get("valid_nontrivial"), n),
                              "Ciąg znaleziony przez BFS": ratio(r.get("witness_found"), n),
                              "Nierozstrzygnięte z poprawnych": r.get("unresolved_among_valid"),
                              "Brak wyniku infrastruktury": r.get("missing_infrastructure_outputs")})
            display(pd.DataFrame(table).set_index("Metoda"))
            details = []
            for r in rows:
                outcomes = r.get("solver_statuses")
                if outcomes is not None:
                    outcomes = {("znaleziono ciąg" if key == "witness_found" else key): value
                                for key, value in outcomes.items()}
                details.append({"Metoda": names.get(r["method"], r["method"]),
                                "Typy warunków intencji": r.get("effect_types", "brak danych"),
                                "Liczba warunków w intencji": r.get("conjunction_sizes", "brak danych"),
                                "Wyniki i limity BFS": outcomes if outcomes is not None else "brak danych",
                                "Rozwinięte stany BFS": r.get("solver_expanded_nodes"),
                                "Łączny czas BFS [s]": r.get("solver_seconds")})
            print("Struktura intencji i koszty wyszukiwania (typy warunków mogą wystąpić razem):")
            display(pd.DataFrame(details).set_index("Metoda"))
            pair_source = icl_summary if icl_summary is not None else summary
            pairs = {key: pair_source[key] for key in ("paired_adapted_minus_base", "paired_in_context_minus_base",
                                                       "paired_in_context_minus_adapted") if key in pair_source}
            if pairs:
                print("Sparowane różnice metod; przedziały liczone przez bootstrap plansz:")
                pair_labels = {"paired_adapted_minus_base": "LoRA − base", "paired_in_context_minus_base": "ICL − base",
                               "paired_in_context_minus_adapted": "ICL − LoRA"}
                pair_columns = {"contexts": "konteksty", "boards": "plansze",
                                "valid_delta": "różnica poprawnych intencji",
                                "valid_delta_95_board_bootstrap": "95% CI poprawnych intencji",
                                "witness_delta": "różnica znalezionych ciągów",
                                "witness_delta_95_board_bootstrap": "95% CI znalezionych ciągów"}
                display(pd.DataFrame(pairs).T.rename(index=pair_labels, columns=pair_columns))
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for ax, field, ci_field, num, label in [
                (axes[0], "board_mean_valid_fraction", "valid_fraction_95_board_bootstrap", "valid_nontrivial", "Poprawna nietrywialna intencja"),
                (axes[1], "board_mean_witness_fraction", "witness_fraction_95_board_bootstrap", "witness_found", "BFS znalazł ciąg ruchów"),
            ]:
                for i, r in enumerate(rows):
                    value = r.get(field)
                    if value is None:
                        ax.text(i, .03, "brak danych", ha="center", rotation=90)
                        continue
                    ax.bar(i, value, color=colors.get(r["method"], "0.6"))
                    ci = r.get(ci_field)
                    if ci is not None:
                        ax.vlines(i, ci[0], ci[1], color="black", lw=1)
                        ax.hlines([ci[0], ci[1]], i-.07, i+.07, color="black", lw=1)
                    ax.text(i, min(1.07, max(value, ci[1] if ci else value) + .025),
                            f"{r.get(num)}/{r.get('outputs')}", ha="center", fontsize=8)
                ax.set_xticks(range(len(rows)), [names.get(r["method"], r["method"]) for r in rows], rotation=20, ha="right")
                ax.set_xlim(-0.6, len(rows) - 0.4)
                ax.set_ylim(0, 1.16)
                ax.set_ylabel("Średni odsetek na planszę")
                ax.set_title(label)
                ax.grid(axis="y", alpha=.2)
                ax.set_axisbelow(True)
            fig.suptitle(title)
            fig.tight_layout()
            display(fig)
            plt.close(fig)

        display_comparison(strict, "Ścisły JSON — wynik główny", icl, include_icl=True)
        '''),
        markdown("""
        Przedstawiony przez `local_rule` odsetek dotyczy prostych intencji
        skonstruowanych z wiedzą o legalnych ruchach. Modele były oceniane na
        własnych propozycjach; różnice ich trudności nie są kontrolowane.
        Wysokość słupka oznacza średnią odsetków na planszach, natomiast
        `x/N` to łączna liczba rekordów. Gdy liczby kontekstów na planszach
        nie są równe, te wartości nie muszą być identyczne.
        """),
        code('''
        icl_posthoc_path = RUN / "icl_followup/fence_feasibility/summary.json"
        icl_posthoc = read_json(icl_posthoc_path) if icl_posthoc_path.exists() else None
        display_comparison(posthoc, "Analiza pomocnicza: usunięto zewnętrzną ramkę Markdown", icl_posthoc,
                           include_icl=True)
        print("Przy ścisłym parsowaniu bazowy Qwen zwrócił 0/100 poprawnych JSON-ów;",
              "wszystkie 100 odpowiedzi miały ramkę Markdown.")
        '''),
        markdown("""
        ### Czy intencja jest etapem, czy powtórzeniem całego zadania?

        Formalnie poprawna nietrywialna intencja może powtarzać całe zadanie:
        umieścić wszystkie skrzynie na celach. W takim przypadku sama poprawność
        nie dowodzi, że model zaproponował użyteczny etap pośredni. Konkretne
        liczby dla zapisanych wyników ICL i LoRA obliczamy poniżej. Pierwotne
        100 rekordów obejmuje 99 unikalnych wejść na 20 planszach; nie są to
        niezależne próby.

        Poniższa tabela liczy osobno intencje **zawierające** warunek
        odpowiadający ułożeniu wszystkich skrzyń oraz intencje **równe** temu
        zadaniu w sprawdzalnym fragmencie JSON-a: jest to ich jedyny warunek,
        a `preserve` jest puste. Opcjonalny swobodny tekst `instruction` nie
        jest tu traktowany jako sprawdzalny warunek. Mianownik stanowią formalnie
        poprawne intencje po usunięciu ramki, a nie wszystkie niezależne plansze.
        """),
        code('''
        goal_rows_path = RUN / "icl_followup/fence_feasibility/feasibility.jsonl"
        if not goal_rows_path.exists():
            goal_rows_path = RUN / "fence_feasibility/feasibility.jsonl"
        goal_rows = read_jsonl(goal_rows_path)
        methods = ("base", "adapted", "in_context")
        goal_comparison = []
        for method in methods:
            evaluated = [row for row in goal_rows if row.get("method") == method and row.get("status") == "evaluated"]
            valid = [row for row in evaluated if row.get("valid_nontrivial") is True]
            contains_goal = 0
            exact_goal = 0
            for row in valid:
                intent = row["intent"]
                box_count = sum(ch in "$*" for board_row in row["start_observation"] for ch in board_row)
                predicates = intent.get("effects", [])
                includes_goal = box_count > 0 and any(
                    predicate.get("predicate") == "boxes_on_goals_at_least" and predicate.get("count") == box_count
                    for predicate in predicates
                )
                contains_goal += int(includes_goal)
                exact_goal += int(includes_goal and len(predicates) == 1 and not intent.get("preserve"))
            goal_comparison.append({"Metoda": method, "Dostępne odpowiedzi": len(evaluated),
                                    "Formalnie poprawne": len(valid),
                                    "Zawiera cały cel": f"{contains_goal}/{len(valid)}" if valid else "brak poprawnych danych",
                                    "Dokładnie cały cel": f"{exact_goal}/{len(valid)}" if valid else "brak poprawnych danych",
                                    "Dokładnie cały cel [%]": round(100 * exact_goal / len(valid), 1) if valid else None})
        display(pd.DataFrame(goal_comparison).set_index("Metoda"))
        icl_valid = [row for row in goal_rows if row.get("method") == "in_context"
                     and row.get("status") == "evaluated" and row.get("valid_nontrivial") is True]
        icl_evaluated = [row for row in goal_rows if row.get("method") == "in_context"
                         and row.get("status") == "evaluated"]
        if icl_evaluated:
            icl_counts = next(row for row in goal_comparison if row["Metoda"] == "in_context")
            found = sum(bool((row.get("solver") or {}).get("witness_found")) for row in icl_evaluated)
            cutoffs = sum((row.get("solver") or {}).get("status") in ("node_limit", "time_limit", "depth_limit")
                          for row in icl_evaluated)
            print("ICL po zdjęciu ramki: dokładnie cały cel w", icl_counts["Dokładnie cały cel"],
                  "formalnie poprawnych odpowiedzi; BFS znalazło ciąg dla", f"{found}/{len(icl_evaluated)}",
                  "odpowiedzi, a limit osiągnięto dla", f"{cutoffs}/{len(icl_evaluated)}" + ".")
            if found and icl_valid and all(
                len(row["intent"].get("effects", [])) == 1 and not row["intent"].get("preserve")
                and row["intent"]["effects"][0].get("predicate") == "boxes_on_goals_at_least"
                and row["intent"]["effects"][0].get("count") == sum(
                    ch in "$*" for board_row in row["start_observation"] for ch in board_row)
                for row in icl_valid
            ):
                print("W znalezionych ciągach dla tych intencji BFS doprowadziło do ułożenia wszystkich skrzyń; "
                      "to wynik BFS, nie wykonanie przez model lub kontroler. Inne metody proponowały "
                      "różne intencje, więc nie jest to porównanie ich skuteczności rozwiązywania plansz.")
                found_boards = {row["layout_id"] for row in icl_evaluated
                                if (row.get("solver") or {}).get("witness_found")}
                print("Znalezione ciągi dotyczą", len(found_boards), "różnych plansz.")
        else:
            print("Brak zapisanych wyników ICL do tej analizy.")
        '''),
        markdown("""
        ### Dodatkowe konteksty na tych samych planszach

        To analiza eksploracyjna: 59 nowych wejść wybrano po obejrzeniu
        pierwszego wyniku. Nie oznaczają one 59 nowych plansz. Metoda bez
        zapisanych wyników nie jest automatycznie liczona jako niepowodzenie.
        """),
        code('''
        extended_path = RUN / "icl_followup/extended/assessment/summary.json"
        extended_posthoc_path = RUN / "icl_followup/extended/fence_feasibility/summary.json"
        if extended_path.exists():
            extended = read_json(extended_path)
            display_comparison(extended, "Dodatkowe konteksty — ścisły JSON", include_icl=True)
        else:
            print("Brak zapisanego podsumowania dodatkowych kontekstów.")
        if extended_posthoc_path.exists():
            extended_posthoc = read_json(extended_posthoc_path)
            display_comparison(extended_posthoc, "Dodatkowe konteksty — po usunięciu ramki Markdown",
                               include_icl=True)
        else:
            print("Brak zapisanej dodatkowej analizy formatowania.")
        '''),
        markdown("""
        ### Analiza czułości: usunięcie zduplikowanego wejścia

        Po zauważeniu jednego duplikatu wybieramy pierwsze wystąpienie według
        zapisanej kolejności testu, bez sprawdzania odpowiedzi modeli. To analiza
        post-hoc 99 wejść; głównych wyników nie zastępuje. Liczby surowe podano
        na dostępnych odpowiedziach, a braki infrastruktury są wyszczególnione.
        Przedziały dla średnich na planszach losują całe plansze, zachowując
        także plansze bez sukcesu.
        """),
        code('''
        unique_ids, keys_seen = [], set()
        for identifier in frozen_ids:
            rec = by_id[identifier]
            key = (tuple(rec["start_observation"]), rec["task_goal"])
            if key not in keys_seen:
                unique_ids.append(identifier)
                keys_seen.add(key)

        def deduplicated_table(path, label):
            if not path.exists():
                print("Brak danych dla:", label)
                return
            data = read_jsonl(path)
            lookup = {(r["method"], r["record_id"]): r for r in data}
            methods = [m for m in ("base", "adapted", "in_context") if any(r["method"] == m for r in data)]
            table = []
            for method in methods:
                records = [lookup[(method, identifier)] for identifier in unique_ids
                           if (method, identifier) in lookup and lookup[(method, identifier)].get("status") == "evaluated"]
                board_values = {}
                for rec in records:
                    success = int(bool((rec.get("solver") or {}).get("witness_found")))
                    board_values.setdefault(rec["layout_id"], []).append(success)
                means = np.array([np.mean(values) for values in board_values.values()])
                if len(means):
                    rng = np.random.default_rng(20261114)
                    samples = rng.choice(means, size=(2000, len(means)), replace=True).mean(axis=1)
                    interval = np.quantile(samples, [.025, .975])
                    board_result = f"{means.mean():.3f} [{interval[0]:.3f}, {interval[1]:.3f}]"
                else:
                    board_result = "brak danych"
                n = len(records)
                table.append({"Metoda": method, "Dostępne odpowiedzi": n, "Planowane": len(unique_ids),
                              "Brak infrastruktury": len(unique_ids) - n,
                              "Poprawny JSON": f"{sum(r['json_valid'] for r in records)}/{n}",
                              "Poprawna intencja": f"{sum(r['valid_nontrivial'] for r in records)}/{n}",
                              "BFS znalazł ciąg": f"{sum(bool((r.get('solver') or {}).get('witness_found')) for r in records)}/{n}",
                              "Średnia po planszach [95% CI]": board_result})
            print(label)
            display(pd.DataFrame(table).set_index("Metoda"))

        strict_rows = RUN / "icl_followup/assessment/assessment.jsonl"
        posthoc_rows = RUN / "icl_followup/fence_feasibility/feasibility.jsonl"
        deduplicated_table(strict_rows if strict_rows.exists() else RUN / "assessment/assessment.jsonl", "Ścisły JSON")
        deduplicated_table(posthoc_rows if posthoc_rows.exists() else RUN / "fence_feasibility/feasibility.jsonl", "Po zdjęciu zewnętrznej ramki")
        '''),
        markdown("""
        ## Przykład działania znalezionego ciągu

        Wybieramy pierwszy testowy przypadek, w którym dla intencji LoRA BFS
        znalazło ciąg. Poniższe plansze otrzymujemy przez **ponowne odtworzenie
        zapisanych ruchów w środowisku**. To pokaz algorytmu BFS, nie działanie
        wyuczonego kontrolera. Przyszły stan demonstracji pokazano oddzielnie.
        """),
        code('''
        from carl.icl.boards import from_rows, to_rows
        from carl.icl.data_sources.sokoban import environment
        from carl.icl.intent_data import predicate_value

        example = next(r for r in assessment if r.get("method") == "adapted" and (r.get("solver") or {}).get("witness_found"))
        case_id = example["record_id"]
        same = [r for r in assessment if r["record_id"] == case_id]
        followup_rows = RUN / "icl_followup/assessment/assessment.jsonl"
        icl_examples = []
        if followup_rows.exists():
            icl_examples = [r for r in read_jsonl(followup_rows) if r["method"] == "in_context"]
            same.extend(r for r in icl_examples if r["record_id"] == case_id)
        if not any(row["method"] == "in_context" for row in same):
            print("Brak zapisanej odpowiedzi ICL dla tego przypadku.")
        for row in same:
            print("\\nMetoda:", row["method"], "| status:", row.get("status"),
                  "| poprawna:", row.get("valid_nontrivial"), "| błędy:", row.get("validation_errors"))
            print("Pełna odpowiedź:", row.get("raw_output", "brak zapisanego wyniku"))
            search_status = (row.get("solver") or {}).get("status", "nie uruchomiono")
            print("Status BFS:", {"witness_found": "znaleziono ciąg", "node_limit": "limit stanów",
                                  "time_limit": "limit czasu", "depth_limit": "limit głębokości",
                                  "exhausted": "przeszukiwanie zakończone"}.get(search_status, search_status))

        def show_search_sequence(record, label):
            print("\\nPrzypadek:", record["record_id"], "|", label)
            print("Intencja:", json.dumps(record["intent"], ensure_ascii=False))
            print("Stan demonstracji (nie wynik kontrolera):")
            print("\\n".join(record["demonstration_later_observation"]))
            state = from_rows(record["start_observation"])
            env = environment(state)
            states = [record["start_observation"]]
            actions = record["solver"]["witness_actions"]
            for action in actions:
                state = np.asarray(env.next_state(state, action))
                states.append(to_rows(state))
            assert states[-1] == record["solver"]["reached_observation"]
            print("Ciąg ruchów BFS:", " ".join({0:"↑", 1:"↓", 2:"←", 3:"→"}[a] for a in actions))
            print("Czy ciąg BFS rozwiązał całą planszę:", bool(env.is_solved(state)),
                  "(wynik BFS, nie działania modelu ani kontrolera)")
            print("Sprawdzenie warunków intencji przed → po:")
            for predicate in record["intent"]["effects"]:
                print(predicate, predicate_value(predicate, states[0]), "→", predicate_value(predicate, states[-1]))
            cols = min(4, len(states))
            rows = (len(states) + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows), squeeze=False)
            for i, ax in enumerate(axes.flat):
                if i >= len(states):
                    ax.axis("off")
                    continue
                title = "Start" if i == 0 else f"{i}: " + {0:"↑", 1:"↓", 2:"←", 3:"→"}[actions[i - 1]]
                draw_board(ax, states[i], title)
            fig.suptitle(f"{label}: odtworzony ciąg znaleziony przez BFS", y=1.01)
            fig.tight_layout()
            display(fig)
            plt.close(fig)

        show_search_sequence(example, "LoRA")
        icl_replay = next((row for row in icl_examples if row["record_id"] == case_id
                           and (row.get("solver") or {}).get("witness_found")), None)
        if icl_replay is None:
            icl_replay = next((row for row in icl_examples if (row.get("solver") or {}).get("witness_found")), None)
        icl_replay_label = "ICL, ścisły JSON"
        icl_posthoc_rows = RUN / "icl_followup/fence_feasibility/feasibility.jsonl"
        if icl_replay is None and icl_posthoc_rows.exists():
            normalized = [row for row in read_jsonl(icl_posthoc_rows) if row["method"] == "in_context"]
            icl_replay = next((row for row in normalized if row["record_id"] == case_id
                               and (row.get("solver") or {}).get("witness_found")), None)
            if icl_replay is None:
                icl_replay = next((row for row in normalized if (row.get("solver") or {}).get("witness_found")), None)
            icl_replay_label = "ICL, po zdjęciu ramki Markdown (analiza post-hoc)"
        if icl_replay is not None:
            show_search_sequence(icl_replay, icl_replay_label)
        elif icl_examples:
            print("W zapisanych wynikach ICL nie ma ciągu znalezionego przez BFS.")
        '''),
        markdown("""
        ## ICL: przykłady w kontekście zamiast zmiany wag

        Wariant ICL używa bazowego Qwena i stałych przykładów wybranych
        wyłącznie z treningu. Wagi nie są aktualizowane. Plansza ocenianego
        przypadku jest podawana bez swojej przyszłej obserwacji i ruchów;
        pełne prompty poszczególnych przypadków są zapisywane oddzielnie.
        Jeżeli plik wyników nie powstał, poniższa komórka wyświetla brak danych
        zamiast przypisywać metodzie wynik zerowy.
        """),
        code('''
        icl_dir = RUN / "icl_followup"
        if icl is None:
            print("Wyniki oceny ICL nie są jeszcze zapisane w", icl_path)
        else:
            print("Wyniki ICL znajdują się w tabelach i na wykresach porównawczych powyżej.")
            print("Pola historycznego modelu bazowego, których nie dało się niezależnie potwierdzić:")
            print(json.dumps(icl.get("comparison_metadata_missing"), ensure_ascii=False, indent=2))
        support_path = icl_dir / "support.json"
        if support_path.exists():
            print("Zamrożony wybór przykładów z treningu:")
            display(Code(support_path.read_text(encoding="utf-8"), language="json"))
        prompt_files = sorted(icl_dir.rglob("*.prompts.jsonl")) if icl_dir.exists() else []
        if prompt_files:
            default_prompt = icl_dir / "predictions.prompts.jsonl"
            PROMPT_FILE_INDEX = prompt_files.index(default_prompt) if default_prompt in prompt_files else 0
            PROMPT_INDEX = 0  # dowolny indeks albo None, aby pokazać wszystkie z pliku
            for i, path in enumerate(prompt_files):
                print(f"{i}: {path.relative_to(RUN)}")
            prompt_path = prompt_files[PROMPT_FILE_INDEX]
            prompts = read_jsonl(prompt_path)
            print(f"Liczba zapisanych pełnych promptów w pliku: {len(prompts)}.")
            selected_prompts = prompts if PROMPT_INDEX is None else prompts[PROMPT_INDEX:PROMPT_INDEX + 1]
            for prompt in selected_prompts:
                print("Kompletny, zapisany prompt ICL:", prompt.get("record_id"))
                rendered = next((prompt[k] for k in ("rendered_prompt", "prompt_text", "rendered", "prompt")
                                 if isinstance(prompt.get(k), str)), None)
                if rendered:
                    display(Code(rendered, language="text"))
                else:
                    display(Code(json.dumps(prompt, ensure_ascii=False, indent=2), language="json"))
            print("Wszystkie pełne prompty:", prompt_path)
        else:
            print("Pełne prompty ICL nie zostały jeszcze zapisane.")
        '''),
        markdown("""
        ## Wnioski i ograniczenia

        LoRA poprawiła zgodność z wymaganym formatem odpowiedzi w tym teście;
        po pomocniczym usunięciu ramki Markdown obie wersje miały podobną
        liczbę poprawnych nietrywialnych intencji. BFS częściej znajdowało ciąg
        dla propozycji LoRA, lecz model bazowy często żądał ułożenia wszystkich
        skrzyń na celach, a LoRA przede wszystkim proponowała pozycję skrzyni.
        Liczby dla ICL oraz informacja o ewentualnym powtarzaniu całego zadania
        są wyliczane z zapisanych wyników w tabeli powyżej. Trudność propozycji
        nie jest wyrównana. Nie oceniano kontrolera warunkowanego intencją ani
        transferu; wyniki BFS nie są porównaniem skuteczności rozwiązywania
        pełnych plansz przez metody generujące różne intencje.

        Dane pochodzą z wcześniej wyselekcjonowanej syntetycznej kohorty, a
        walidacja obejmowała tylko trzy niezależne plansze. Wskaźniki testowe
        wykorzystują 20 plansz; pokazany bootstrap losuje plansze parami.
        Jeden kontekst nie jest niezależną planszą. Zarejestrowane braki,
        odcięcia i błędy są częścią wyników, nie wyzerowanymi sukcesami.
        """),
        markdown("""
        **Propozycja krótkiego opisu do pracy (British English):**

        > In an exploratory Sokoban study, a small pretrained model was adapted
        > to propose structured intentions from the current board. Under strict
        > JSON parsing on held-out contexts, valid non-trivial intentions
        > occurred in 0/100 base proposals and 66/100 adapted proposals. After
        > removing one outer Markdown fence in a post-hoc analysis, both
        > methods produced 66/100 valid proposals; bounded search found paths
        > for 7/100 and 44/100, respectively. The proposals were not matched
        > for difficulty, and these findings do not establish controller
        > reachability or transfer.
        """),
        markdown("""
        ## Odtwarzanie

        Notebook nie trenuje modelu i można go wykonać ponownie lokalnie.
        Uruchom w katalogu CaRL:

        ```bash
        uv run --no-sync python scripts/build_intent_generator_notebook.py \\
          --run-dir analysis/intent_generator_20261114T032232Z --execute
        ```

        Surowe dane i wszystkie metryki są w katalogu przebiegu, obok tego
        notebooka. Poniżej wyświetlamy pełne zapisane komendy etapów zarówno
        pierwotnego przebiegu, jak i uzupełnienia ICL. To zapis audytowy:
        `--stop-monotonic` to granica konkretnej sesji i uruchomienia systemu,
        a nie przenośny czas do wklejenia po restarcie. Ponowne uruchomienie
        kosztownych etapów wymaga nowego, nadzorowanego przebiegu i osobnego
        katalogu wyjściowego; wykonanie tej komórki niczego nie uruchamia.

        Dalej znajduje się pełny archiwalny opis pierwotnego przebiegu LoRA,
        dołączony do notebooka dla audytu.
        """),
        code('''
        import shlex

        for label, path in (("Pierwotny przebieg", RUN / "run_manifest.json"),
                            ("Uzupełnienie ICL", RUN / "icl_followup/run_manifest.json")):
            print("\\n", label, "—", path)
            if not path.exists():
                print("Brak manifestu.")
                continue
            for stage in read_json(path).get("stages", []):
                command = stage.get("command")
                if command:
                    print(f"\\n{stage.get('name')}: {stage.get('status')}, kod {stage.get('exit_code')}")
                    print(shlex.join(str(part) for part in command))
        '''),
        markdown("# Archiwum pierwotnego raportu LoRA\n\n" + original_report),
    ]
    return nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true", help="Execute all local reporting cells")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    run = args.run_dir.resolve()
    output = args.output or run / "intent_generator_report.ipynb"
    report = (run / "REPORT.md").read_text(encoding="utf-8")
    book = notebook(os.path.relpath(run, repo), report)
    if args.execute:
        NotebookClient(book, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(repo)}}).execute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.",
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            nbformat.write(book, stream)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(output)


if __name__ == "__main__":
    main()
