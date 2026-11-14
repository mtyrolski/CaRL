#!/usr/bin/env python3
"""Create an offline notebook describing the prepared Sokoban ICL evaluation."""

import argparse
import os
import tempfile
from pathlib import Path
from textwrap import dedent

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "analysis/icl_followup_20261114T074555Z"


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbformat.v4.new_code_cell(dedent(source).strip())


def make_notebook(run: Path):
    relative_run = str(run.relative_to(ROOT))
    cells = [
        markdown("""
        # Sokoban: przygotowanie eksperymentu ICL

        Ten notebook odczytuje wyłącznie zapisane lokalnie artefakty. Nie używa
        GPU, nie wywołuje API i nie trenuje kontrolera. Pokazuje istniejący
        wynik ICL oraz przygotowany drugi zestaw; brak wyniku nie oznacza zera.
        """),
        code(f'''
        from pathlib import Path
        import hashlib
        import json
        import pandas as pd
        from IPython.display import Code, Image, Markdown, display

        here = Path.cwd().resolve()
        ROOT = next((p for p in (here, *here.parents)
                     if (p / "pyproject.toml").is_file() and (p / "carl").is_dir()), None)
        if ROOT is None:
            raise RuntimeError("Uruchom notebook z checkoutu CaRL")
        RUN = ROOT / {relative_run!r}
        ORIGINAL = ROOT / "analysis/intent_generator_20261114T032232Z"
        MATRIX = ROOT / "analysis/lora_matrix_20261114T065804Z"
        def read_json(path, default=None):
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
        def read_jsonl(path):
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()] if path.exists() else []
        def sha(path):
            return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        print("Katalog:", RUN)
        print("Status nadzorcy:", (read_json(RUN / "run_manifest.json", {{}}) or {{}}).get("status", "brak"))
        '''),
        markdown("""
        ## Dostępne wyniki

        Wcześniejszy ICL używał dwóch stałych, sprawdzonych przykładów z plansz
        treningowych. W pierwotnej ocenie było 100 kontekstów z 20 plansz ×10;
        konteksty tej samej planszy nie są niezależne. Wyniki r2, r4 i r8
        pochodzą z tego samego zestawu, ale to analiza eksploracyjna, bo wyniki
        części metod były znane przed wyborem kolejnych wariantów.

        Kolumna `strict_valid` wymaga ścisłego JSON; `fence_valid` to osobna
        analiza po usunięciu pojedynczej otoczki Markdown. Powtórzenie całego
        celu zadania może przejść sprawdzenie intencji, ale nie musi stanowić
        użytecznej krótkiej intencji.
        """),
        code('''
        summary = pd.read_csv(MATRIX / "assessment_main/summary.csv")
        columns = ["variant", "outputs", "boards_observed", "strict_json", "strict_valid",
                   "fence_json", "fence_valid", "fence_whole_goal_only", "bfs_found", "bfs_attempted"]
        display(summary.loc[(summary.family == "all") & summary.variant.isin(
            ["base", "icl", "r2_100", "r4_100", "r8_100"]), columns])
        for file in (ORIGINAL / "icl_followup/predictions.summary.json",
                     ORIGINAL / "icl_followup/support.json"):
            print("\\n", file, "SHA-256:", sha(file))
            display(Code(json.dumps(read_json(file), ensure_ascii=False, indent=2), language="json"))
        plot = MATRIX / "assessment_main/validity_overall.png"
        if plot.exists():
            display(Image(filename=str(plot)))
        '''),
        markdown("""
        ## Przykłady w prompcie ICL

        Przykłady są ustalone deterministycznie na podstawie zbioru treningowego,
        a nie wybierane po wyniku testowym. Pokazujemy ich pełne etykiety i
        bieżące plansze. Drugi przykład zawiera żądanie co najmniej dziesięciu
        skrzyń na celach, czyli cały cel dla planszy ×10. Może to zachęcać do
        kopiowania celu; sam wynik nie dowodzi, że właśnie to jest przyczyną.
        Zmiana przykładów po obejrzeniu wyników wymagałaby osobnego, jawnie
        nazwanego eksperymentu i niezależnego testu.
        """),
        code('''
        support = read_json(ORIGINAL / "icl_followup/support.json", {})
        training_data = ORIGINAL / "dataset.jsonl"
        rows = {r["record_id"]: r for r in read_jsonl(training_data)}
        for identifier in support.get("record_ids", []):
            row = rows[identifier]
            print("\\nPrzykład", identifier, "split", row["split"], "typ", row.get("effect_type"))
            print("\\n".join(row["start_observation"]))
            print("Cel:", row["task_goal"])
            print("Intencja:", json.dumps(row["intent"], ensure_ascii=False))
        '''),
        markdown("""
        ## Przygotowany zbiór ×4 i ×10

        Test zawiera po 20 kontekstów z różnych plansz w każdej rodzinie. Plansze
        ×4 pochodzą z innego źródła niż ×10, a część ×10 występuje również w
        poprzednim teście. Nie wolno sumować tych wyników jako niezależnych ani
        przypisywać różnicy wyłącznie liczbie skrzyń.

        Wcześniejsza próba ICL zatrzymała się przed wygenerowaniem odpowiedzi:
        plik ID nie zawierał hasha zbioru. Przygotowany plik `icl_test_ids.json`
        zawiera te same ID w tej samej kolejności i wiąże je z hashem danych;
        oryginalny plik pozostał zachowany. Żaden wynik nie jest dopisywany.
        """),
        code('''
        data = MATRIX / "eval_transfer/dataset.jsonl"
        old_ids = MATRIX / "eval_transfer/test_ids.json"
        ids_file = MATRIX / "eval_transfer/icl_test_ids.json"
        old = read_json(old_ids, {})
        frozen = read_json(ids_file, {})
        items = read_jsonl(data)
        support_layouts = set(support.get("layout_ids", []))
        test_layouts = {r["layout_id"] for r in items if r.get("record_id") in frozen.get("record_ids", [])}
        print("Hash danych:", sha(data))
        print("Hash w manifeście:", frozen.get("dataset_sha256"))
        print("ID identyczne i w tej samej kolejności:", old.get("record_ids") == frozen.get("record_ids"))
        print("Przecięcie plansz przykładów i testu:", len(support_layouts & test_layouts))
        frame = pd.DataFrame(items)
        family = "evaluation_family" if "evaluation_family" in frame else "family"
        if not frame.empty and family in frame:
            display(frame.groupby(family).agg(konteksty=("record_id", "size"), plansze=("layout_id", "nunique")))
        manifest = MATRIX / "eval_transfer/icl_test_ids.manifest.json"
        display(Code(json.dumps(read_json(manifest), ensure_ascii=False, indent=2), language="json"))
        print("Przygotowane odpowiedzi transferowe:", len(read_jsonl(RUN / "transfer.jsonl")))
        '''),
        markdown("""
        ## Pełne prompty

        Poniżej są pełne wiadomości i wyrenderowany prompt każdego dostępnego
        kontekstu z drugiego zestawu. Pokazujemy liczbę znalezionych promptów,
        żeby brakującego preflightu nie pomylić z wykonaną oceną. Żaden prompt
        zapytania nie powinien zawierać późniejszej planszy ani demonstracyjnych
        ruchów. Dwa przykłady treningowe wraz z ich etykietami są dozwolonym
        wejściem ICL.
        """),
        code('''
        prompts_path = RUN / "transfer.prompts.jsonl"
        prompts = read_jsonl(prompts_path)
        print("Plik:", prompts_path, "SHA-256:", sha(prompts_path), "liczba:", len(prompts))
        for p in prompts:
            print("\\nKontekst:", p["record_id"], "tokeny:", p.get("prompt_tokens"))
            print(json.dumps(p["messages"], ensure_ascii=False, indent=2))
            print("Wyrenderowany prompt:\\n", p.get("rendered_prompt"))
        '''),
        markdown("""
        ## Kontroler niskiego poziomu

        Proponowana osobna ocena może podawać każdą intencję temu samemu,
        zamrożonemu kontrolerowi wraz z bieżącą planszą i historią jego ruchów.
        Nie wolno podawać przyszłej planszy demonstracji. Wagi, prompt, sposób
        dekodowania oraz budżet akcji trzeba ustalić przed porównaniem metod.
        Ruchy zablokowane również liczą się do budżetu. Generowanie intencji,
        pomiar BFS i wykonanie przez kontroler są różnymi pomiarami.

        Istniejący `carl.icl.luna_intent_rollout.rollout` ma taki kontrakt dla
        pojedynczych ruchów. Pilot lokalnej polityki Qwen2.5-0.5B jest testem
        integracji: wykorzystuje zamrożone wagi, niewielki budżet i intencję
        z demonstracji. Jeden przypadek nie służy do oceny uogólnienia ani
        porównania generatorów. Wagi polityki nie były trenowane.
        """),
        code('''
        for name in ("icl_transfer_plan_verified.json", "policy_plan.json", "policy_manifest.json", "run_manifest.json"):
            path = RUN / name
            if path.exists():
                print("\\nZapisany plik:", path)
                display(Code(json.dumps(read_json(path), ensure_ascii=False, indent=2), language="json"))
        policy = RUN / "qwen_policy_pilot"
        policy_manifest = read_json(policy / "manifest.json")
        if policy_manifest:
            print("\\nPełna konfiguracja, prompt i komenda pilota polityki:")
            display(Code(json.dumps(policy_manifest, ensure_ascii=False, indent=2), language="json"))
        else:
            print("Brak zapisanego manifestu pilota polityki.")
        policy_rows = read_jsonl(policy / "results.jsonl")
        print("Zapisane przypadki pilota:", len(policy_rows))
        for item in policy_rows:
            outcome = item.get("result") or {}
            status = outcome.get("status")
            if status == "api_error":
                status = "lokalna awaria inferencji"
            print("\\nPrzypadek:", item.get("record_id"), "status:", status,
                  "osiągnięto intencję:", outcome.get("achieved"),
                  "wywołania modelu:", outcome.get("model_calls"))
            for step in outcome.get("steps", []):
                print("\\nKrok", step.get("index"), "pełny prompt:")
                print(step.get("prompt"))
                print("Odpowiedź:", step.get("raw_response"), "akcja:", step.get("action"),
                      "bez zmiany:", step.get("no_op"),
                      "intencja spełniona:", step.get("intent_holds"),
                      "zachowanie spełnione:", step.get("preservation_holds"))
                print("Plansza po ruchu:\\n" + "\\n".join(step.get("after") or []))
        '''),
        markdown("""
        ## Odtworzenie i ograniczenia

        Polecenia zapisane w manifeście nadzorcy dotyczą oryginalnego przebiegu;
        jego termin nie został przedłużony. Przyszłą ocenę należy uruchomić jako
        jawnie oddzielony eksperyment z własnym nadzorem, zapisem statusu oraz
        limitami. Nie używaj wyników testowych do wybierania przykładów ICL.

        Przykład komendy oceny na wcześniej zamrożonych 40 kontekstach, po
        ustaleniu nowego nadzoru i katalogu wyników:

        ```bash
        uv run --offline --no-sync --with peft==0.21.3 python -m carl.icl.intent_in_context evaluate \\
          --data analysis/lora_matrix_20261114T065804Z/eval_transfer/dataset.jsonl \\
          --test-ids analysis/lora_matrix_20261114T065804Z/eval_transfer/icl_test_ids.json \\
          --support analysis/intent_generator_20261114T032232Z/icl_followup/support.json \\
          --support-data analysis/intent_generator_20261114T032232Z/dataset.jsonl \\
          --output analysis/icl_future_run/transfer.jsonl \\
          --model analysis/intent_generator_20261114T032232Z/models/Qwen3-0.6B \\
          --revision c1899de289a04d12100db370d81485cdf75e47ca \\
          --max-length 768 --max-new-tokens 128
        ```

        Notebook można ponownie wykonać bez GPU i API z katalogu CaRL:

        ```bash
        uv run --offline --no-sync python scripts/build_icl_followup_notebook.py
        ```

        Pełne wcześniej wykonane porównanie i wykresy są w notebooku
        `analysis/lora_matrix_20261114T065804Z/lora_matrix_report.ipynb`.
        """),
    ]
    document = nbformat.v4.new_notebook(cells=cells)
    document.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    document.metadata["language_info"] = {"name": "python", "version": "3.11"}
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run.resolve()
    document = make_notebook(run)
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
            cell.outputs.extend(nbformat.v4.new_output("display_data", data=o.data, metadata=o.metadata)
                                for o in captured.outputs)
    finally:
        os.chdir(previous_directory)
    run.mkdir(parents=True, exist_ok=True)
    destination = run / "icl_followup_report.ipynb"
    fd, temporary = tempfile.mkstemp(prefix=".icl-notebook-", suffix=".ipynb", dir=run)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            nbformat.write(document, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(destination)


if __name__ == "__main__":
    main()
