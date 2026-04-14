import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASELINE_PATH = ROOT / "baseline_CoTs_options_ollama.jsonl"
CORRECTIONS_PATH = ROOT / "baseline_CoTs_options_ollama_corrections.json"


def main():
    corrections = json.loads(CORRECTIONS_PATH.read_text(encoding="utf-8"))
    rows = []

    with BASELINE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            correction = corrections.get(str(obj.get("question_id")))
            if correction:
                obj["parsed_answer"] = correction["parsed_answer"]
                obj["correct"] = correction["correct"]
            rows.append(obj)

    with BASELINE_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


if __name__ == "__main__":
    main()
