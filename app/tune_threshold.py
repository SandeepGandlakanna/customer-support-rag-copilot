
import json
from pathlib import Path

from app.rag import vector_store


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_FILE = BASE_DIR / "data" / "evaluation_dataset.json"

K = 2
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]

with open(DATASET_FILE, "r", encoding="utf-8") as file:
    dataset = json.load(file)


# Retrieve candidates once per question
all_results = []

for item in dataset:
    results = vector_store.similarity_search_with_score(
        item["question"],
        k=K,
    )

    all_results.append({
        "question": item["question"],
        "expected_document": item["expected_document"],
        "results": results,
    })


print("\n========== THRESHOLD COMPARISON ==========")

print(
    f"{'Threshold':<12}"
    f"{'Precision':<14}"
    f"{'Hit@2':<12}"
    f"{'Rejection':<14}"
)

for threshold in THRESHOLDS:
    precision_sum = 0
    hits = 0
    correct_rejections = 0

    in_scope_count = 0
    out_scope_count = 0

    for item in all_results:
        expected = item["expected_document"]

        accepted = [
            (doc, score)
            for doc, score in item["results"]
            if score <= threshold
        ]

        if expected is not None:
            in_scope_count += 1

            relevant = sum(
                1 for doc, score in accepted
                if expected in doc.page_content
            )

            precision = (
                relevant / len(accepted)
                if accepted else 0
            )

            precision_sum += precision

            if relevant > 0:
                hits += 1

        else:
            out_scope_count += 1

            if not accepted:
                correct_rejections += 1

    precision_at_k = precision_sum / in_scope_count
    hit_at_k = hits / in_scope_count
    rejection_accuracy = correct_rejections / out_scope_count

    print(
        f"{threshold:<12.2f}"
        f"{precision_at_k:<14.2%}"
        f"{hit_at_k:<12.2%}"
        f"{rejection_accuracy:<14.2%}"
    )
