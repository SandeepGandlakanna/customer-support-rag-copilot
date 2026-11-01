
import json
from pathlib import Path

from app.rag import vector_store

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "evaluation_dataset.json"

K = 2
DISTANCE_THRESHOLD = 0.70

total_precision = 0
total_hits = 0
total_retrieved = 0

with open(DATASET_PATH, "r", encoding="utf-8") as file:
    test_cases = json.load(file)

for test_case in test_cases:
    question = test_case["question"]
    expected_document = test_case["expected_document"]

    # Retrieve documents with their distance scores
    results = vector_store.similarity_search_with_score(
        question,
        k=K,
    )

    # Keep documents within the distance threshold
    filtered_results = [
        (document, score)
        for document, score in results
        if score <= DISTANCE_THRESHOLD
    ]

    relevant_count = sum(
        1
        for document, score in filtered_results
        if expected_document in document.page_content
    )

    retrieved_count = len(filtered_results)

    # Precision among documents that survived filtering
    precision = (
        relevant_count / retrieved_count
        if retrieved_count > 0 else 0
    )

    # Did we retrieve the expected document?
    hit = 1 if relevant_count > 0 else 0

    total_precision += precision
    total_hits += hit
    total_retrieved += retrieved_count

    print("\nQuestion:", question)
    print("Expected:", expected_document)

    print("\nFiltered documents:")
    for document, score in filtered_results:
        print(f"Distance: {score:.4f}")
        print(document.page_content.splitlines()[0])

    print("Documents retained:", retrieved_count)
    print("Documents rejected:", len(results) - retrieved_count)
    print(f"Filtered precision: {precision:.2%}")
    print(f"Hit@{K}: {hit:.2%}")

total = len(test_cases)

print("\n========== THRESHOLD EVALUATION ==========")
print(f"Threshold: {DISTANCE_THRESHOLD}")
print(f"Average filtered precision: {total_precision / total:.2%}")
print(f"Average Hit@{K}: {total_hits / total:.2%}")
print(f"Total documents retained: {total_retrieved}")
