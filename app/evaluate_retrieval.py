
import json
from pathlib import Path

from app.rag import vector_store


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_FILE = BASE_DIR / "data" / "evaluation_dataset.json"

K = 2
DISTANCE_THRESHOLD = 0.55


with open(DATASET_FILE, "r", encoding="utf-8") as file:
    dataset = json.load(file)


total_precision = 0
total_hits = 0
total_in_scope = 0

total_oos = 0
correct_rejections = 0

print("\n========== RETRIEVAL EVALUATION ==========")

for item in dataset:
    question = item["question"]
    expected_document = item["expected_document"]

    results = vector_store.similarity_search_with_score(
        question,
        k=K,
    )

    # Keep only documents within the distance threshold
    accepted_results = [
        (doc, score)
        for doc, score in results
        if score <= DISTANCE_THRESHOLD
    ]

    retrieved_documents = [
        doc.page_content for doc, score in accepted_results
    ]

    print(f"\nQuestion: {question}")

    if expected_document is not None:
        total_in_scope += 1

        relevant_count = sum(
            1
            for content in retrieved_documents
            if expected_document in content
        )

        # Precision among accepted documents
        precision = (
            relevant_count / len(retrieved_documents)
            if retrieved_documents
            else 0
        )

        hit = 1 if relevant_count > 0 else 0

        total_precision += precision
        total_hits += hit

        print(f"Expected: {expected_document}")
        print(f"Accepted documents: {len(accepted_results)}")
        print(f"Precision: {precision:.2f}")
        print(f"Hit: {hit}")

    else:
        total_oos += 1

        rejected = len(accepted_results) == 0

        if rejected:
            correct_rejections += 1

        print("Expected: OUT OF SCOPE")
        print(f"Accepted documents: {len(accepted_results)}")
        print(f"Correctly rejected: {rejected}")


precision_at_k = (
    total_precision / total_in_scope
    if total_in_scope else 0
)

hit_at_k = (
    total_hits / total_in_scope
    if total_in_scope else 0
)

rejection_accuracy = (
    correct_rejections / total_oos
    if total_oos else 0
)


print("\n========== FINAL METRICS ==========")

print(f"Precision@{K}: {precision_at_k:.2%}")
print(f"Hit@{K}: {hit_at_k:.2%}")
print(f"Rejection Accuracy: {rejection_accuracy:.2%}")

print(f"\nIn-scope questions: {total_in_scope}")
print(f"Out-of-scope questions: {total_oos}")
