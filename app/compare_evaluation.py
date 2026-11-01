
import json
from pathlib import Path

from app.rag import vector_store

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "evaluation_dataset.json"

K = 2
FETCH_K = 3
LAMBDA_MULT = 0.5


def evaluate_retrieval(documents, expected_document):
    relevant_count = sum(
        1
        for document in documents
        if expected_document in document.page_content
    )

    precision_at_k = relevant_count / K
    hit_at_k = 1.0 if relevant_count > 0 else 0.0

    return precision_at_k, hit_at_k


with open(DATASET_PATH, "r", encoding="utf-8") as file:
    test_cases = json.load(file)

similarity_precision = []
similarity_hits = []

mmr_precision = []
mmr_hits = []

for test_case in test_cases:
    question = test_case["question"]
    expected_document = test_case["expected_document"]

    # Similarity Search
    similarity_results = vector_store.similarity_search(
        question,
        k=K,
    )

    # MMR Search
    mmr_results = vector_store.max_marginal_relevance_search(
        question,
        k=K,
        fetch_k=FETCH_K,
        lambda_mult=LAMBDA_MULT,
    )

    similarity_p, similarity_h = evaluate_retrieval(
        similarity_results, expected_document
    )

    mmr_p, mmr_h = evaluate_retrieval(
        mmr_results, expected_document
    )

    similarity_precision.append(similarity_p)
    similarity_hits.append(similarity_h)

    mmr_precision.append(mmr_p)
    mmr_hits.append(mmr_h)

    print("\nQuestion:", question)
    print("Expected:", expected_document)

    print("\nSimilarity Search:")
    for document in similarity_results:
        print("-", document.page_content.splitlines()[0])

    print(f"Precision@{K}: {similarity_p:.2%}")
    print(f"Hit@{K}: {similarity_h:.2%}")

    print("\nMMR Search:")
    for document in mmr_results:
        print("-", document.page_content.splitlines()[0])

    print(f"Precision@{K}: {mmr_p:.2%}")
    print(f"Hit@{K}: {mmr_h:.2%}")


total = len(test_cases)

print("\n========== FINAL EVALUATION ==========")

print("\nSimilarity Search")
print(
    f"Average Precision@{K}: "
    f"{sum(similarity_precision) / total:.2%}"
)
print(
    f"Average Hit@{K}: "
    f"{sum(similarity_hits) / total:.2%}"
)

print("\nMMR Search")
print(
    f"Average Precision@{K}: "
    f"{sum(mmr_precision) / total:.2%}"
)
print(
    f"Average Hit@{K}: "
    f"{sum(mmr_hits) / total:.2%}"
)
