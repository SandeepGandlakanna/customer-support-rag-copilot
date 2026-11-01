
from app.rag import vector_store

question = "How can I request a refund?"

# 1. Similarity Search
print("\n========== SIMILARITY SEARCH ==========")

similarity_results = vector_store.similarity_search_with_score(
    question,
    k=2,
)

for index, (document, score) in enumerate(similarity_results, start=1):
    print(f"\nDocument {index}")
    print("Distance:", round(score, 4))
    print("Content:", document.page_content)


# 2. MMR Search
print("\n========== MMR SEARCH ==========")

mmr_results = vector_store.max_marginal_relevance_search(
    question,
    k=2,
    fetch_k=3,
    lambda_mult=0.5,
)

for index, document in enumerate(mmr_results, start=1):
    print(f"\nDocument {index}")
    print("Content:", document.page_content)
