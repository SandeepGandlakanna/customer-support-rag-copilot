
import json
import logging
from pathlib import Path

from app.rag import ask_question

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "data" / "answer_evaluation_dataset.json"
REPORT_PATH = BASE_DIR / "data" / "answer_evaluation_report.json"


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_answer(expected_answer, generated_answer):
    expected_words = set(expected_answer.lower().split())
    generated_words = set(generated_answer.lower().split())

    if not expected_words:
        return 0.0

    matched_words = expected_words.intersection(generated_words)

    return round(
        len(matched_words) / len(expected_words),
        2,
    )


def main():
    dataset = load_dataset()

    results = []

    for item in dataset:
        question = item["question"]
        expected_answer = item["expected_answer"]

        response = ask_question(question)

        generated_answer = response["answer"]

        score = evaluate_answer(
            expected_answer,
            generated_answer,
        )

        result = {
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "status": response["status"],
            "answer_overlap_score": score,
        }

        results.append(result)

        print("\nQuestion:", question)
        print("Expected:", expected_answer)
        print("Generated:", generated_answer)
        print("Score:", score)
        print("Status:", response["status"])

    average_score = sum(
        item["answer_overlap_score"] for item in results
    ) / len(results)

    report = {
        "total_questions": len(results),
        "average_answer_overlap_score": round(average_score, 2),
        "results": results,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\n========== EVALUATION SUMMARY ==========")
    print("Total questions:", len(results))
    print("Average answer overlap:", round(average_score, 2))
    print("Report saved to:", REPORT_PATH)


if __name__ == "__main__":
    main()
