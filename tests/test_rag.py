
from unittest.mock import MagicMock, patch

from app.rag import ask_question, FALLBACK_MESSAGE


def test_relevant_question_returns_answer_and_sources():
    fake_document = MagicMock()
    fake_document.page_content = (
        "1. What is the refund policy?\n"
        "Customers can request a refund within 30 days."
    )
    fake_document.metadata = {"source": "support_faq.txt"}

    fake_response = MagicMock()
    fake_response.content = (
        "Customers can request a refund within 30 days."
    )

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = fake_response

    with patch(
        "app.rag.vector_store.similarity_search_with_score",
        return_value=[(fake_document, 0.30)],
    ), patch(
        "app.rag.llm",
        fake_llm,
    ):
        result = ask_question("What is the refund policy?")

    assert "30 days" in result["answer"]
    assert result["sources"][0]["source"] == "support_faq.txt"
    assert result["sources"][0]["section"] == (
        "1. What is the refund policy?"
    )
    fake_llm.invoke.assert_called_once()


def test_irrelevant_question_returns_fallback():
    fake_document = MagicMock()
    fake_document.page_content = "Unrelated document content."
    fake_document.metadata = {"source": "other.txt"}

    fake_llm = MagicMock()

    with patch(
        "app.rag.vector_store.similarity_search_with_score",
        return_value=[(fake_document, 0.90)],
    ), patch(
        "app.rag.llm",
        fake_llm,
    ):
        result = ask_question("What is the capital of France?")

    assert result["answer"] == FALLBACK_MESSAGE
    assert result["sources"] == []
    fake_llm.invoke.assert_not_called()

from unittest.mock import patch

from app.rag import ask_question, ERROR_MESSAGE


def test_retrieval_failure_returns_error():

    with patch(
        "app.rag.vector_store.similarity_search_with_score",
        side_effect=Exception("Gemini embedding service unavailable"),
    ), patch("app.rag.llm") as fake_llm:

        result = ask_question("What is the refund policy?")

    assert result["answer"] == ERROR_MESSAGE
    assert result["sources"] == []
    assert result["status"] == "error"

    fake_llm.invoke.assert_not_called()

def test_groq_failure_returns_error():

    fake_document = MagicMock()

    fake_document.page_content = (
        "1. What is the refund policy?\n"
        "Customers can request a refund within 30 days."
    )

    fake_document.metadata = {
        "source": "support_faq.txt"
    }

    fake_llm = MagicMock()

    fake_llm.invoke.side_effect = Exception(
        "Groq service unavailable"
    )

    with patch(
        "app.rag.vector_store.similarity_search_with_score",
        return_value=[(fake_document, 0.30)],
    ), patch("app.rag.llm", fake_llm):

        result = ask_question("What is the refund policy?")

    assert result["answer"] == ERROR_MESSAGE
    assert result["sources"] == []
    assert result["status"] == "error"

    fake_llm.invoke.assert_called_once()
