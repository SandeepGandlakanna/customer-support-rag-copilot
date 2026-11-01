
import logging
import secrets
import time

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field, field_validator

from app.config import APP_API_KEY
from app.rag import ask_question


# ==========================================
# 1. LOGGING CONFIGURATION
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ==========================================
# 2. FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Customer Support RAG Copilot",
    description="AI-powered customer support question-answering API",
    version="1.0.0",
)


# ==========================================
# 3. REQUEST MODEL
# ==========================================

class QuestionRequest(BaseModel):

    question: str = Field(
        min_length=1,
        max_length=2000,
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value):

        if not value.strip():
            raise ValueError(
                "Question cannot contain only whitespace."
            )

        return value.strip()


# ==========================================
# 4. API KEY AUTHENTICATION
# ==========================================

def verify_api_key(
    x_api_key: str = Header(...),
):

    if not APP_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="API authentication is not configured.",
        )

    if not secrets.compare_digest(
        x_api_key,
        APP_API_KEY,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )

    return True


# ==========================================
# 5. HEALTH CHECK ENDPOINT
# ==========================================

@app.get("/health")
def health_check():

    logger.info("Health check requested.")

    return {
        "status": "healthy"
    }


# ==========================================
# 6. ASK QUESTION ENDPOINT
# ==========================================

@app.post("/ask")
def ask(
    request: QuestionRequest,
    authenticated: bool = Depends(verify_api_key),
):

    request_start = time.perf_counter()

    logger.info("Received /ask request.")

    try:

        result = ask_question(request.question)

        request_time = time.perf_counter() - request_start

        logger.info(
            "Request outcome: %s | Duration: %.2f seconds",
            result.get("status", "unknown"),
            request_time,
        )

        return result

    except Exception:

        logger.exception("API request failed.")

        raise HTTPException(
            status_code=500,
            detail="Failed to process the question. Check server logs.",
        )
