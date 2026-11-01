from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
import time

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")

llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0,
)

print(f"Testing Groq model: {GROQ_MODEL}")

start = time.perf_counter()

try:
    response = llm.invoke(
        "Explain RAG in simple English in 3 sentences."
    )

    print("\nResponse:")
    print(response.content)

    print(
        f"\nTime taken: "
        f"{time.perf_counter() - start:.2f} seconds"
    )

except Exception as error:
    print("Error:", type(error).__name__)
    print("Details:", error)