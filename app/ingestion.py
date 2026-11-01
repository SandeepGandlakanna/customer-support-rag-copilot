
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import GEMINI_API_KEY

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "support_faq.txt"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Load FAQ document
loader = TextLoader(
    str(DATA_FILE),
    encoding="utf-8"
)

documents = loader.load()

# Extract FAQ text
faq_text = documents[0].page_content

# Split using FAQ question numbers
faq_sections = faq_text.split("\n\n")

faq_chunks = []
current_faq = []

for section in faq_sections:

    section = section.strip()

    if not section:
        continue

    # Detect a new FAQ question
    if section[0].isdigit() and ". " in section:

        if current_faq:
            faq_chunks.append("\n\n".join(current_faq))

        current_faq = [section]

    elif current_faq:
        current_faq.append(section)

# Save final FAQ
if current_faq:
    faq_chunks.append("\n\n".join(current_faq))

# Convert text chunks into LangChain Documents
chunk_documents = [
    Document(
        page_content=chunk,
        metadata={
            "source": DATA_FILE.name,
            "chunk_id": f"faq_{index:03d}"
        }
    )
    for index, chunk in enumerate(faq_chunks, start=1)
]

print(f"Total FAQ chunks created: {len(chunk_documents)}")

for index, chunk in enumerate(chunk_documents, start=1):
    print(f"\nChunk {index}:")
    print(chunk.page_content)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY,
)

# Initialize ChromaDB
vector_store = Chroma(
    collection_name="customer_support",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR),
)

# Replace old indexed records to avoid duplicates
existing = vector_store.get()

if existing["ids"]:
    vector_store.delete(ids=existing["ids"])

# Add new FAQ chunks
vector_store.add_documents(chunk_documents)

print(
    f"\nSuccessfully indexed "
    f"{len(chunk_documents)} FAQ chunks."
)
