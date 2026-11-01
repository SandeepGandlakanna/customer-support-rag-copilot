
import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv


# ==========================================
# 1. ENVIRONMENT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

APP_API_KEY = os.getenv("APP_API_KEY")

API_URL = "http://127.0.0.1:8000/ask"


# ==========================================
# 2. STREAMLIT PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Customer Support RAG Copilot",
    page_icon="🤖",
    layout="centered",
)

st.title("Customer Support RAG Copilot")

st.write(
    "Ask questions about your customer support documents."
)


# ==========================================
# 3. API KEY VALIDATION
# ==========================================

if not APP_API_KEY:

    st.error(
        "APP_API_KEY is missing. "
        "Please configure it in your .env file."
    )

    st.stop()


# ==========================================
# 4. CHAT HISTORY
# ==========================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("Sources"):

                for source in message["sources"]:
                    st.write(source)


# ==========================================
# 5. USER INPUT
# ==========================================

user_question = st.chat_input(
    "Ask a support question..."
)


# ==========================================
# 6. SEND REQUEST TO FASTAPI
# ==========================================

if user_question:

    # Display and save user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    with st.chat_message("user"):

        st.markdown(user_question)

    # Call backend

    with st.chat_message("assistant"):

        with st.spinner("Searching support documents..."):

            try:

                response = requests.post(
                    API_URL,
                    json={
                        "question": user_question
                    },
                    headers={
                        "X-API-Key": APP_API_KEY
                    },
                    timeout=120,
                )

                response.raise_for_status()

                result = response.json()

                answer = result.get(
                    "answer",
                    "No answer returned by the backend.",
                )

                sources = result.get("sources", [])

                status = result.get("status", "unknown")

                st.markdown(answer)

                if sources:

                    with st.expander("Sources"):

                        for source in sources:
                            st.write(source)

                if status == "error":

                    st.warning(
                        "The backend encountered an error while "
                        "processing this question."
                    )

                # Save assistant response

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except requests.exceptions.HTTPError as error:

                error_message = (
                    f"Backend request failed: {error}"
                )

                try:
                    details = response.json().get(
                        "detail",
                        "No additional error details.",
                    )

                    error_message += f"\n\nDetails: {details}"

                except (ValueError, AttributeError):
                    pass

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )

            except requests.exceptions.Timeout:

                error_message = (
                    "The backend request timed out. "
                    "Please try again."
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )

            except requests.exceptions.RequestException as error:

                error_message = (
                    f"Unable to connect to the backend: {error}"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )
