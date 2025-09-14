import os
import json
import time
import pathlib
import streamlit as st
import boto3
from dotenv import load_dotenv
from bedrockModels import (
    converse_with_document_from_s3,
    count_tokens,
)

load_dotenv()
region = os.environ.get("AWS_REGION")
bucket_name = os.environ.get("S3_BUCKET")

st.title("📘 QP Generator — Document Understanding via Bedrock (S3)")
st.markdown(
    "Provide a Course ID and an S3 key to a PDF. The app will send the PDF bytes directly to a Bedrock model in document-understanding mode and ask the model to return a strict JSON of chapters & topics with page numbers."
)

# --- Load available models from models.json in repo root ---
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "models.json"
if not MODEL_FILE.exists():
    st.error(f"models.json not found at {MODEL_FILE}. Please ensure models.json exists in the repo root.")
    st.stop()

with open(MODEL_FILE, "r") as f:
    MODELS = json.load(f)

model_names = [m.get("name", m.get("id")) for m in MODELS]

course_id = st.text_input("Course ID")
s3_key = st.text_input("S3 PDF Key (e.g. Textbooks/Book.pdf)")
selected_model_name = st.selectbox("Choose model (document understanding)", model_names)
model_config = next((m for m in MODELS if m.get("name", m.get("id")) == selected_model_name), None)
if not model_config:
    st.error("Selected model configuration not found in models.json")
    st.stop()

model_id = model_config["id"]

# Provide document format and a max_tokens slider that respects model limits (but keeps slider usable)
document_format = st.selectbox("Document format", ["pdf", "docx", "txt"], index=0)
max_allowed = int(model_config.get("max_tokens", 4096))
# cap the slider upper bound to a reasonable interactive max to avoid huge sliders in UI, but allow large values if model supports them
slider_max = max(4096, min(max_allowed, 65000))
max_tokens = st.slider("Model max tokens (response)", 256, slider_max, min(2048, slider_max), step=128)

st.warning(
    "⚠️ Sending large PDFs directly to Bedrock can be slow and may consume a lot of tokens (input tokens are billed). Use small/sectioned PDFs when possible."
)

if st.button("Run Document Understanding"):
    if not course_id or not s3_key:
        st.error("Provide both Course ID and S3 Key")
    else:
        session = boto3.Session(region_name=region)

        # build a controlled instruction for the model
        # plain-template + replace (avoids f-string brace escaping problems)
        template = """
INSTRUCTION:
You are an automated document-understanding assistant specialized in extracting a course outline.

INPUT:
- Course ID: <<COURSE_ID>>
- Document: a PDF will be attached to this request (do not output or repeat the document contents).

TASK (what to produce):
Return a single, valid JSON object EXACTLY matching the schema below. Do not add any extra keys or top-level fields.

SCHEMA (required):
{
  "course_id": "<course_id>",
  "chapters": [
    {
      "chapter_name": "<string>",
      "start_page": <int>,      # (optional) first page of chapter
      "end_page": <int>,        # (optional) last page of chapter
      "topics": [
        { "topic": "<string>", "pages": [<int>, ...] },
        ...
      ]
    },
    ...
  ]
}

EXAMPLE:
{
  "course_id": "<<COURSE_ID>>",
  "chapters": [
    {
      "chapter_name": "Introduction",
      "start_page": 1,
      "end_page": 5,
      "topics": [
        { "topic": "What is X", "pages": [1] },
        { "topic": "History of X", "pages": [2,3] }
      ]
    }
  ]
}

STRICT RULES:
1) ONLY output the JSON object and nothing else — no explanations, no metadata, no commentary.
2) Use integers for page numbers and a list of integers for `pages`.
3) Prefer minimal page lists: include only pages where the topic actually appears.
4) Use `start_page` and `end_page` for chapter spans when you can determine them; these fields are optional but helpful.
5) If you cannot find chapters/topics, return "chapters": [] (still return a valid JSON object with course_id).
6) Ensure the returned JSON is syntactically valid and parseable by a JSON parser.

Produce the JSON now.
"""

# Replace placeholder with actual course_id
        user_message = template.replace("<<COURSE_ID>>", course_id)

# debug: optionally print or log a small excerpt to ensure substitution worked
# st.caption(repr(user_message[:500]))


        try:
            with st.spinner("Fetching document from S3 and calling Bedrock (this may take a while)..."):
                response_text, raw = converse_with_document_from_s3(
                    session_or_client=session,
                    model_id=model_id,
                    bucket=bucket_name,
                    key=s3_key,
                    user_message=user_message,
                    document_format=document_format,
                    document_name="SPM",
                    max_tokens=max_tokens,
                )

            st.subheader("Model response (raw)")
            st.code(response_text)

            # try parse JSON
            try:
                parsed = json.loads(response_text)
                st.success("✅ Parsed JSON successfully")
                st.json(parsed)

                # basic validation: ensure course_id present
                if parsed.get("course_id") != course_id:
                    st.warning("Parsed JSON course_id does not match input Course ID (model may have altered it).")

            except Exception as e:
                st.error(f"Failed to parse JSON: {e}")
                # offer a helper to extract JSON-like substring
                try:
                    # naive attempt: find first { and last }
                    start = response_text.find("{")
                    end = response_text.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        candidate = response_text[start : end + 1]
                        st.caption("Attempting to parse a JSON substring extracted from the response")
                        try:
                            parsed = json.loads(candidate)
                            st.success("Parsed substring as JSON")
                            st.json(parsed)
                        except Exception:
                            st.code(candidate)
                    else:
                        st.info("No JSON object boundaries found in the model output.")
                except Exception:
                    pass

            # token info for the prompt/message
            tok = count_tokens(user_message)
            st.caption(f"Approx prompt tokens (instruction only): {tok}")

        except Exception as e:
            st.error(f"Error calling Bedrock document understanding flow: {e}")
            raise
