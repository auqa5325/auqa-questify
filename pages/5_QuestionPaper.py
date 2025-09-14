import os
import time
import json
import re
import hashlib
import pathlib

import boto3
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import parallel_bulk
from requests_aws4auth import AWS4Auth

from bedrockModels import build_request, count_tokens

# --- Load environment variables ---
load_dotenv()
region = os.environ["AWS_REGION"]
bucket_name = os.environ["S3_BUCKET"]
os_domain = os.environ["OS_DOMAIN"]
index_name = "test-auqa"

# --- Pricing (hardcoded) ---
TEXTRACT_RATE = 1.50 / 1000      # $ per page
EMBEDDING_RATE = 0.10 / 1000     # $ per 1k tokens

# --- Setup AWS & OpenSearch ---
session = boto3.Session(region_name=region)
credentials = session.get_credentials().get_frozen_credentials()

awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    region,
    "es",
    session_token=credentials.token
)

client = OpenSearch(
    hosts=[{"host": os_domain, "port": 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

s3 = boto3.client("s3", region_name=region)
textract = boto3.client("textract", region_name=region)
bedrock_rt = session.client("bedrock-runtime", region_name=region)

# --- Load models.json from repo root ---
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "models.json"
if not MODEL_FILE.exists():
    raise FileNotFoundError(f"models.json not found at {MODEL_FILE}. Place it in the script directory.")
with open(MODEL_FILE, "r") as f:
    MODELS = json.load(f)

# --- Helpers ---
def normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def make_chunk_id(course_id: str, chunk_text: str) -> str:
    if len(chunk_text) > 48:
        norm = normalize_text(chunk_text[:48])
    else:
        norm = normalize_text(chunk_text)
    raw = f"{course_id}|{norm}"
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()

def make_page_chunks(docs, chunk_size=6, overlap=2):
    docs_sorted = sorted(docs, key=lambda x: x["page_no"])
    new_chunks = []
    step = chunk_size - overlap
    total_pages = len(docs_sorted)

    for i in range(0, total_pages, step):
        chunk_docs = docs_sorted[i:i+chunk_size]
        if not chunk_docs:
            continue
        chunk_text = "\n\n".join(d["chunk_text"] for d in chunk_docs)
        page_start, page_end = chunk_docs[0]["page_no"], chunk_docs[-1]["page_no"]

        new_chunks.append({
            "chunk_text": chunk_text,
            "course_id": chunk_docs[0]["course_id"],
            "filename": chunk_docs[0]["filename"],
            "page_range": f"{page_start}-{page_end}",
            "pages": [d["page_no"] for d in chunk_docs]
        })
    return new_chunks

def doc_to_action(doc, index_name=index_name):
    doc_id = make_chunk_id(doc["course_id"], doc["chunk_text"])
    return {
        "_op_type": "index",
        "_index": index_name,
        "_id": doc_id,
        "_source": {
            "chunk_text": doc["chunk_text"],
            "course_id": doc["course_id"],
            "filename": doc["filename"],
            "page_range": doc.get("page_range"),
            "pages": doc.get("pages", [doc.get("page_no")]),
            "vector_field": doc["vector_field"]
        }
    }

# --- Streamlit UI ---
st.title("📥 Course PDF Ingestion (Textract first-k pages → Bedrock prompt)")

course_id = st.text_input("Enter Course ID:")
file_key = st.text_input("Enter PDF Key in S3 (e.g., Textbooks/SPM.pdf):")
k_pages = st.number_input("Number of pages to extract (k)", min_value=1, max_value=1000, value=30, step=1)

# model selection from models.json
model_names = [m["name"] for m in MODELS]
selected_model = st.selectbox("Choose Model:", model_names)
model_config = next(m for m in MODELS if m["name"] == selected_model)

# model details
st.markdown(
    f"**Model:** {model_config['name']}  —  max_tokens: {model_config['max_tokens']}  |  "
    f"Input ${model_config['price_input']}/1k · Output ${model_config['price_output']}/1k"
)

if st.button("Ingest & Generate Outline"):
    if not course_id or not file_key:
        st.error("Please enter both course ID and S3 file key")
    else:
        filename = os.path.basename(file_key)
        st.info(f"Starting Textract job on `{file_key}` in `{bucket_name}` (extract first {k_pages} pages)...")

        # --- Start Textract job on S3 reference ---
        start_time = time.time()
        response = textract.start_document_text_detection(
            DocumentLocation={"S3Object": {"Bucket": bucket_name, "Name": file_key}}
        )
        job_id = response["JobId"]

        # Poll until finished
        while True:
            result = textract.get_document_text_detection(JobId=job_id)
            status = result["JobStatus"]
            if status in ["SUCCEEDED", "FAILED"]:
                break
            time.sleep(5)

        elapsed = time.time() - start_time
        if status == "FAILED":
            st.error("Textract job failed!")
            st.stop()

        pages_total = result.get("DocumentMetadata", {}).get("Pages", 0)
        st.success(f"✅ Textract processed {pages_total} pages in {elapsed:.2f} seconds.")

        # --- Collect page texts (only pages <= k_pages) ---
        page_texts, next_token = {}, None
        while True:
            if next_token:
                chunk = textract.get_document_text_detection(JobId=job_id, NextToken=next_token)
            else:
                chunk = textract.get_document_text_detection(JobId=job_id)

            for block in chunk.get("Blocks", []):
                if block.get("BlockType") == "LINE":
                    page_no = block.get("Page")
                    if page_no <= k_pages:
                        page_texts.setdefault(page_no, []).append(block.get("Text"))
            next_token = chunk.get("NextToken")
            if not next_token:
                break

        if not page_texts:
            st.error("No text extracted from the selected pages.")
            st.stop()

        docs = []
        for page_no, lines in page_texts.items():
            docs.append({
                "chunk_text": "\n".join(lines),
                "course_id": course_id,
                "filename": filename,
                "page_no": page_no
            })

        # --- Optionally chunk into multi-page chunks for later indexing (unchanged) ---
        chunked_docs = make_page_chunks(docs)

        # --- Build the prompt (use the structured template with <<COURSE_ID>> placeholder) ---
        template = """
INSTRUCTION:
You are an automated document-understanding assistant specialized in extracting a course outline.

INPUT:
- Course ID: <<COURSE_ID>>
- Document: the extracted text from the first k pages (page markers are added). Do not repeat the document contents.

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

        # assemble document_text with page markers
        doc_pages = []
        for p in sorted(page_texts.keys()):
            text = "\n".join(page_texts[p])
            doc_pages.append(f"===PAGE:{p}===\n{text}")

        document_text = "\n\n".join(doc_pages)

        # create full prompt by replacing placeholder
        user_message = template.replace("<<COURSE_ID>>", course_id) + "\n---DOCUMENT-BEGIN---\n" + document_text + "\n---DOCUMENT-END---\n"

        # token-count for input (approx)
        input_tokens_est = count_tokens(user_message)
        st.caption(f"Approx input tokens for model prompt: {input_tokens_est}")

        # --- Send to Bedrock via build_request + invoke_model ---
        model_id = model_config["id"]
        model_name = model_config["name"]
        # choose an output token limit conservatively (bounded by model max)
        max_gen_len = min(2048, int(model_config.get("max_tokens", 2048)))

        body = build_request(model_id, model_name, user_message, max_gen_len)

        # Call invoke_model with the generated body
        try:
            resp = bedrock_rt.invoke_model(modelId=model_id, body=json.dumps(body))
            model_response = json.loads(resp["body"].read())
        except Exception as e:
            st.error(f"Error invoking model: {e}")
            st.stop()

        # --- Extract generated text and usage if present ---
        generated_text = None
        usage = model_response.get("usage", {})
        # Try a few common response shapes:
        if isinstance(model_response, dict):
            # Anthropics / Claude style
            if "content" in model_response and isinstance(model_response["content"], list):
                generated_text = model_response["content"][0].get("text")
            # Mistral maybe under outputs
            elif "outputs" in model_response and isinstance(model_response["outputs"], list):
                generated_text = model_response["outputs"][0].get("text")
            # Llama style
            elif "generation" in model_response:
                generated_text = model_response.get("generation")
            # Fallback: stringify
            else:
                generated_text = json.dumps(model_response)
        else:
            generated_text = str(model_response)

        # --- If usage missing, approximate tokens ---
        if not usage:
            output_tokens_est = count_tokens(generated_text or "")
        else:
            output_tokens_est = usage.get("outputTokens") or usage.get("output_tokens") or count_tokens(generated_text or "")

        input_tokens_final = usage.get("inputTokens") or usage.get("input_tokens") or input_tokens_est

        # --- Show results ---
        st.subheader("Model Response")
        st.code(generated_text or "<no text returned>")

        # Attempt to parse JSON returned by model
        try:
            parsed = json.loads(generated_text)
            st.success("Parsed JSON from model response")
            st.json(parsed)
        except Exception:
            st.warning("Model output is not valid JSON (attempting to show raw output)")

        # Token & cost metrics
        input_cost = (input_tokens_final / 1000) * float(model_config["price_input"])
        output_cost = (output_tokens_est / 1000) * float(model_config["price_output"])
        st.metric("Input Tokens", f"{input_tokens_final}")
        st.metric("Output Tokens", f"{output_tokens_est}")
        st.metric("Input Cost", f"${input_cost:.4f}")
        st.metric("Output Cost", f"${output_cost:.4f}")
        st.metric("Total Cost", f"${(input_cost+output_cost):.4f}")

        # --- (optional) index the chunked_docs into OpenSearch (same as your previous flow) ---
        # you can index chunked_docs if you want to keep them in the index for QA/hybrid search
        # actions = (doc_to_action(d, index_name=index_name) for d in chunked_docs)
        # errors = []
        # for ok, result in parallel_bulk(client, actions, thread_count=1, chunk_size=50):
        #     if not ok:
        #         errors.append(result)
        # if errors:
        #     st.error(f"Some documents failed to index: {errors[:3]}")
        # else:
        #     st.success(f"Bulk upsert finished. {len(chunked_docs)} chunks indexed/updated.")
