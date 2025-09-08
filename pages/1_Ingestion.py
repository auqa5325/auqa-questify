import os, time, json, re, hashlib
import boto3
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import parallel_bulk
from requests_aws4auth import AWS4Auth

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

# --- Helpers ---
def normalize_text(s: str) -> str:
    """Normalize text to avoid minor whitespace/case changes causing new IDs."""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def make_chunk_id(course_id: str, chunk_text: str) -> str:
    """Deterministic ID based on course_id + chunk_text."""
    if(len(chunk_text)>48):
        norm = normalize_text(chunk_text[:48])
    else:
        norm = normalize_text(chunk_text)
    raw = f"{course_id}|{norm}"
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()

def make_page_chunks(docs, chunk_size=6, overlap=2):
    """Re-chunk page-level docs into multi-page chunks."""
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
    """Convert doc into OpenSearch bulk index action with deterministic ID."""
    doc_id = make_chunk_id(doc["course_id"], doc["chunk_text"])
    return {
        "_op_type": "index",    # overwrite if exists
        "_index": index_name,
        "_id": doc_id,
        "_source": {
            "chunk_text": doc["chunk_text"],
            "course_id": doc["course_id"],
            "filename": doc["filename"],
            "page_range": doc.get("page_range"),
            "pages": doc.get("pages", [doc.get("page_no")]),
            "vector_field": doc["vector_field"]   # Titan embedding
        }
    }

# --- Streamlit UI ---
st.title("📥 Course PDF Ingestion")

course_id = st.text_input("Enter Course ID:")
file_key = st.text_input("Enter PDF Key in S3 (e.g., Textbooks/SPM.pdf):")

if st.button("Ingest"):
    if not course_id or not file_key:
        st.error("Please enter both course ID and S3 file key")
    else:
        filename = os.path.basename(file_key)
        st.info(f"Starting Textract job on `{file_key}` in `{bucket_name}`...")

        # --- Start Textract job ---
        start_time = time.time()   # ✅ capture start
        response = textract.start_document_text_detection(
            DocumentLocation={"S3Object": {"Bucket": bucket_name, "Name": file_key}}
        )
        job_id = response["JobId"]

        # Wait until finished
        while True:
            result = textract.get_document_text_detection(JobId=job_id)
            status = result["JobStatus"]
            if status in ["SUCCEEDED", "FAILED"]:
                break
            time.sleep(5)

        end_time = time.time()     # ✅ capture end
        elapsed = end_time - start_time

        if status == "FAILED":
            st.error("Textract job failed!")
            st.stop()

        pages = result["DocumentMetadata"]["Pages"]
        st.success(f"✅ Textract processed {pages} pages in {elapsed:.2f} seconds.")

        # --- Collect page texts ---
        page_texts, next_token = {}, None
        while True:
            if next_token:
                result = textract.get_document_text_detection(JobId=job_id, NextToken=next_token)
            else:
                result = textract.get_document_text_detection(JobId=job_id)

            for block in result["Blocks"]:
                if block["BlockType"] == "LINE":
                    page_no = block["Page"]
                    page_texts.setdefault(page_no, []).append(block["Text"])
            next_token = result.get("NextToken")
            if not next_token:
                break

        docs = []
        for page_no, lines in page_texts.items():
            docs.append({
                "chunk_text": "\n".join(lines),
                "course_id": course_id,
                "filename": filename,
                "page_no": page_no
            })

        # --- Chunk into multi-page ---
        chunked_docs = make_page_chunks(docs)

        

        # --- Embed using Titan (filter by token threshold) ---
        MODEL_ID = "amazon.titan-embed-text-v2:0"
        MIN_TOKENS = 24   # or expose as Streamlit control
        total_tokens, all_embeddings = 0, []
        skipped = 0

        for d in chunked_docs:
            body = json.dumps({"inputText": d["chunk_text"]})
            resp = bedrock_rt.invoke_model(modelId=MODEL_ID, body=body)
            result = json.loads(resp["body"].read())

            # ✅ use Titan's reported token count if available
            input_tokens = result.get("inputTextTokenCount")
            

           

            d["tokens"] = input_tokens
            d["vector_field"] = result["embedding"]

            all_embeddings.append(d)
            total_tokens += input_tokens

  


        # --- Bulk index into OpenSearch ---
        actions = (doc_to_action(d, index_name=index_name) for d in all_embeddings)
        errors = []
        for ok, result in parallel_bulk(client, actions, thread_count=1, chunk_size=50):
            if not ok:
                errors.append(result)

        if errors:
            st.error(f"❌ Some documents failed to index: {errors[:3]}")
        else:
            st.success(f"✅ Bulk upsert finished. {len(all_embeddings)} chunks indexed/updated.")

        # --- Show token stats + costs ---
        df = pd.DataFrame([{"Chunk": d["page_range"], "Tokens": d["tokens"]} for d in all_embeddings])
        st.dataframe(df)

        textract_cost = pages * TEXTRACT_RATE
        embed_cost = (total_tokens / 1000) * EMBEDDING_RATE
        total_cost = textract_cost + embed_cost

        st.metric("Textract Cost", f"${textract_cost:.4f}")
        st.metric("Embedding Cost", f"${embed_cost:.4f}")
        st.metric("Total Cost", f"${total_cost:.4f}")
