import os, time, json, pathlib
import boto3
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from bedrockModels import build_request, count_tokens

# --- Helpers ---
def truncate_to_limit(text: str, max_tokens: int, buffer: int = 2500):
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) > (max_tokens - buffer):
        return enc.decode(tokens[: max_tokens - buffer]), True
    return text, False

def normalize_ratios(e: float, m: float, h: float):
    s = e + m + h
    if s <= 0:
        return 1/3, 1/3, 1/3
    return e/s, m/s, h/s

# --- Load environment ---
load_dotenv()
region = os.environ["AWS_REGION"]
bucket_name = os.environ["S3_BUCKET"]
os_domain = os.environ["OS_DOMAIN"]
index_name = "test-auqa"

# --- Load models.json ---
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "models.json"
with open(MODEL_FILE, "r") as f:
    MODELS = json.load(f)

# --- AWS setup ---
session = boto3.Session(region_name=region)
credentials = session.get_credentials().get_frozen_credentials()
awsauth = AWS4Auth(
    credentials.access_key, credentials.secret_key,
    region, "es", session_token=credentials.token
)
client = OpenSearch(
    hosts=[{"host": os_domain, "port": 443}],
    http_auth=awsauth, use_ssl=True, verify_certs=True,
    connection_class=RequestsHttpConnection
)
s3 = boto3.client("s3", region_name=region)
textract = boto3.client("textract", region_name=region)
bedrock = session.client("bedrock-runtime", region_name=region)

# --- Streamlit UI ---
st.title("📘 Syllabus → Unit-wise Question Generator")

course_id = st.text_input("Enter Course ID:", "OS")
file_key = st.text_input("Enter PDF Key in S3:", "syllabus/CS23501.Syllabus.pdf")

# Model selection
model_names = [m["name"] for m in MODELS]
selected_model = st.selectbox("Choose Model:", model_names)
model_config = next(m for m in MODELS if m["name"] == selected_model)

st.markdown(
    f"**Model:** {model_config['name']}  | "
    f"Max {model_config['max_tokens']} tokens | "
    f"💰 Input: ${model_config['price_input']}/1k · "
    f"Output: ${model_config['price_output']}/1k"
)

# session state
if "units_parsed" not in st.session_state:
    st.session_state.units_parsed = None

# Step 1: Extract syllabus into JSON
if st.button("Ingest PDF & Extract Syllabus"):
    filename = os.path.basename(file_key)
    st.info(f"Running Textract on `{file_key}` ...")

    resp = textract.start_document_text_detection(
        DocumentLocation={"S3Object": {"Bucket": bucket_name, "Name": file_key}}
    )
    job_id = resp["JobId"]

    while True:
        result = textract.get_document_text_detection(JobId=job_id)
        status = result["JobStatus"]
        if status in ["SUCCEEDED", "FAILED"]:
            break
        time.sleep(5)

    if status == "FAILED":
        st.error("Textract failed")
        st.stop()

    page_texts = {}
    next_token = None
    while True:
        if next_token:
            chunk = textract.get_document_text_detection(JobId=job_id, NextToken=next_token)
        else:
            chunk = textract.get_document_text_detection(JobId=job_id)

        for block in chunk.get("Blocks", []):
            if block.get("BlockType") == "LINE":
                page_no = block.get("Page")
                page_texts.setdefault(page_no, []).append(block.get("Text"))
        next_token = chunk.get("NextToken")
        if not next_token:
            break

    full_text = "\n\n".join("\n".join(lines) for lines in [page_texts[p] for p in sorted(page_texts.keys())])

    template ="""
INSTRUCTION:
You are an automated assistant specialized in extracting a course outline.

INPUT:
- Course ID: <<COURSE_ID>>
- Document: the extracted text from the PDF. Do not repeat the text.

TASK:
Return ONLY a JSON object, with schema:
{
  "course_id": "<course_id>",
  "units": [
    { "unit_no": <int>, "unit_name": "<string>", "topics": ["<string>", ...] }
  ]
}

RULES:
1) ONLY output the JSON object and nothing else — no explanations, no commentary.
2) Use integers for unit_no.                
3) If you cannot find any units/topics, return "units": [] (still return a valid JSON object with course_id).
4) Ensure the JSON is valid.
Produce the JSON now. No other comment or text like '''json''' 
""".strip()

    user_message = template.replace("<<COURSE_ID>>", course_id) + "\n---DOCUMENT-BEGIN---\n" + full_text + "\n---DOCUMENT-END---"

    body = build_request(model_config["id"], selected_model, user_message, 2048)
    resp = bedrock.invoke_model(modelId=model_config["id"], body=json.dumps(body))
    model_response = json.loads(resp["body"].read())

    if "content" in model_response:  # Anthropic
        generated_text = model_response["content"][0]["text"]
    elif "outputs" in model_response:  # Mistral
        generated_text = model_response["outputs"][0]["text"]
    else:
        generated_text = str(model_response)

    st.subheader("Extracted JSON")
    st.code(generated_text)

    try:
        parsed = json.loads(generated_text)
        st.session_state.units_parsed = parsed.get("units", [])
        st.success("Syllabus parsed successfully")
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")

# Step 2: Configure and generate questions per unit
if st.session_state.units_parsed:
    st.subheader("Select Units for Question Generation")

    unit_settings = {}
    for unit in st.session_state.units_parsed:
        unit_no, unit_name = unit["unit_no"], unit["unit_name"]
        with st.expander(f"Unit {unit_no}: {unit_name}"):
            enabled = st.checkbox("Enable", key=f"enable_{unit_no}")
            if enabled:
                n_q = st.number_input("Number of Questions", 1, 50, 5, key=f"nq_{unit_no}")
                c1, c2, c3 = st.columns(3)
                with c1:
                    easy_raw = st.number_input("Easy ratio", 0.0, 1.0, 0.34, step=0.01, key=f"easy_{unit_no}")
                with c2:
                    med_raw = st.number_input("Medium ratio", 0.0, 1.0, 0.33, step=0.01, key=f"med_{unit_no}")
                with c3:
                    hard_raw = st.number_input("Hard ratio", 0.0, 1.0, 0.33, step=0.01, key=f"hard_{unit_no}")
                _e, _m, _h = normalize_ratios(easy_raw, med_raw, hard_raw)
                unit_settings[unit_no] = {
                    "name": unit_name,
                    "topics": unit["topics"],
                    "nq": n_q,
                    "ratio": (_e, _m, _h)
                }

    if st.button("🚀 Generate Questions"):
        for unit_no, conf in unit_settings.items():
            st.subheader(f"📘 Unit {unit_no}: {conf['name']}")
            query = " ".join(conf["topics"])

            # --- Hybrid Retrieval ---
            # BM25
            bm25_query = {"size": 5, "query": {"match": {"chunk_text": query}}}
            bm25_resp = client.search(index=index_name, body=bm25_query)
            bm25_docs = [(hit["_id"], hit["_source"]["chunk_text"], hit["_source"].get("page_range", "?"))
                         for hit in bm25_resp["hits"]["hits"]]

            # Vector
            embedding_body = json.dumps({"inputText": query})
            resp = bedrock.invoke_model(modelId="amazon.titan-embed-text-v2:0", body=embedding_body)
            emb = json.loads(resp["body"].read())["embedding"]

            vector_query = {"size": 5, "query": {"knn": {"vector_field": {"vector": emb, "k": 5}}}}
            vector_resp = client.search(index=index_name, body=vector_query)
            vector_docs = [(hit["_id"], hit["_source"]["chunk_text"], hit["_source"].get("page_range", "?"))
                           for hit in vector_resp.get("hits", {}).get("hits", [])]

            # Merge
            selected = bm25_docs[:1] + vector_docs[:4]
            context = "\n".join([t for _, t, _ in selected])

            df_chunks = pd.DataFrame(selected, columns=["Doc ID", "Chunk Text", "Page Range"])
            st.dataframe(df_chunks)

            # --- Build prompt ---
            _e, _m, _h = conf["ratio"]
            guidance = f"""
You are an **AI assistant** specialized in **automatic question generation**.
Your task is to create insightful and meaningful questions based on the provided **context** and **user query**.

---

### **Goals**
1. Generate {conf['nq']} exam-style questions for Unit {unit_no}: {conf['name']}., distributed based on the given difficulty ratio:
2. Ensure the ratio is **strictly followed**. If the sum of ratios is not exactly 1.0, adjust proportionally.
3. Classify each question by:
   - **Difficulty Level** → Easy / Medium / Hard
   - **Bloom's Taxonomy Level** → Choose one of:
        * Remember
        * Understand
        * Apply
        * Analyze
        * Evaluate
        * Create

---

**User Requirements**:
- Total Questions:  {conf['nq']}
- Difficulty Ratio: - Easy={_e:.2f}, Medium={_m:.2f}, Hard={_h:.2f}



Rules:
- Do NOT copy sentences directly from the context.
- Strictly follow the difficulty ratio; adjust proportionally if raw ratios don't sum to 1.
""".strip()

            prompt = f"""

Context:
{context}

{guidance}

Expected JSON:
[
  {{
    "question": "...",
    "difficulty": "Easy|Medium|Hard",
    "blooms_level": "Remember|Understand|Apply|Analyze|Evaluate|Create",
    "context": "...",
    "page_no": "..."
  }}
]
RULES:
1.Make questions conceptual and scenario based, and also include one or two numerical problem questions.
2.Question should test the understanding of the topic.
3.ONLY output a JSON array of question objects as shown above; do not include any other text or commentary.

Just output the Json code only , No other text at the start or end like '''json ''' needed.
""".strip()

            prompt, _ = truncate_to_limit(prompt, model_config["max_tokens"])

            body = build_request(model_config["id"], selected_model, prompt, 4096)
            resp = bedrock.invoke_model(modelId=model_config["id"], body=json.dumps(body))
            model_response = json.loads(resp["body"].read())
            if "outputs" in model_response:
                generated_text = model_response["outputs"][0]["text"]
            elif "content" in model_response:
                generated_text = model_response["content"][0]["text"]
            else:
                generated_text = str(model_response)

            st.code(generated_text)
            try:
                df = pd.DataFrame(json.loads(generated_text))
                st.dataframe(df)
            except:
                st.warning("Model did not return valid JSON")
