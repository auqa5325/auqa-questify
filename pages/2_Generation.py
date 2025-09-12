import os, json, pathlib
import boto3
import pandas as pd
import streamlit as st
import tiktoken
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
region = os.environ["AWS_REGION"]
os_domain = os.environ["OS_DOMAIN"]
index_name = "test-auqa"

# --- Load models config from outer directory ---
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "models.json"
with open(MODEL_FILE, "r") as f:
    MODELS = json.load(f)

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

bedrock = session.client("bedrock-runtime", region_name=region)

# --- Token helpers ---
def count_tokens(text: str) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


def truncate_to_limit(text: str, max_tokens: int, buffer: int = 2500):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) > (max_tokens - buffer):
        return enc.decode(tokens[: max_tokens - buffer]), True
    return text, False


# --- NEW: difficulty helpers (ratios) ---
def normalize_ratios(e: float, m: float, h: float):
    s = e + m + h
    if s <= 0:
        return 1/3, 1/3, 1/3
    return e/s, m/s, h/s


# --- Helper to build request ---
def build_request(model_id, model_name, prompt, max_gen_len):
    if model_id.startswith("amazon.nova"):
        return {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_gen_len, "temperature": 0.5, "topP": 0.9}
        }
    elif "anthropic" in model_id or "claude" in model_name.lower():
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_gen_len,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
        }
    elif "llama" in model_id:
        formatted_prompt = f"""
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
        return {"prompt": formatted_prompt, "max_gen_len": max_gen_len, "temperature": 0.5}
    elif "mistral" in model_id:
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        return {"prompt": formatted_prompt, "max_tokens": max_gen_len, "temperature": 0.5}
    else:
        return {"prompt": prompt, "max_tokens": max_gen_len, "temperature": 0.5}

# --- Streamlit UI ---
st.title("🔎 Question Generation")

query = st.text_input("Enter Search Query:")
focus_topic = st.text_input("Focus Topic (optional):")
ratio = st.slider("Hybrid Search Ratio (BM25 vs Vector)", 0.0, 1.0, 0.5)
num_hits = st.slider("Number of Search Hits (BM25 + Vector)", 1, 20, 3)
max_gen_len = st.slider("Max Generation Length (tokens)", 50, 4096, 512, step=50)

# Model selection
model_names = [m["name"] for m in MODELS]
selected_model = st.selectbox("Choose Model:", model_names)
model_config = next(m for m in MODELS if m["name"] == selected_model)

# Show model info
st.markdown(
    f"**Max Context:** {model_config['max_tokens']} tokens | "
    f"💰 Input: ${model_config['price_input']}/1k | "
    f"💰 Output: ${model_config['price_output']}/1k"
)

prompt_template = st.text_area(
    "Edit Question Generation Prompt:",
    '''Generate exam-style questions based on the retrieved context and query.
    Expected Output (strict JSON):
    [
      {{
    "question": "...",
    "difficulty": "Easy|Medium|Hard",
    "blooms_level": "Remember|Understand|Apply|Analyze|Evaluate|Create"
    "context":"Where the answer for the question lies"
    "page_no":"For the answer of question"
      }}
    ]'''
    
)

# --- NEW: difficulty UI (doesn't change generation logic — only enriches the prompt) ---
no = st.number_input("Total Questions", min_value=1, max_value=100, value=5, step=1)
col1, col2, col3 = st.columns(3)
with col1:
    easy_raw = st.number_input("Easy ratio", min_value=0.0, max_value=1.0, value=0.34, step=0.01, format="%.2f")
with col2:
    medium_raw = st.number_input("Medium ratio", min_value=0.0, max_value=1.0, value=0.33, step=0.01, format="%.2f")
with col3:
    hard_raw = st.number_input("Hard ratio", min_value=0.0, max_value=1.0, value=0.33, step=0.01, format="%.2f")

# normalize and show
_easy, _medium, _hard = normalize_ratios(easy_raw, medium_raw, hard_raw)
st.caption(f"Normalized ratios → Easy={_easy:.2f}, Medium={_medium:.2f}, Hard={_hard:.2f}")

# Guidance block to append to the user's prompt
ratio_guidance = f"""
You are an **AI assistant** specialized in **automatic question generation**.
Your task is to create insightful and meaningful questions based on the provided **context** and **user query**.

---

### **Goals**
1. Generate **{no} questions** in total, distributed based on the given difficulty ratio:
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
- Total Questions: {no}
- Difficulty Ratio: Easy={_easy}, Medium={_medium}, Hard={_hard}




Rules:
- Do NOT copy sentences directly from the context.
- Make questions conceptual; include scenario-based items where suitable.
- Strictly follow the difficulty ratio; adjust proportionally if raw ratios don't sum to 1.
""".strip()

if query:
    # --- BM25 search ---
    bm25_resp = client.search(
        index=index_name,
        body={"size": num_hits, "query": {"match": {"chunk_text": query}}}
    )
    bm25_docs = [(hit["_id"], hit["_source"]["chunk_text"], hit["_source"].get("page_range", "?"))
                 for hit in bm25_resp["hits"]["hits"]]

    # --- Vector search ---
    embedding_body = json.dumps({"inputText": query})
    resp = bedrock.invoke_model(modelId="amazon.titan-embed-text-v2:0", body=embedding_body)
    emb = json.loads(resp["body"].read())["embedding"]

    vector_resp = client.search(
        index=index_name,
        body={"size": num_hits, "query": {"knn": {"vector_field": {"vector": emb, "k": num_hits}}}}
    )
    vector_docs = [(hit["_id"], hit["_source"]["chunk_text"], hit["_source"].get("page_range", "?"))
                   for hit in vector_resp["hits"]["_hits"]] if "_hits" in vector_resp.get("hits", {}) else [
                   (hit["_id"], hit["_source"]["chunk_text"], hit["_source"].get("page_range", "?"))
                   for hit in vector_resp.get("hits", {}).get("hits", [])]

    # --- Weighted merge & dedup by doc_id ---
    bm25_take = int(num_hits * ratio)
    vector_take = num_hits - bm25_take
    selected = bm25_docs[:bm25_take] + vector_docs[:vector_take]

    seen_ids, final_chunks = set(), []
    for doc_id, text, page_range in selected:
        if doc_id not in seen_ids:
            final_chunks.append((doc_id, text, page_range))
            seen_ids.add(doc_id)

    # --- Build context ---
    context = "\n".join([t for _, t, _ in final_chunks])

    # --- Show retrieved chunks ---
    df_chunks = pd.DataFrame(final_chunks, columns=["Doc ID", "Chunk Text", "Page Range"])
    st.subheader("📄 Retrieved Chunks (deduplicated by Doc ID)")
    st.dataframe(df_chunks)

    # --- Preview prompt (UNCHANGED logic: still uses prompt_template; we only append guidance) ---
    preview_prompt = (
        f"{prompt_template}\n\n"
        f"Query: {query}\n"
        f"Focus: {focus_topic}\n"
        f"Context: {context}\n\n"
        f"{ratio_guidance}"
    )
    token_count = count_tokens(preview_prompt)
    st.info(f"📏 Full Prompt Length: **{token_count} tokens** (limit {model_config['max_tokens']})")

if st.button("Generate Questions"):
    if not query:
        st.error("Please enter a query")
    else:
        # Truncate prompt if needed
        prompt, truncated = truncate_to_limit(preview_prompt, model_config["max_tokens"])
        if truncated:
            st.warning(f"⚠️ Prompt truncated to fit {model_config['max_tokens']} tokens.")

        model_id = model_config["id"]
        body = build_request(model_id, selected_model, prompt, max_gen_len)

        # --- Call Bedrock ---
        if model_id.startswith("amazon.nova"):
            resp = bedrock.converse(
                modelId=model_id,
                messages=body["messages"],
                inferenceConfig=body["inferenceConfig"]
            )
            generated_text = resp["output"]["message"]["content"][0]["text"]
            usage = resp.get("usage", {})
        else:
            resp = bedrock.invoke_model(modelId=model_id, body=json.dumps(body))
            model_response = json.loads(resp["body"].read())
            usage = model_response.get("usage", {})

            if "anthropic" in model_id or "claude" in selected_model.lower():
                generated_text = model_response["content"][0]["text"]
            elif "llama" in model_id:
                generated_text = model_response["generation"]
            elif "mistral" in model_id:
                generated_text = model_response["outputs"][0]["text"]
            else:
                generated_text = str(model_response)

        # --- Show output ---
        st.subheader("Generated Questions")
        st.write(generated_text)

        # --- Token usage & cost ---
        if usage:
            input_tokens = usage.get("inputTokens") or usage.get("input_tokens", 0)
            output_tokens = usage.get("outputTokens") or usage.get("output_tokens", 0)
            approx = False
        else:
            input_tokens = count_tokens(prompt)
            output_tokens = count_tokens(generated_text)
            approx = True

        input_cost = (input_tokens / 1000) * model_config["price_input"]
        output_cost = (output_tokens / 1000) * model_config["price_output"]
        total_cost = input_cost + output_cost

        st.metric("Input Tokens", f"{input_tokens}{' (approx)' if approx else ''}")
        st.metric("Output Tokens", f"{output_tokens}{' (approx)' if approx else ''}")
        st.metric("Input Cost", f"${input_cost:.4f}{' (approx)' if approx else ''}")
        st.metric("Output Cost", f"${output_cost:.4f}{' (approx)' if approx else ''}")
        st.metric("Total Cost", f"${total_cost:.4f}{' (approx)' if approx else ''}")

        