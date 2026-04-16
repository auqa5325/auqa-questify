import os, json, pathlib
import boto3
import pandas as pd
import streamlit as st
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from dotenv import load_dotenv
from bedrockModels import count_tokens, build_request

# ----------------------------------------------------------------------------
# AUQA - Streamlit Question / Answer / Summarization App
# This file is a drop-in replacement for your original app. It includes
# separate, clarified prompt templates for each MODE: QUESTION GENERATION,
# ANSWER EXTRACTION and SUMMARIZATION. The app builds the final prompt by
# combining the selected mode's template with retrieval context and UI guidance.
# ----------------------------------------------------------------------------

# --- Local helpers ---
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

# --- Load environment variables ---
load_dotenv()
region = os.environ["AWS_REGION"]
os_domain = os.environ["OS_DOMAIN"]
index_name = os.environ.get("OS_INDEX", "test-auqa")

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

# --- Streamlit UI ---
st.set_page_config(page_title="AUQA — Questioning Toolkit", layout="wide")
st.title("🔎 AUQA — Questions & Answers Toolkit")

query = st.text_input("Enter Search Query:")
focus_topic = st.text_input("Focus Topic (optional):")
course_id = st.text_input("Course ID (filter results by course)")
ratio = st.slider("Hybrid Search Ratio (BM25 vs Vector)", 0.0, 1.0, 0.5)
num_hits = st.slider("Number of Search Hits (BM25 + Vector)", 1, 20, 3)
max_gen_len = st.slider("Max Generation Length (tokens)", 50, 4096, 512, step=50)

# Model selection
model_names = [m["name"] for m in MODELS]
selected_model = st.selectbox("Choose Model:", model_names)
model_config = next(m for m in MODELS if m["name"] == selected_model)

mode = ["QUESTION GENERATION", "ANSWER EXTRACTION", "SUMMARIZATION"]
selected_mode = st.selectbox("Choose Mode:", mode)

# Default prompt templates per mode
PROMPTS = {
    "QUESTION GENERATION": {
        "title": "Question Generation Prompt",
        "template": (
            "You are an expert exam-question writer.\n"
            "Given the USER QUERY, FOCUS, and the CONTEXT (source passages), generate a list of exam-style questions.\n"
            "Return STRICT JSON: an array of objects with the following keys:\n"
            "  - question (string)\n"
            "  - difficulty (Easy|Medium|Hard)\n"
            "  - blooms_level (Remember|Understand|Apply|Analyze|Evaluate|Create)\n"
            "  - context (short excerpt or location hint where the answer lies)\n"
            "  - page_no (page number or page range for the answer)\n"
            "Rules:\n"
            "  - Generate the TOTAL number of questions requested and strictly follow the difficulty ratio provided.\n"
            "  - Do NOT copy sentences verbatim from the context. Rephrase and create conceptual or scenario-based items.\n"
            "  - If an answer cannot be found in the provided context, mark the question as 'requires_external' = true.\n"
        )
    },
    "ANSWER EXTRACTION": {
        "title": "Answer Extraction Prompt",
        "template": (
            "You are a precise answer extraction assistant.\n"
            "Task: Find a concise, evidence-backed answer to the USER QUERY using ONLY the provided CONTEXT (retrieved chunks).\n"
            "Rules:\n"
            "  - If the context does not contain an answer, return answer = null and explain briefly why.\n"
            "  - Do NOT hallucinate facts; cite exact context segments as the source.\n"
        )
    },
    "SUMMARIZATION": {
        "title": "Summarization Prompt",
        "template": (
            "You are a concise summarization assistant.\n"
            "Task: Produce a clear, structured summary of the provided CONTEXT tailored to the USER QUERY and FOCUS.\n"
            "Rules:\n"
            "  - Prioritize facts and statements present in the context.\n"
            "  - Keep the summary neutral and citation-aware (include page_range or doc ids for claims when possible).\n"
        )
    }
}

# Show model info
st.markdown(
    f"**Max Context:** {model_config['max_tokens']} tokens | "
    f"💰 Input: ${model_config['price_input']}/1k | "
    f"💰 Output: ${model_config['price_output']}/1k"
)

# Editable prompt area (mode-specific default)
default_prompt = PROMPTS[selected_mode]["template"]
prompt_template = st.text_area(
    f"Edit {PROMPTS[selected_mode]['title']}:",
    default_prompt,
    height=220
)

# --- NEW: difficulty UI (only used for question generation) ---
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

# Guidance block to append to the user's prompt (mode-aware)
if selected_mode == "QUESTION GENERATION":
    guidance = f"""
You are an **AI assistant** specialized in **automatic question generation**.

Goals:
1. Generate **{no} questions** in total, distributed based on the given difficulty ratio.
2. Classify each question by Difficulty and Bloom's Taxonomy level.

User Requirements:
- Total Questions: {no}
- Difficulty Ratio: Easy={_easy}, Medium={_medium}, Hard={_hard}

Rules:
- Do NOT copy sentences directly from the context.
- Make questions conceptual; include scenario-based items where suitable.
- Strictly follow the difficulty ratio; adjust proportionally if raw ratios don't sum to 1.
""".strip()
elif selected_mode == "ANSWER EXTRACTION":
    guidance = (
        "You are an extractive assistant. Return a concise answer supported by the exact context passages for a User Query given. "
        "If the answer is not present, be explicit and return answer=null. Prefer short, evidence-backed outputs."
    )
else:  # SUMMARIZATION
    guidance = (
        "You are a summarization assistant. Produce a short structured summary focused on the user query and focus topic. "
        "Include key points and important terms where possible. Keep it factual and cite page ranges/doc ids."
    )

# --- Retrieval logic (only run when query provided) ---
preview_prompt = None
if query:
    # --- BM25 search (with optional course_id filter) ---
    if course_id:
        bm25_query = {
            "size": num_hits,
            "query": {
                "bool": {
                    "filter": [{"term": {"course_id": course_id}}],
                    "must": {"match": {"chunk_text": query}}
                }
            }
        }
    else:
        bm25_query = {"size": num_hits, "query": {"match": {"chunk_text": query}}}

    bm25_resp = client.search(index=index_name, body=bm25_query)
    bm25_docs = [(hit["_id"], hit["_source"]["chunk_text"], hit["_source"].get("page_range", "?"))
                 for hit in bm25_resp["hits"]["hits"]]

    # --- Vector search (with optional course_id filter) ---
    embedding_body = json.dumps({"inputText": query})
    resp = bedrock.invoke_model(modelId="amazon.titan-embed-text-v2:0", body=embedding_body)
    emb = json.loads(resp["body"].read())["embedding"]

    if course_id:
        vector_query = {
            "size": num_hits,
            "query": {
                "bool": {
                    "filter": [{"term": {"course_id": course_id}}],
                    "must": {"knn": {"vector_field": {"vector": emb, "k": num_hits}}}
                }
            }
        }
    else:
        vector_query = {"size": num_hits, "query": {"knn": {"vector_field": {"vector": emb, "k": num_hits}}}}

    vector_resp = client.search(index=index_name, body=vector_query)
    vector_docs = [(hit["_id"], hit["_source"]["chunk_text"], hit["_source"].get("page_range", "?"))
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
    # For extraction and summarization, provide additional metadata to the model (doc id + page_range)
    context_entries = []
    for did, txt, pr in final_chunks:
        # keep each chunk reasonably short in the prompt
        context_entries.append(f"[DOC_ID:{did} | PAGES:{pr}] {txt}")

    context = "\n\n".join(context_entries)

    # --- Show retrieved chunks ---
    df_chunks = pd.DataFrame(final_chunks, columns=["Doc ID", "Chunk Text", "Page Range"])
    st.subheader("📄 Retrieved Chunks (deduplicated by Doc ID)")
    st.dataframe(df_chunks)

    # --- Preview prompt (mode-aware) ---
    preview_prompt = (
        f"{prompt_template}\n\n"
        f"Query: {query}\n"
        f"Focus: {focus_topic}\n"
        f"Context: {context}\n\n"
        f"{guidance}"
    )

    token_count = count_tokens(preview_prompt)
    st.info(f"📏 Full Prompt Length: **{token_count} tokens** (limit {model_config['max_tokens']})")

# --- ACTION: Generate ---
if st.button("Generate"):
    if not query:
        st.error("Please enter a query")
    else:
        # ensure preview_prompt exists (should if query provided earlier)
        prompt_to_send = preview_prompt or (prompt_template + f"\nQuery: {query}\nFocus: {focus_topic}")

        # Truncate if needed
        prompt, truncated = truncate_to_limit(prompt_to_send, model_config["max_tokens"]) if model_config.get("max_tokens") else (prompt_to_send, False)
        if truncated:
            st.warning(f"⚠️ Prompt truncated to fit {model_config['max_tokens']} tokens.")

        model_id = model_config["id"]
        # For amazon.nova family use converse API (returns structured output), else use invoke_model
        if model_id.startswith("amazon.nova"):
            conversation = [{"role": "user", "content": [{"text": prompt}]}]
            resp = bedrock.converse(modelId=model_id, messages=conversation,
                                     inferenceConfig={"maxTokens": max_gen_len, "temperature": 0.0, "topP": 0.9})
            generated_text = resp["output"]["message"]["content"][0]["text"]
            usage = resp.get("usage", {})
        else:
            body = build_request(model_id, selected_model, prompt, max_gen_len)
            resp = bedrock.invoke_model(modelId=model_id, body=json.dumps(body))
            model_response = json.loads(resp["body"].read())
            usage = model_response.get("usage", {})

            if "anthropic" in model_id or "claude" in selected_model.lower():
                generated_text = model_response["content"][0]["text"]
            elif "llama" in model_id:
                generated_text = model_response.get("generation") or str(model_response)
            elif "mistral" in model_id:
                generated_text = model_response.get("outputs", [{"text": str(model_response)}])[0].get("text")
            else:
                generated_text = str(model_response)

        # --- Show output ---
        st.subheader("Generated Output")

        # Try to pretty-print JSON if it looks like JSON
        try:
            parsed = json.loads(generated_text)
            st.json(parsed)
        except Exception:
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

# Footer guidance
st.markdown("---")
st.caption("Tip: Edit the mode-specific prompt above for custom behavior. For question generation, use the difficulty controls; for extraction, keep queries concise; for summarization, set a short focus to get targeted summaries.")
