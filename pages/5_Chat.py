import os
import time
import json
import uuid
import pathlib
import boto3
import streamlit as st
from dotenv import load_dotenv

from bedrockModels import (
    count_tokens,
    build_request,
    converse_stream_claude,
)

# ----------------------------------------------------------------------------
# AUQA — Generic Chat with optional PDF + Page Range (Claude Streaming)
# ----------------------------------------------------------------------------

# --- Load env ---
load_dotenv()
REGION = os.environ["AWS_REGION"]
S3_BUCKET = os.environ["S3_BUCKET"]

# --- Load model config ---
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "models.json"

with open(MODEL_FILE, "r") as f:
    MODELS = json.load(f)

# --- AWS clients ---
session = boto3.Session(region_name=REGION)
bedrock = session.client("bedrock-runtime", region_name=REGION)
textract = session.client("textract", region_name=REGION)
s3 = session.client("s3", region_name=REGION)

# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AUQA — Chat", layout="wide")
st.title("💬 AUQA — Generic LLM Chat")

# --- Model selection ---
model_names = [m["name"] for m in MODELS]
selected_model = st.selectbox("Choose Model", model_names)
model_config = next(m for m in MODELS if m["name"] == selected_model)
model_id = model_config["id"]

st.markdown(
    f"**Max Context:** {model_config['max_tokens']} tokens | "
    f"💰 Input: ${model_config['price_input']}/1k | "
    f"💰 Output: ${model_config['price_output']}/1k"
)

# --- System prompt ---
system_prompt = st.text_area(
    "System Prompt",
    "You are a helpful assistant. Use the document only if it is provided.",
    height=120
)

# --- User query ---
user_query = st.text_area("Your Message", height=180)

# --- PDF upload ---
uploaded_pdf = st.file_uploader("Upload PDF (optional)", type=["pdf"])

# ----------------------------------------------------------------------------
# Page range (ONLY if PDF is uploaded)
# ----------------------------------------------------------------------------
page_start = page_end = None

if uploaded_pdf:
    st.subheader("📄 Select PDF Page Range")
    col1, col2 = st.columns(2)
    with col1:
        page_start = st.number_input("From Page", min_value=1, value=1)
    with col2:
        page_end = st.number_input(
            "To Page",
            min_value=page_start,
            value=page_start
        )

# ----------------------------------------------------------------------------
# Controls
# ----------------------------------------------------------------------------
max_output_tokens = st.slider("Max Output Tokens", 50, 12000, 512, step=50)
temperature = st.slider("Temperature", 0.0, 1.0, 0.5, step=0.05)

# ----------------------------------------------------------------------------
# Send
# ----------------------------------------------------------------------------
if st.button("Send"):
    if not user_query and not uploaded_pdf:
        st.error("Please enter a message or upload a PDF.")
        st.stop()

    pdf_text = ""

    # ------------------------------------------------------------------------
    # PDF → S3 → Textract → Page Range
    # ------------------------------------------------------------------------
    if uploaded_pdf:
        with st.spinner("Uploading PDF to S3..."):
            s3_key = f"textract-chat/{uuid.uuid4()}.pdf"
            s3.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Body=uploaded_pdf.read()
            )

        with st.spinner("Running Textract..."):
            start_resp = textract.start_document_text_detection(
                DocumentLocation={
                    "S3Object": {"Bucket": S3_BUCKET, "Name": s3_key}
                }
            )
            job_id = start_resp["JobId"]

            while True:
                resp = textract.get_document_text_detection(JobId=job_id)
                status = resp["JobStatus"]
                if status in ["SUCCEEDED", "FAILED"]:
                    break
                time.sleep(3)

            if status == "FAILED":
                st.error("Textract failed.")
                st.stop()

        page_lines = {}
        next_token = None

        while True:
            if next_token:
                resp = textract.get_document_text_detection(
                    JobId=job_id, NextToken=next_token
                )
            else:
                resp = textract.get_document_text_detection(JobId=job_id)

            for block in resp["Blocks"]:
                if block["BlockType"] == "LINE":
                    page_no = block["Page"]
                    if page_start <= page_no <= page_end:
                        page_lines.setdefault(page_no, []).append(block["Text"])

            next_token = resp.get("NextToken")
            if not next_token:
                break

        if not page_lines:
            st.error("No text found in selected page range.")
            st.stop()

        pdf_text = "\n\n".join(
            f"[Page {p}]\n" + "\n".join(lines)
            for p, lines in sorted(page_lines.items())
        )

    # ------------------------------------------------------------------------
    # Build prompt
    # ------------------------------------------------------------------------
    final_prompt = system_prompt.strip()

    if user_query:
        final_prompt += f"\n\nUser Query:\n{user_query}"

    if pdf_text:
        final_prompt += (
            f"\n\nReference Document (Pages {page_start}-{page_end}):\n"
            f"{pdf_text}"
        )

    token_count = count_tokens(final_prompt)
    st.info(f"📏 Prompt Tokens: {token_count}")

    # ------------------------------------------------------------------------
    # Bedrock call (Claude STREAMING)
    # ------------------------------------------------------------------------
    output_text = ""
    usage = {}

    # -------- Amazon Nova (non-stream) --------
    if model_id.startswith("amazon.nova"):
        messages = [{"role": "user", "content": [{"text": final_prompt}]}]
        resp = bedrock.converse(
            modelId=model_id,
            messages=messages,
            inferenceConfig={
                "maxTokens": max_output_tokens,
                "temperature": temperature,
                "topP": 0.9
            }
        )
        output_text = resp["output"]["message"]["content"][0]["text"]
        usage = resp.get("usage", {})

        st.subheader("🧠 LLM Response")
        st.write(output_text)

    # -------- Claude (STREAMING) --------
    elif "anthropic" in model_id or "claude" in selected_model.lower():
        st.subheader("🧠 LLM Response")
        placeholder = st.empty()

        live_text = ""
        chunks = []

        for chunk in converse_stream_claude(
            bedrock,
            model_id=model_id,
            user_message=final_prompt,
            max_tokens=max_output_tokens,
            temperature=temperature,
        ):
            chunks.append(chunk)
            live_text += chunk
            placeholder.markdown(live_text)

        output_text = "".join(chunks)
        usage = {}  # Claude streaming has no usage yet

    # -------- Others (invoke_model) --------
    else:
        body = build_request(
            model_id,
            selected_model,
            final_prompt,
            max_output_tokens
        )

        resp = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps(body)
        )

        model_response = json.loads(resp["body"].read())
        usage = model_response.get("usage", {})

        if "llama" in model_id:
            output_text = model_response.get("generation", "")
        else:
            output_text = str(model_response)

        st.subheader("🧠 LLM Response")
        st.write(output_text)

    # ------------------------------------------------------------------------
    # Cost & Usage
    # ------------------------------------------------------------------------
    in_tok = usage.get("inputTokens", count_tokens(final_prompt))
    out_tok = usage.get("outputTokens", count_tokens(output_text))

    input_cost = (in_tok / 1000) * model_config["price_input"]
    output_cost = (out_tok / 1000) * model_config["price_output"]

    st.markdown("### 📊 Usage & Cost")
    st.metric("Input Tokens", in_tok)
    st.metric("Output Tokens", out_tok)
    st.metric("Total Cost", f"${(input_cost + output_cost):.4f}")
