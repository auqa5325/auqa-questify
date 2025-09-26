import os
import time
import json
import pathlib
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from bedrockModels import build_request, count_tokens
import boto3
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from qp_pdf_generator import render_qp_pdf
import math
from typing import List, Tuple
from datetime import datetime
# ---------------- Environment & simple config --------------------------------
load_dotenv()

OUT_DIR = os.environ.get("AUQA_OUT_DIR", "/tmp/auqa_output")
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# AWS / Textract config
region = os.environ.get("AWS_REGION", "us-east-1")
bucket_name = os.environ.get("S3_BUCKET", "")   # set in .env or Streamlit UI
textract = boto3.client("textract", region_name=region)
os_domain = os.environ.get("OS_DOMAIN", "opensearch-domain")
index_name = os.environ.get("INDEX_NAME", "test-auqa")

session = boto3.Session(region_name=region)
credentials = session.get_credentials().get_frozen_credentials()
awsauth = AWS4Auth(
    credentials.access_key, credentials.secret_key,
    region, "es", session_token=credentials.token
)
client = OpenSearch(hosts=[{"host": os_domain, "port": 443}],
                    http_auth=awsauth,
                    use_ssl=True, verify_certs=True,
                    connection_class=RequestsHttpConnection)

s3 = boto3.client("s3", region_name=region)
textract = boto3.client("textract", region_name=region)
bedrock = session.client("bedrock-runtime", region_name=region)
# ---------------- HELPER FUNCTIONS ----------------------------------------------
def clean_json_output(text: str) -> str:
    """Strip fence markers like ```json ... ``` or ``` ... ``` from model outputs."""
    if not isinstance(text, str):
        return text
    t = text.strip()
    if t.startswith("```json"):
        t = t[len("```json"):]
    if t.startswith("```"):
        t = t[3:]
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


def truncate_to_limit(text: str, max_tokens: int, buffer: int = 2500):
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        if len(tokens) > (max_tokens - buffer):
            return enc.decode(tokens[: max_tokens - buffer]), True
        return text, False
    except Exception:
        return text, False


# ---------------- Streamlit UI ----------------------------------------------
st.set_page_config(layout="wide")
st.title("📘 AUQA: Minimal Question Paper Generator")

# Basic metadata

title= "ANNA UNIVERSITY (UNIVERSITY DEPARTMENTS)"
stream=st.text_input("stream","B.E. /B. Tech / B. Arch (Full Time) ")
exam_title =st.text_input("Exam title","END SEMESTER EXAMINATIONS,")
exam_session = st.text_input("Exam Session (e.g. NOV/DEC 2025)", "NOV/DEC 2025")
course=st.text_input("Course","COMPUTER SCIENCE AND ENGINEERING")
semester = st.text_input("Semester", "VII / VIII")
subject_code = st.text_input("Subject Code", "CN")
subject_name = st.text_input("Subject Name", "Computer Networks")
department = st.text_input("Department", "Computer Technology")
regulation= st.text_input("Regulation","Regulation 2023")
date_val = st.date_input("Date")  # returns a datetime.date
# S3 PDF key (for Textract)
s3_key = st.text_input("S3 PDF Key (syllabus)", "syllabus/CN.pdf")
st.markdown("---")
# Models config
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "models.json"
if MODEL_FILE.exists():
    with open(MODEL_FILE, "r") as f:
        MODELS = json.load(f)
    model_names = [m["name"] for m in MODELS]
    selected_model = st.selectbox("Choose Model:", model_names)
    model_config = next(m for m in MODELS if m["name"] == selected_model)
else:
    st.warning("models.json not found. LLM invocation will be disabled.")
    MODELS, model_config = [], None
# Initialize session state holders
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""
if "units_parsed" not in st.session_state:
    st.session_state.units_parsed = []
if "COs" not in st.session_state:
    st.session_state.COs = []
if "qn_matrix" not in st.session_state:
    st.session_state.qn_matrix = pd.DataFrame()

# ---------------- Step 1: Ingest syllabus -----------------------------------
if st.button("Ingest syllabus PDF from S3 and parse units & COs"):
    st.info(f"Starting Textract job for s3://{bucket_name}/{s3_key}")
    try:
        resp = textract.start_document_text_detection(
            DocumentLocation={"S3Object": {"Bucket": bucket_name, "Name": s3_key}}
        )
        job_id = resp["JobId"]

        # Wait until job finishes
        while True:
            status_resp = textract.get_document_text_detection(JobId=job_id)
            status = status_resp.get("JobStatus")
            if status in ("SUCCEEDED", "FAILED"):
                break
            time.sleep(3)

        if status == "FAILED":
            st.error("Textract job failed")
        else:
            # Collect all pages into text
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
            full_text = "\n\n".join("\n".join(page_texts[p]) for p in sorted(page_texts.keys()))

            # Prompt template for parsing
            parse_template = """
INSTRUCTION:
You are an assistant specialized in extracting a course outline.

INPUT:
- Course ID: {SUBJECT_CODE}
- Document: extracted text below.

TASK:
Return ONLY a JSON object:
{
  "course_id": "{SUBJECT_CODE}",
  "course_objectives": [
     { "id": "CO1", "description": "..." }
  ],
  "units": [
    { "unit_no": <int>, "unit_name": "<string>", "topics": ["<string>", ...] }
  ]
}

RULES:
- Output JSON only (no commentary).
- Integers for unit_no.
- If missing, output empty lists.
- Ensure valid JSON.

---DOCUMENT-BEGIN---
{FULL_TEXT}
---DOCUMENT-END---
"""
            parse_prompt = parse_template.replace("{SUBJECT_CODE}", subject_code)\
                                         .replace("{FULL_TEXT}", full_text)

            if model_config:
                body = build_request(model_config["id"], selected_model, parse_prompt, 4096)
                resp = bedrock.invoke_model(modelId=model_config["id"], body=json.dumps(body))
                model_response = json.loads(resp["body"].read())
                generated_text = (model_response.get("outputs", [{}])[0].get("text") or
                                  model_response.get("content", [{}])[0].get("text") or
                                  str(model_response))
                generated_text = clean_json_output(generated_text)

                try:
                    parsed = json.loads(generated_text)
                    st.session_state.units_parsed = parsed.get("units", [])
                    st.session_state.COs = parsed.get("course_objectives", [])

                    # Display neatly
                    st.success("Parsed syllabus successfully")

                    st.subheader("📌 Course Outcomes (COs)")
                    for co in st.session_state.COs:
                        st.markdown(f"- **{co.get('id','')}**: {co.get('description','')}")

                    st.subheader("📘 Units")
                    for unit in st.session_state.units_parsed:
                        st.markdown(f"**Unit {unit.get('unit_no','?')}: {unit.get('unit_name','')}**")
                        #topics = unit.get("topics", [])
                        #if topics:
                        #    for t in topics:
                        #        st.markdown(f"  - {t}")
                        #else:
                        #    st.markdown("  _(No topics found)_")

                except Exception as e:
                    st.error(f"Failed to parse JSON: {e}")
                    st.text_area("Model raw output", generated_text, height=300)
            else:
                st.warning("Model config missing; saving raw text only.")
                st.session_state.raw_text = full_text
    except Exception as e:
        st.error(f"Textract error: {e}")

# ---------------- AUQA — Step 1.5: Interactive Mapping ----------------
if "units_parsed" not in st.session_state or st.session_state.units_parsed==[]:
    st.error("No Units found — Ingest Syllabus first")
    st.stop()
st.markdown("## AUQA: Question Mapping (Configurable)")

# --- Helpers / defaults ---------------------------------------------------
co_options = [c.get("id", f"CO{i+1}") for i, c in enumerate(st.session_state.get("COs", []))]
if not co_options:
    co_options = ["CO1", "CO2", "CO3", "CO4", "CO5"]
bl_options = ["L1", "L2", "L3", "L4", "L5", "L6"]

# Units available: prefer parsed units from syllabus, otherwise default 1..5
parsed_units = st.session_state.get("units_parsed", [])
if parsed_units:
    unit_labels = [str(u.get("unit_no", i+1)) + (": " + u.get("unit_name", "") if u.get("unit_name") else "") for i, u in enumerate(parsed_units)]
    unit_values = [str(u.get("unit_no", i+1)) for i, u in enumerate(parsed_units)]
else:
    unit_labels = [f"Unit {i}" for i in range(1, 6)]
    unit_values = [str(i) for i in range(1, 6)]

# --- Step 0: Select units to include & marks pattern ----------------------
st.markdown("### Select Units & Exam Pattern")
col0, col1 = st.columns([2, 1])
with col0:
    selected_units = st.multiselect("Select units to include (questions will be evenly distributed across these units)", options=unit_values, format_func=lambda x: next((lbl for lbl in unit_labels if lbl.startswith(str(x)+":")) , f"Unit {x}"), default=unit_values)
    if not selected_units:
        st.info("No units selected — defaulting to all units.")
        selected_units = unit_values.copy()
with col1:
    marks_pattern = st.selectbox("Total exam marks", options=[100, 50], index=0)
    

# --- Part-wise inputs ------------------------------------------------------
st.markdown("### Configure Part-wise Questions and Marks")
colA, colB, colC = st.columns(3)
with colA:
    partA_q = st.number_input("Part A - No. of questions", min_value=0, max_value=50, value=10 if marks_pattern==100 else 5, step=1)
    partA_m = st.number_input("Marks per Part A question", min_value=1, max_value=20, value=2, step=1)
with colB:
    if marks_pattern == 100:
        default_bq, default_bm = 5, 13
    else:
        default_bq, default_bm = 2, 16
    partB_q = st.number_input("Part B - No. of questions", min_value=0, max_value=50, value=default_bq, step=1)
    partB_m = st.number_input("Marks per Part B question", min_value=1, max_value=30, value=default_bm, step=1)
with colC:
    if marks_pattern == 100:
        default_cq, default_cm = 1, 15
    else:
        default_cq, default_cm = 1, 8
    partC_q = st.number_input("Part C - No. of questions", min_value=0, max_value=10, value=default_cq, step=1)
    partC_m = st.number_input("Marks per Part C question", min_value=1, max_value=30, value=default_cm, step=1)

# compute implied total marks
implied_total = partA_q * partA_m + partB_q * partB_m + partC_q * partC_m
st.markdown(f"**Implied total marks using the current values:** {implied_total}")
if implied_total != marks_pattern:
    st.warning("Implied total does not match selected total marks. Edit counts/marks to match or proceed intentionally.")

# --- When user confirms, build the question matrix ------------------------
if st.button("Generate question mapping matrix"):
    rows = []
    qno = 1
    # Part A
    for i in range(partA_q):
        rows.append({"QNo": qno, "Section": "Part A", "Marks": partA_m})
        qno += 1
    # Part B
    for i in range(partB_q):
        rows.append({"QNo": qno, "Section": "Part B", "Marks": partB_m})
        qno += 1
    # Part C
    for i in range(partC_q):
        rows.append({"QNo": qno, "Section": "Part C", "Marks": partC_m})
        qno += 1

    total_rows = len(rows)
    if total_rows == 0:
        st.error("No questions generated — set counts for parts.")
    else:
        df = pd.DataFrame(rows)

        # distribute Units, COs, BLs round-robin but restart for each Part
        unit_col, co_col, bl_col = [], [], []
        for section in ["Part A", "Part B", "Part C"]:
            group = df[df["Section"] == section]
            size = len(group)
            if size == 0:
                continue
            # ensure cycle matches group size
            unit_cycle = (selected_units * ((size // len(selected_units)) + 1))[:size]
            co_cycle = (co_options * ((size // len(co_options)) + 1))[:size]
            if section == "Part A":
                # Restrict BLs to only L1–L3 for Part A
                partA_bl = bl_options[:3]
                bl_cycle = (partA_bl * ((size // len(partA_bl)) + 1))[:size]
            elif section == "Part B":
                # For Part B, start BL from L3 (sequence L3–L6)
                shifted_bl = bl_options[2:]  # L3, L4, L5, L6
                bl_cycle = (shifted_bl * ((size // len(shifted_bl)) + 1))[:size]
                shifted_co = co_options[:] 
                co_cycle = (shifted_co * ((size // len(shifted_co)) + 1))[:size]
            else:
                # For Part C, reverse BL order from L5 to L3 and CO from CO3
                reversed_bl = ["L5", "L4", "L3"]
                bl_cycle = (reversed_bl * ((size // len(reversed_bl)) + 1))[:size]
                shifted_co = co_options[3:] + co_options[:3]
                co_cycle = (shifted_co * ((size // len(shifted_co)) + 1))[:size]
            unit_col.extend(unit_cycle)
            co_col.extend(co_cycle)
            bl_col.extend(bl_cycle)
        df["Unit"] = unit_col
        df["CO"] = co_col
        df["BL"] = bl_col

        st.session_state.qn_matrix = df
        st.success(f"Generated mapping matrix with {total_rows} rows (questions). You can now edit CO/BL/Unit below.")

# --- If matrix exists, show editable table and summaries ------------------
if "qn_matrix" in st.session_state and not st.session_state.qn_matrix.empty:
    st.markdown("## Edit Question → CO / BL / Unit mapping")
    df = st.session_state.qn_matrix.copy()
    df["Unit"] = df["Unit"].astype(str)

    edited = st.data_editor(
        df,
        num_rows="fixed",
        column_config={
            "QNo": st.column_config.NumberColumn("QNo", disabled=True),
            "Section": st.column_config.TextColumn("Section", disabled=True),
            "Marks": st.column_config.NumberColumn("Marks", disabled=True),
            "CO": st.column_config.SelectboxColumn("CO", options=co_options),
            "BL": st.column_config.SelectboxColumn("BL", options=bl_options),
            "Unit": st.column_config.SelectboxColumn("Unit", options=selected_units if selected_units else unit_values),
        },
        use_container_width=True,
    )
    st.session_state.qn_matrix = edited.copy()
    if(1==2):
        # Summary & graphs
        st.markdown("### Summary & Distribution")
        total_marks = int(edited["Marks"].sum())
        st.write(f"Grand total = {total_marks} marks (target pattern = {marks_pattern})")

        co_marks = edited.groupby("CO")["Marks"].sum().reindex(co_options, fill_value=0)
        st.write("**CO-wise Marks Distribution**")
        st.bar_chart(co_marks)

        bl_marks = edited.groupby("BL")["Marks"].sum().reindex(bl_options, fill_value=0)
        st.write("**Bloom’s Level Marks Distribution**")
        st.bar_chart(bl_marks)

        unit_marks = edited.groupby("Unit")["Marks"].sum().reindex(selected_units if selected_units else unit_values, fill_value=0)
        st.write("**Unit-wise Marks Distribution**")
        st.bar_chart(unit_marks)
    total_marks = int(edited["Marks"].sum())

    co_marks = edited.groupby("CO")["Marks"].sum().reindex(co_options, fill_value=0)
    st.write("**CO % of Total**")
    percent_by_co = (co_marks / total_marks * 100).round(1)
    st.table(percent_by_co.to_frame("Percentage"))

    bl_marks = edited.groupby("BL")["Marks"].sum().reindex(bl_options, fill_value=0)
    st.write("**BL % of Total**")
    percent_by_bl = (bl_marks / total_marks * 100).round(1)
    st.table(percent_by_bl.to_frame("Percentage"))

    unit_marks = edited.groupby("Unit")["Marks"].sum().reindex(selected_units if selected_units else unit_values, fill_value=0)
    st.write("**Unit % of Total**")
    percent_by_unit = (unit_marks / total_marks * 100).round(1)
    st.table(percent_by_unit.to_frame("Percentage"))

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download Mapping (CSV)", edited.to_csv(index=False), file_name="qn_mapping.csv", mime="text/csv")
    with col2:
        st.download_button("Download Mapping (JSON)", edited.to_json(orient="records", indent=2), file_name="qn_mapping.json", mime="application/json")
    

# assume these exist from your environment
# from bedrockModels import build_request, count_tokens
# client, index_name, bedrock, model_config, selected_model, subject_code, subject_name

# ---------------- Prompt template (single source of truth) -----------------
PROMPT_TEMPLATE = """INSTRUCTION:
You are an experienced university exam question-writer. Use ONLY the provided Context for factual support. 
Return ONLY a JSON array (no commentary). Each item must exactly follow this schema:

{{
  "QNo": <int>,
  "Section": "Part A" | "Part B" | "Part C",
  "Marks": <int>,
  "Unit": "{unit_no}",
  "CO": "COx",
  "BL": "Ly",
  "SUB": "a" | "b" | null,
  "Qn": "<Question text (concise)>",
  "Content": "<Short model answer/key points (1-3 sentences)>",
  "Page": "page range from Context>"
}}


INSTRUCTIONS:
1.	Map to Course Outcome (CO) and Bloom's Level (BL): Each generated question must strictly adhere to the specified CO and BL.
	* CO: Ensure the question effectively tests the knowledge or skill described in the given Course Outcome.
	* BL: The question's cognitive demand must match the specified Bloom's Level. Use the following as a guide:
        * 	L1 (Remember): Ask the user to recall facts, definitions, basic concepts, or answer direct "what," "who," or "when" questions based on the text.
        *	L2 (Understand): Ask for an explanation of concepts or ideas. The question should require the user to summarize, classify, or describe the "how" or "why" of a topic in their own words, demonstrating comprehension.
        *	L3 (Apply): Pose a problem where the user must apply a known concept, formula, or procedure to a new but similar situation to find a definitive solution.
        *	L4 (Analyze): Require the user to break down information into its constituent parts to examine relationships. This could involve comparing/contrasting elements, differentiating between ideas, or interpreting data to draw a specific conclusion.
        *	L5 (Evaluate): Ask the user to make a judgment or form an opinion based on specific criteria. The question should require justifying a decision, critiquing a statement, or arguing for a particular standpoint.
        *	L6 (Create): Challenge the user to synthesize information to generate a new product, plan, or point of view. This could involve designing a solution, formulating a hypothesis, or developing a novel approach.
2.	Question and Content Length:
    *	Part A: Questions should be concise and direct.
    *	Part B & Part C: Questions can be more detailed and multi-faceted to adequately assess higher-order thinking. The length should be appropriate for the complexity of the task.
    *	Content: The supporting content field should remain brief (1–3 sentences).
3.	Question Structure for Part B and Part C:
    *   For each Part B row: produce two subdivisions with SUB = "a" and SUB = "b". Both subdivisions carry the full Marks value of the original row (i.e., each is worth the same full marks as the parent). The two subdivisions should be complementary in format or cognitive demand and each must be independently answerable to full-mark standards.
    *	Part C: For each row in the mapping, you must produce a single question object containing two subdivisions (SUB = "a" and SUB = "b"). These subdivisions should be complementary (e.g., a calculation followed by an interpretation). The total marks for the row must be split between the two subdivisions. For odd-numbered marks, split them with the larger portion first (e.g., 13 marks become 7 for part 'a' and 6 for part 'b').
4.	Context and Citations: Use the provided context chunks to source factual details for the questions. For each question, you must cite the Page number from the first context chunk that contains the necessary information. If a question is purely conceptual or applicative, set the Page to the document ID of the most relevant context chunk.
5.	Factual Integrity: Do NOT invent facts or details not present in the provided context. If the context lacks the specificity for a factual recall question (L1), create a conceptual (L2) or applicative (L3+) question instead.
6.	Inclusion of Contextual Questions: With a probability of approximately 50%, you should incorporate a relevant "book-back" style question found verbatim within the provided context. The chosen question must match the assigned CO and BL. Append an asterisk (*) to the end of any such question.



--- INPUT (placeholders filled at runtime) ---
Course: {course_id} - {subject_name}
Unit: {unit_no}
Unit topics: {unit_topics}

Part summary (paper-level): 
{part_summary}

Questions mapping for this unit (list — each entry includes CO and BL):
{questions_in_unit}

Context (top retrieved chunks; format [DOC_ID:... | PAGES:...] text):
{context}

Please output the JSON array now.
"""

# ---------------- Helper utilities (reuse your earlier helpers) -------------

def expand_subdivisions(rows: List[dict]) -> List[dict]:
    """
    Expand Part B/C rows into subdivision rows (a/b). Return the expanded list.
    Each expanded row keeps original QNo, Section, Unit, CO, BL but has SUB and Marks set.
    """
    out = []
    for r in rows:
        section = r.get("Section", "")
        marks = int(r.get("Marks", 0))

        if section == "Part B":
            r_a = r.copy(); r_a["SUB"] = "a"; r_a["Marks"] = marks
            r_b = r.copy(); r_b["SUB"] = "b"; r_b["Marks"] = marks
            out.append(r_a); out.append(r_b)

        elif section == "Part C":
            a = math.ceil(marks / 2)
            b = marks - a
            r_a = r.copy(); r_a["SUB"] = "a"; r_a["Marks"] = a
            r_b = r.copy(); r_b["SUB"] = "b"; r_b["Marks"] = b
            out.append(r_a); out.append(r_b)

        else:
            nr = r.copy(); nr["SUB"] = None
            out.append(nr)
    return out

# reuse your BM25/vector/merge helpers defined earlier (bm25_search, vector_search, merge_and_dedupe)
# (Assume these are available in the same module as earlier code.)

# ---------------- Main generation step (Streamlit button) ------------------
# Helper: ensure mapping exists
if "qn_matrix" not in st.session_state or st.session_state.qn_matrix.empty:
    st.error("No mapping found in session_state.qn_matrix — Generate Qn Matrix First")
    st.stop()
st.header("🔎 Step 2: Unit-wise Retrieval & Question Generation (with CO/BL detail)")

num_hits = st.number_input("Top-K docs to retrieve (per method)", min_value=2, max_value=50, value=15, step=1)
merge_top_k = st.number_input("Final unique chunks to keep per unit", min_value=1, max_value=50, value=10, step=1)
ratio = st.slider("BM25 vs Vector weighting (fraction BM25)", 0.0, 1.0, 0.5, 0.1)
max_gen_len = st.number_input("Max generation tokens", min_value=128, max_value=4096, value=2048, step=10)
temperature = float(st.number_input("Temperature (0.0 deterministic)", min_value=0.0, max_value=1.0, value=0.5, step=0.1))


# Helper: ensure mapping exists
if "qn_matrix" not in st.session_state or st.session_state.qn_matrix.empty:
    st.error("No mapping found in session_state.qn_matrix — run Step 1.5 first.")
    st.stop()

qn_df: pd.DataFrame = st.session_state.qn_matrix.copy()

def sub_sort_key(sub):
    if sub is None:
        return 0
    if str(sub).lower() == "a":
        return 1
    if str(sub).lower() == "b":
        return 2
    return 3

def bm25_search(query: str, k: int, course_id: str = None) -> List[Tuple[str, str, str]]:
    """Return list of (doc_id, chunk_text, page_range) from BM25 open search."""
    if course_id:
        bm25_q = {
            "size": k,
            "query": {
                "bool": {
                    "filter": [{"term": {"course_id": course_id}}],
                    "must": {"match": {"chunk_text": query}}
                }
            }
        }
    else:
        bm25_q = {"size": k, "query": {"match": {"chunk_text": query}}}
    resp = client.search(index=index_name, body=bm25_q)
    hits = resp.get("hits", {}).get("hits", [])
    return [(h["_id"], h["_source"]["chunk_text"], h["_source"].get("page_range", "?")) for h in hits]

def vector_search(query: str, k: int, course_id: str = None) -> List[Tuple[str, str, str]]:
    """Return list of (doc_id, chunk_text, page_range) using embedding + kNN."""
    emb_body = json.dumps({"inputText": query})
    emb_resp = bedrock.invoke_model(modelId="amazon.titan-embed-text-v2:0", body=emb_body)
    emb = json.loads(emb_resp["body"].read())["embedding"]
    if course_id:
        vector_q = {
            "size": k,
            "query": {
                "bool": {
                    "filter": [{"term": {"course_id": course_id}}],
                    "must": {"knn": {"vector_field": {"vector": emb, "k": k}}}
                }
            }
        }
    else:
        vector_q = {"size": k, "query": {"knn": {"vector_field": {"vector": emb, "k": k}}}}
    vresp = client.search(index=index_name, body=vector_q)
    hits = vresp.get("hits", {}).get("hits", [])
    return [(h["_id"], h["_source"]["chunk_text"], h["_source"].get("page_range", "?")) for h in hits]

def merge_and_dedupe(bm25_docs, vector_docs, take_top: int, bm25_take: int):
    """Weighted merge, dedupe by doc id, and return up to take_top chunks."""
    bm25_part = bm25_docs[:bm25_take]
    vec_part = vector_docs[: max(0, take_top - bm25_take)]
    combined = bm25_part + vec_part
    seen = set()
    final = []
    for did, txt, pr in combined:
        if did not in seen:
            final.append((did, txt, pr))
            seen.add(did)
            if len(final) >= take_top:
                break
    return final

if st.button("GENERATE QUESTION PAPER"):
    client = OpenSearch(hosts=[{"host": os_domain, "port": 443}],
                    http_auth=awsauth,
                    use_ssl=True, verify_certs=True,
                    connection_class=RequestsHttpConnection)
    if "qn_matrix" not in st.session_state or st.session_state.qn_matrix.empty:
        st.error("No mapping found in st.session_state.qn_matrix — run Step 1.5 first.")
        st.stop()

    qn_df = st.session_state.qn_matrix.copy()
    generated_items = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    # Build paper-level part_summary with per-question CO & BL details
    def build_part_summary(df: pd.DataFrame):
        ps = {}
        for part in ("PartA", "PartB", "PartC"):
            section = "Part A" if part=="PartA" else ("Part B" if part=="PartB" else "Part C")
            rows = df[df["Section"] == section].to_dict(orient="records")
            # For each question we include QNo, Marks, CO, BL
            ps[part] = {
                "num_questions": len(rows),
                "marks_each": rows[0]["Marks"] if rows else 0,
                "questions": [{"QNo": r["QNo"], "Marks": r["Marks"], "CO": r["CO"], "BL": r["BL"]} for r in rows]
            }
            if part in ("PartB","PartC"):
                ps[part]["subdivisions"] = True
        return ps

    part_summary = build_part_summary(qn_df)

    # iterate units deterministically
    for unit in sorted(qn_df["Unit"].unique(), key=lambda x: int(x)):
        st.write(f"---\n### Running Process on Unit {unit}")
        # collect topics (if present)
        unit_topics = []
        for u in st.session_state.get("units_parsed", []):
            if str(u.get("unit_no")) == str(unit):
                unit_topics = u.get("topics", []) or []
                break

        # Build query from topics
        query = "; ".join(unit_topics) if unit_topics else f"Unit {unit}"

        # --- Retrieval with optional course_id filter applied to both searches
        bm25_docs = bm25_search(query, num_hits, course_id=subject_code or None)
        vector_docs = vector_search(query, num_hits, course_id=subject_code or None)

        bm25_take = int(num_hits * ratio)
        final_chunks = merge_and_dedupe(bm25_docs, vector_docs, take_top=merge_top_k, bm25_take=bm25_take)

        # Show retrieved chunks (optional)
        #   st.dataframe(pd.DataFrame(final_chunks, columns=["Doc ID", "Chunk Text", "Page Range"]))
        #else:
        #    st.info("No chunks retrieved; prompt will use unit topics only.")

        # Build context string (concatenate top chunks)
        context_entries = [f"[DOC_ID:{did} | PAGES:{pr}] {txt}" for did, txt, pr in final_chunks]
        context_text = "\n\n".join(context_entries)

        # Build mapping rows for this unit and expand subdivisions
        rows_for_unit = qn_df[qn_df["Unit"] == str(unit)].to_dict(orient="records")
        mapping_expanded = expand_subdivisions(rows_for_unit)

        # Build the per-unit prompt by filling placeholders
        prompt_filled = PROMPT_TEMPLATE.format(
            course_id=(subject_code),
            subject_name=subject_name,
            unit_no=unit,
            unit_topics=json.dumps(unit_topics),
            part_summary=json.dumps(part_summary, indent=2),
            questions_in_unit=json.dumps(mapping_expanded, indent=2),
            context=context_text
        )

        # --- Call model using your provided logic (amazon.nova vs invoke_model)
        model_id = model_config["id"]
        #st.write(f"Calling model: {model_id} ...")
        if model_id.startswith("amazon.nova"):
            conversation = [{"role": "user", "content": [{"text": prompt_filled}]}]
            resp = bedrock.converse(modelId=model_id, messages=conversation,
                                     inferenceConfig={"maxTokens": max_gen_len, "temperature": temperature, "topP": 0.9})
            generated_text = resp["output"]["message"]["content"][0]["text"]
            usage = resp.get("usage", {})
        else:
            body = build_request(model_id, selected_model, prompt_filled, max_gen_len)
            resp = bedrock.invoke_model(modelId=model_id, body=json.dumps(body))
            model_response = json.loads(resp["body"].read())
            usage = model_response.get("usage", {})

            # model-specific extraction
            if "anthropic" in model_id or "claude" in selected_model.lower():
                generated_text = model_response["content"][0]["text"]
            elif "llama" in model_id:
                generated_text = model_response.get("generation") or str(model_response)
            elif "mistral" in model_id:
                generated_text = model_response.get("outputs", [{"text": str(model_response)}])[0].get("text")
            else:
                # generic fallback
                generated_text = str(model_response)

        generated_text = clean_json_output(generated_text)

        # --- Token usage & cost calculation (prefer reported usage)
        if usage:
            in_tokens = usage.get("inputTokens") or usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            out_tokens = usage.get("outputTokens") or usage.get("output_tokens") or usage.get("completion_tokens") or 0
        else:
            in_tokens = count_tokens(prompt_filled)
            out_tokens = count_tokens(generated_text)
        total_input_tokens += in_tokens if 'total_input_tokens' in locals() else in_tokens
        total_output_tokens += out_tokens if 'total_output_tokens' in locals() else out_tokens

        price_in = model_config.get("price_input", 0.0)
        price_out = model_config.get("price_output", 0.0)
        unit_cost = (in_tokens / 1000.0) * price_in + (out_tokens / 1000.0) * price_out
        total_cost += unit_cost

        # --- Parse & validate JSON output
        try:
            parsed = json.loads(generated_text)
            # normalize to list
            if isinstance(parsed, dict):
                parsed = [parsed]
            if not isinstance(parsed, list):
                raise ValueError("Model output JSON is not a list")

            valid = []
            for item in parsed:
                # basic normalization and checking
                if not isinstance(item, dict):
                    continue
                # ensure required keys
                required = ["QNo","Section","Marks","Unit","CO","BL","SUB","Qn","Content","Page"]
                for k in required:
                    item.setdefault(k, None)
                # check CO & BL adherence: if item contains CO/BL, compare to mapping
                # (basic check: item.CO and item.BL must match one of the mapping rows for same QNo)
                qno_matches = [r for r in mapping_expanded if r["QNo"] == item["QNo"]]
                if qno_matches:
                    expected = qno_matches[0]
                    if str(item.get("CO")) != str(expected.get("CO")) or str(item.get("BL")) != str(expected.get("BL")):
                        st.warning(f"Unit {unit} QNo {item.get('QNo')}: Model CO/BL ({item.get('CO')}/{item.get('BL')}) differs from mapping ({expected.get('CO')}/{expected.get('BL')}).")
                valid.append(item)
            generated_items.extend(valid)
            generated_items = sorted(
                generated_items,
                key=lambda x: (int(x.get("QNo", 0)), sub_sort_key(x.get("SUB")))
            )
            st.success(f"Unit {unit}: parsed {len(valid)} items (cost ${unit_cost:.4f})")
            #st.json(valid)
        except Exception as e:
            st.error(f"Failed to parse model JSON for Unit {unit}: {e}")
            #st.text_area("Raw output for debugging", generated_text, height=300)
   
    # Save & show totals
    st.session_state.generated_qns = generated_items
    st.write("## Generation summary")
    st.write(f"- Units processed: {len(sorted(qn_df['Unit'].unique()))}")
    st.write(f"- Total generated items: {len(generated_items)}")
    st.write(f"- Input tokens (approx): {total_input_tokens}, Output tokens (approx): {total_output_tokens}")
    st.write(f"- Estimated cost: ${total_cost:.6f}")

    # --- Download outputs ---------------------------------------------------
if "generated_qns" in st.session_state and st.session_state.generated_qns:
    df_out = pd.DataFrame(st.session_state.generated_qns)

    st.markdown("## 📥 Download Generated Questions")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download as JSON",
            json.dumps(st.session_state.generated_qns, indent=2),
            file_name="generated_questions.json",
            mime="application/json"
        )
    with col2:
        st.download_button(
            "Download as CSV",
            df_out.to_csv(index=False),
            file_name="generated_questions.csv",
            mime="text/csv"
        )

   # Optional: show preview table with filtering
    st.markdown("### Preview (all generated questions)")

    df_out = pd.DataFrame(st.session_state.generated_qns)

    # --- Filter/search bar ---
    search_term = st.text_input("🔍 Search (by QNo, Unit, CO, BL, etc.)", "")
    if search_term:
        mask = df_out.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
        filtered_df = df_out[mask]
    else:
        filtered_df = df_out

    st.dataframe(filtered_df, use_container_width=True, height=600)
    st.write(f"Showing {len(filtered_df)} of {len(df_out)} total questions")


    session_COs = st.session_state.get("COs", [])               # expecting list of {"id":"CO1","description":"..."}
    session_qns = st.session_state.get("generated_qns", [])     # expecting list of question dicts
    session_units = st.session_state.get("units_parsed", [])           # expecting list of units in your format

    # ---------- Helper transformations ----------
    # 1) marks -> time mapping
    if marks_pattern == 100:
        time_str = "3 Hours"
        max_marks = 100
    else:
        time_str = "90 Minutes"
        max_marks = 50

    # 2) format date as DD/MM/YYYY
    if isinstance(date_val, (datetime, )):
        date_str = date_val.strftime("%d/%m/%Y")
    else:
        # date_val is probably a datetime.date
        date_str = date_val.strftime("%d/%m/%Y")

    # 3) convert course objectives into {"code","text"} if needed
    course_outcomes = []
    for co in session_COs:
        # support both {"id","description"} and {"code","text"} formats gracefully
        if isinstance(co, dict):
            code = co.get("code") or co.get("id") or co.get("CO") or co.get("co") 
            text = co.get("text") or co.get("description") or co.get("desc") or co.get("name")
            if code and text:
                course_outcomes.append({"code": code, "text": text})
            elif code:  # if text missing, keep empty string
                course_outcomes.append({"code": code, "text": co.get("description","")})
            else:
                # fallback: try to convert small dict into a string
                course_outcomes.append({"code": str(co.get("id","COx")), "text": str(co.get("description", str(co)))})
        else:
            # not a dict — skip or convert
            continue

    # 4) standardize questions list: ensure expected keys exist (QNo, Section, Marks, Unit, CO, BL, SUB, Qn, Content, Page)
    standard_qns = []
    for q in session_qns:
        if not isinstance(q, dict):
            continue
        # create a minimal normalized dict
        norm = {
            "QNo": q.get("QNo"),
            "Section": q.get("Section"),
            "Marks": q.get("Marks"),
            "Unit": q.get("Unit"),
            "CO": q.get("CO"),
            "BL": q.get("BL"),
            "SUB": q.get("SUB") if "SUB" in q else q.get("sub") if "sub" in q else None,
            "Qn": q.get("Qn") or q.get("question") or q.get("q"),
            "Content": q.get("Content") or q.get("Answer") or "",
            "Page": q.get("Page") or q.get("page") or ""
        }
        standard_qns.append(norm)

    # 5) units (if present) — we assume session_units are already in the required structure; otherwise normalize minimal fields
    normalized_units = []
    for u in session_units:
        if not isinstance(u, dict):
            continue
        normalized_units.append({
            "unit_no": u.get("unit_no") or u.get("unit") or u.get("no"),
            "unit_name": u.get("unit_name") or u.get("name") or u.get("title"),
            "topics": u.get("topics") or u.get("topic_list") or u.get("topics_list") or []
        })
    df1 = st.session_state.qn_matrix.copy()  # DataFrame

        # sanitize: replace NaN / None with empty string to avoid "None" showing
    df1 = df1.fillna("")

        # convert to list of plain dicts (JSON-serialisable)
    qn_records = df1.to_dict(orient="records")

    part_summary = (
    df1.groupby("Section")["Marks"]
      .agg(["count", "sum"])
      .rename(columns={"count": "qn_count", "sum": "total_marks"})
      .to_dict(orient="index")
    )

    # ---------- Build final dict ----------
    exam_dict = {
        "title": title,
        "stream": stream,
        "exam_title": exam_title,
        "exam_session": exam_session,
        "course": course,
        "semester": semester,
        "subject_code": subject_code,
        "subject_name": subject_name,
        "department": department,
        "regulation": regulation,
        "date": date_str,
        "time": time_str,
        "max_marks": max_marks,
        "course_outcomes": course_outcomes,
        "units": normalized_units,
        "questions": standard_qns,
        "qn_matrix": qn_records,
        "part_summary":part_summary
    }

    # ---------- UI: show and export ----------
    #st.subheader("Generated exam JSON")
    #st.json(exam_dict)

    # Download button
    if(1==2):
        exam_json_str = json.dumps(exam_dict, indent=2, ensure_ascii=False)
        st.download_button(
            label="Download exam JSON",
            data=exam_json_str,
            file_name=f"{subject_code}_exam.json",
            mime="application/json"
        )

    # Also put it into session state if you want to reuse
    st.session_state["exported_exam_dict"] = exam_dict
    # Either use parsed/generated data OR fallback to default_data
    qp_data = st.session_state.get("exported_exam_dict")

    # Render the PDF UI (uses Playwright internally now)
    render_qp_pdf(qp_data, template_name="template2.html", title="CS23303 — Generate Question Paper PDF")
