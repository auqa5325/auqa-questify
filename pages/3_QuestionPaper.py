import os
import time
import json
import pathlib
import re
import hashlib
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from bedrockModels import build_request, count_tokens
import boto3
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from qp_pdf_generator import render_qp_pdf
import math
from typing import Any, Dict, List, Optional, Tuple
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
client = OpenSearch(
    hosts=[{"host": os_domain, "port": 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=60,            # request timeout
    max_retries=3,         # retry attempts
    retry_on_timeout=True  # retry if timed out
)


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
stream=st.text_input("stream","B.E. /B. Tech ")
exam_title =st.text_input("Exam title","END SEMESTER EXAMINATIONS,")
exam_session = st.text_input("Exam Session (e.g. NOV/DEC 2025)", "NOV/DEC 2025")
course=st.text_input("Course","COMPUTER SCIENCE AND ENGINEERING")
semester = st.text_input("Semester", "VII ")
subject_code = st.text_input("Subject Code", "CS23602")
subject_name = st.text_input("Subject Name", "Compiler ")
department = st.text_input("Department", "Computer Technology")
regulation= st.text_input("Regulation","Regulation 2023")
date_val = st.date_input("Date")  # returns a datetime.date
# S3 PDF key (for Textract)
s3_key = st.text_input("S3 PDF Key (syllabus)", "syllabus/CS23602.pdf")
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
    marks_pattern = st.selectbox("Total exam marks", options=[100, 50, 26], index=0)
    

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
Generate exam questions that give data/scenarios in the question itself (dont point back or refer tables or figures in the context) and ask students to deduce, compute, pseudocode or construct queries, solutions for numerical/algorithmic courses (arrays, trees, equations) 
and to analyze, design or justify solutions for design/theory courses (case studies, models, architecture).
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


Strictly follow these Instructions for Question Generation
1. Mapping to CO & Bloom's Level (BL)
   * CO: Ensure the question effectively tests the knowledge or skill described in the given Course Outcome.
   * BL: The question's cognitive demand must match the specified Bloom's Level. Use the following as a guide:
        * 	L1 (Remember): Ask the user to recall facts, definitions, basic concepts, or answer direct "what," "who," or "when" questions based on the text.
        *	L2 (Understand): Ask for an explanation of concepts or ideas. The question should require the user to summarize, classify, or describe the "how" or "why" of a topic in their own words, demonstrating comprehension.
        *	L3 (Apply): Pose a problem where the user must apply a known concept, formula, or procedure to a new but similar situation to find a definitive solution.
        *	L4 (Analyze): Require the user to break down information into its constituent parts to examine relationships. This could involve comparing/contrasting elements, differentiating between ideas, or interpreting data to draw a specific conclusion.
        *	L5 (Evaluate): Ask the user to make a judgment or form an opinion based on specific criteria. The question should require justifying a decision, critiquing a statement, or arguing for a particular standpoint.
        *	L6 (Create): Challenge the user to synthesize information to generate a new product, plan, or point of view. This could involve designing a solution, formulating a hypothesis, or developing a novel approach.
2. Length & Format
   * Write clear, academic-style questions similar to a university exam
   * Ensure balance of theory and problem-solving questions and coding questions as appropriate.
   * Part A: Concise, direct questions.
   * Part B/C: More detailed and multi-faceted.
3. Subdivisions
   * Part B: Each question splits into a and b, both carrying full marks and independently answerable.
   * Part C: One question with a and b, marks split (larger share first for odd totals, e.g., 13 → 7 & 6).
    Subdivisions must be complementary but independently answerable.
4. Context & Citations
   * Base all questions on the provided context.
   * Cite the Page number of the first chunk containing the required info.
5. Factual Integrity
   * Do not invent facts.
   * Questions should remain generic and must not explicitly reference the context text.
   * Include all data necessary for solving any problem within the question itself.
   * Fallback:If the context lacks details for L2, create an L1 or lower question instead.


--- INPUT (placeholders filled at runtime) ---
Course: {course_id} - {subject_name}
Unit: {unit_no}
Unit topics: {unit_topics}
Mapped chapters: {mapped_chapters}
Mapped sections: {mapped_sections}
Mapping confidence: {mapping_confidence}
Retrieval mode used: {retrieval_mode_used}

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
retrieval_mode_label = st.selectbox("Retrieval Method", ["Hybrid", "Embedding", "KG"], index=0)
retrieval_mode = retrieval_mode_label.strip().lower()
scope_candidates_k = st.number_input("Scope candidates (stage-1 lexical)", min_value=2, max_value=20, value=8, step=1)
scope_final_k = st.number_input("Scope keep (stage-2 rerank)", min_value=1, max_value=10, value=3, step=1)
max_gen_len = st.number_input("Max generation tokens", min_value=128, max_value=4096, value=2048, step=10)
temperature = float(st.number_input("Temperature (0.0 deterministic)", min_value=0.0, max_value=1.0, value=0.5, step=0.1))


# Helper: ensure mapping exists
if "qn_matrix" not in st.session_state or st.session_state.qn_matrix.empty:
    st.error("No mapping found in session_state.qn_matrix — run Step 1.5 first.")
    st.stop()

qn_df: pd.DataFrame = st.session_state.qn_matrix.copy()


def load_questindex_helpers() -> Dict[str, Any]:
    try:
        from kg.questindex_core import (
            get_books as quest_get_books,
            get_chapter_section_catalog,
            retrieve_kg_context,
            retrieve_kg_context_v2,
        )

        return {
            "ok": True,
            "get_books": quest_get_books,
            "get_catalog": get_chapter_section_catalog,
            "retrieve_kg_context": retrieve_kg_context,
            "retrieve_kg_context_v2": retrieve_kg_context_v2,
            "error": None,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


questindex_helpers = load_questindex_helpers()
quest_books: List[str] = []
if questindex_helpers.get("ok"):
    try:
        quest_books = questindex_helpers["get_books"]()
    except Exception as exc:
        questindex_helpers = {"ok": False, "error": str(exc)}

quest_book_id: Optional[str] = None
if questindex_helpers.get("ok"):
    quest_options = ["None"] + quest_books
    selected_quest_book = st.selectbox(
        "QuestIndex Book (for structure-aware retrieval)",
        quest_options,
        index=0,
    )
    if selected_quest_book != "None":
        quest_book_id = selected_quest_book
else:
    if retrieval_mode == "kg":
        st.warning(f"QuestIndex graph is not available. KG retrieval will run without fallback. ({questindex_helpers.get('error')})")

retrieval_index_name = st.text_input("OpenSearch Retrieval Index", index_name)
source_filter_label = st.selectbox("Chunk Source Filter", ["Any", "8_QuestIndex", "6_KG", "1_Ingestion"], index=0)
retrieval_source_filter = None if source_filter_label == "Any" else source_filter_label
default_retrieval_filter = quest_book_id if (retrieval_source_filter == "8_QuestIndex" and quest_book_id) else subject_code
retrieval_course_filter = st.text_input(
    "OpenSearch Course/Book Filter (optional)",
    default_retrieval_filter if default_retrieval_filter else "",
    help="For QuestIndex chunks, this should usually match QuestIndex Book ID.",
)


def sub_sort_key(sub):
    if sub is None:
        return 0
    if str(sub).lower() == "a":
        return 1
    if str(sub).lower() == "b":
        return 2
    return 3


def normalize_text(text: str) -> str:
    value = str(text or "").strip().lower()
    return re.sub(r"\s+", " ", value)


def make_chunk_hash(text: str) -> str:
    normalized = normalize_text(text)
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).hexdigest()


def parse_page_range(page_range: Any) -> Tuple[Optional[int], Optional[int]]:
    if page_range is None:
        return None, None
    match = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", str(page_range))
    if match:
        return int(match.group(1)), int(match.group(2))
    match_single = re.match(r"^\s*(\d+)\s*$", str(page_range))
    if match_single:
        page = int(match_single.group(1))
        return page, page
    return None, None


def tokens_for_matching(text: str) -> List[str]:
    terms = []
    for token in re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower()):
        if len(token) >= 3:
            terms.append(token)
    return terms


def query_terms_for_kg(query: str) -> List[str]:
    return sorted(set(tokens_for_matching(query)))


def lexical_overlap_score(query_text: str, candidate_text: str) -> int:
    query_terms = set(tokens_for_matching(query_text))
    candidate_terms = set(tokens_for_matching(candidate_text))
    if not query_terms or not candidate_terms:
        return 0
    return len(query_terms.intersection(candidate_terms))


def invoke_model_text(
    model_id: str,
    selected_model_name: str,
    prompt: str,
    max_tokens: int = 900,
    temp: float = 0.0,
) -> str:
    if model_id.startswith("amazon.nova"):
        conversation = [{"role": "user", "content": [{"text": prompt}]}]
        response = bedrock.converse(
            modelId=model_id,
            messages=conversation,
            inferenceConfig={"maxTokens": max_tokens, "temperature": temp, "topP": 0.9},
        )
        return response["output"]["message"]["content"][0]["text"]

    body = build_request(model_id, selected_model_name, prompt, max_tokens)
    response = bedrock.invoke_model(modelId=model_id, body=json.dumps(body))
    payload = json.loads(response["body"].read())

    if "anthropic" in model_id or "claude" in selected_model_name.lower():
        return payload.get("content", [{}])[0].get("text", "")
    if "llama" in model_id:
        return payload.get("generation", "")
    if "mistral" in model_id:
        return payload.get("outputs", [{"text": ""}])[0].get("text", "")
    return (
        payload.get("outputs", [{}])[0].get("text")
        or payload.get("content", [{}])[0].get("text")
        or str(payload)
    )


def invoke_model_json(
    model_id: str,
    selected_model_name: str,
    prompt: str,
    max_tokens: int = 900,
) -> Optional[Dict[str, Any]]:
    try:
        text = invoke_model_text(model_id, selected_model_name, prompt, max_tokens=max_tokens, temp=0.0)
        cleaned = clean_json_output(text)
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def unit_sort_key(value: Any) -> Tuple[int, str]:
    try:
        return int(str(value)), str(value)
    except Exception:
        return 9999, str(value)


def _to_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def hit_to_doc(hit: Dict[str, Any]) -> Dict[str, Any]:
    source = hit.get("_source", {})
    page_start = _to_int(source.get("page_start"))
    page_end = _to_int(source.get("page_end"))
    if page_start is None or page_end is None:
        pr_start, pr_end = parse_page_range(source.get("page_range"))
        page_start = page_start if page_start is not None else pr_start
        page_end = page_end if page_end is not None else pr_end

    chunk_text = str(source.get("chunk_text") or "")
    chunk_hash = str(source.get("chunk_hash") or make_chunk_hash(chunk_text))
    chunk_id = str(source.get("chunk_id") or hit.get("_id") or chunk_hash)

    return {
        "doc_id": str(hit.get("_id") or chunk_id),
        "chunk_id": chunk_id,
        "chunk_hash": chunk_hash,
        "chunk_text": chunk_text,
        "page_range": source.get("page_range", "?"),
        "page_start": page_start,
        "page_end": page_end,
        "chapter_no": source.get("chapter_no"),
        "section_no": source.get("section_no"),
        "source": source.get("source"),
    }


def build_retrieval_filters(course_id: Optional[str], source_filter: Optional[str]) -> List[Dict[str, Any]]:
    filters: List[Dict[str, Any]] = []
    if course_id:
        filters.append({"term": {"course_id": course_id}})
    if source_filter:
        filters.append({"term": {"source": source_filter}})
    return filters


def compute_index_health(
    course_id: Optional[str],
    sample_size: int = 250,
    index_override: Optional[str] = None,
    source_filter: Optional[str] = None,
) -> Dict[str, Any]:
    filters = build_retrieval_filters(course_id, source_filter)
    body = {
        "size": sample_size,
        "_source": ["chunk_id", "chunk_hash", "page_start", "page_end", "page_range", "course_id", "source"],
        "query": {"bool": {"filter": filters}} if filters else {"match_all": {}},
    }

    try:
        resp = client.search(index=index_override or index_name, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        sampled = len(hits)
        if sampled == 0:
            return {"sampled": 0, "chunk_id_pct": 0.0, "page_numeric_pct": 0.0}

        with_chunk_id = 0
        with_numeric_page = 0
        for hit in hits:
            src = hit.get("_source", {})
            if src.get("chunk_id"):
                with_chunk_id += 1
            page_start = _to_int(src.get("page_start"))
            page_end = _to_int(src.get("page_end"))
            if page_start is not None and page_end is not None:
                with_numeric_page += 1

        return {
            "sampled": sampled,
            "chunk_id_pct": round((with_chunk_id / sampled) * 100.0, 2),
            "page_numeric_pct": round((with_numeric_page / sampled) * 100.0, 2),
        }
    except Exception as exc:
        return {"sampled": 0, "chunk_id_pct": 0.0, "page_numeric_pct": 0.0, "error": str(exc)}


def bm25_search_docs(
    query: str,
    k: int,
    course_id: Optional[str] = None,
    index_override: Optional[str] = None,
    source_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    filters = build_retrieval_filters(course_id, source_filter)
    if filters:
        bm25_q = {
            "size": k,
            "query": {
                "bool": {
                    "filter": filters,
                    "must": {"match": {"chunk_text": query}},
                }
            },
        }
    else:
        bm25_q = {"size": k, "query": {"match": {"chunk_text": query}}}

    resp = client.search(index=index_override or index_name, body=bm25_q)
    hits = resp.get("hits", {}).get("hits", [])
    return [hit_to_doc(hit) for hit in hits]


def vector_search_docs(
    query: str,
    k: int,
    course_id: Optional[str] = None,
    index_override: Optional[str] = None,
    source_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    emb_body = json.dumps({"inputText": query})
    emb_resp = bedrock.invoke_model(modelId="amazon.titan-embed-text-v2:0", body=emb_body)
    emb = json.loads(emb_resp["body"].read())["embedding"]

    filters = build_retrieval_filters(course_id, source_filter)
    if filters:
        vector_q = {
            "size": k,
            "query": {
                "bool": {
                    "filter": filters,
                    "must": {"knn": {"vector_field": {"vector": emb, "k": k}}},
                }
            },
        }
    else:
        vector_q = {"size": k, "query": {"knn": {"vector_field": {"vector": emb, "k": k}}}}

    resp = client.search(index=index_override or index_name, body=vector_q)
    hits = resp.get("hits", {}).get("hits", [])
    return [hit_to_doc(hit) for hit in hits]


def doc_identity(doc: Dict[str, Any]) -> str:
    return str(doc.get("chunk_id") or doc.get("chunk_hash") or doc.get("doc_id"))


def doc_matches_scope(doc: Dict[str, Any], scope: Dict[str, Any]) -> bool:
    chapters = set(scope.get("chapters") or [])
    sections = set(scope.get("sections") or [])
    page_ranges = scope.get("page_ranges") or []

    if not chapters and not sections and not page_ranges:
        return True

    doc_section = str(doc.get("section_no") or "").strip()
    doc_chapter = str(doc.get("chapter_no") or "").strip()
    if sections and doc_section:
        return doc_section in sections
    if chapters and doc_chapter:
        return doc_chapter in chapters

    start = _to_int(doc.get("page_start"))
    end = _to_int(doc.get("page_end"))
    if start is not None and end is not None and page_ranges:
        for s, e in page_ranges:
            if max(start, s) <= min(end, e):
                return True
        return False

    # Keep doc if scope exists but doc lacks mappable metadata.
    return True


def apply_scope_filter(docs: List[Dict[str, Any]], scope: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [doc for doc in docs if doc_matches_scope(doc, scope)]


def dedupe_docs(docs: List[Dict[str, Any]], take_top: int) -> List[Dict[str, Any]]:
    final_docs: List[Dict[str, Any]] = []
    seen = set()
    for doc in docs:
        key = doc_identity(doc)
        if key in seen:
            continue
        seen.add(key)
        final_docs.append(doc)
        if len(final_docs) >= take_top:
            break
    return final_docs


def merge_and_dedupe_docs(
    bm25_docs: List[Dict[str, Any]],
    vector_docs: List[Dict[str, Any]],
    take_top: int,
    bm25_take: int,
) -> List[Dict[str, Any]]:
    bm25_part = bm25_docs[: max(0, bm25_take)]
    vec_part = vector_docs[: max(0, take_top - max(0, bm25_take))]
    return dedupe_docs(bm25_part + vec_part, take_top)


def build_scope_page_ranges(scope: Dict[str, Any], catalog: Dict[str, Any]) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    section_rows = catalog.get("sections", [])
    chapter_rows = catalog.get("chapters", [])

    selected_sections = set(scope.get("sections") or [])
    selected_chapters = set(scope.get("chapters") or [])

    if selected_sections:
        for row in section_rows:
            if str(row.get("section_no")) in selected_sections:
                start = _to_int(row.get("page_start"))
                end = _to_int(row.get("page_end"))
                if start is not None and end is not None:
                    ranges.append((start, end))
    elif selected_chapters:
        for row in chapter_rows:
            if str(row.get("chapter_no")) in selected_chapters:
                start = _to_int(row.get("page_start"))
                end = _to_int(row.get("page_end"))
                if start is not None and end is not None:
                    ranges.append((start, end))

    ranges = sorted(set(ranges), key=lambda x: (x[0], x[1]))
    return ranges


def chapter_has_sections(chapter_row: Dict[str, Any]) -> bool:
    if chapter_row is None:
        return False
    if chapter_row.get("has_sections") is not None:
        return bool(chapter_row.get("has_sections"))
    section_count = _to_int(chapter_row.get("section_count"))
    if section_count is not None:
        return section_count > 0
    return True


def build_lexical_candidates(unit_query: str, catalog: Dict[str, Any], keep_top: int) -> List[Dict[str, Any]]:
    chapter_lookup = {str(row.get("chapter_no")): row for row in catalog.get("chapters", [])}
    candidates: List[Dict[str, Any]] = []

    for section in catalog.get("sections", []):
        chapter_no = str(section.get("chapter_no") or "")
        chapter = chapter_lookup.get(chapter_no, {})
        candidate_text = " ".join(
            [
                str(section.get("title") or ""),
                str(section.get("summary_text") or ""),
                str(chapter.get("title") or ""),
                str(chapter.get("summary_text") or ""),
            ]
        )
        score = lexical_overlap_score(unit_query, candidate_text)
        candidates.append(
            {
                "type": "section",
                "chapter_no": chapter_no,
                "section_no": str(section.get("section_no") or ""),
                "title": str(section.get("title") or ""),
                "summary_text": str(section.get("summary_text") or ""),
                "page_start": _to_int(section.get("page_start")),
                "page_end": _to_int(section.get("page_end")),
                "score": score,
            }
        )

    for chapter in catalog.get("chapters", []):
        if not chapter_has_sections(chapter):
            continue
        candidate_text = " ".join([str(chapter.get("title") or ""), str(chapter.get("summary_text") or "")])
        score = lexical_overlap_score(unit_query, candidate_text)
        candidates.append(
            {
                "type": "chapter",
                "chapter_no": str(chapter.get("chapter_no") or ""),
                "section_no": None,
                "title": str(chapter.get("title") or ""),
                "summary_text": str(chapter.get("summary_text") or ""),
                "page_start": _to_int(chapter.get("page_start")),
                "page_end": _to_int(chapter.get("page_end")),
                "score": score,
            }
        )

    candidates.sort(key=lambda row: (row.get("score", 0), row.get("type") == "section"), reverse=True)
    return candidates[: max(1, keep_top)]


def build_scope_fallback(candidates: List[Dict[str, Any]], keep_top: int) -> Dict[str, Any]:
    selected_sections: List[str] = []
    selected_chapters: List[str] = []

    for row in candidates:
        if row.get("type") == "section" and row.get("section_no"):
            selected_sections.append(str(row["section_no"]))
        if row.get("chapter_no"):
            selected_chapters.append(str(row["chapter_no"]))
        if len(selected_sections) >= keep_top:
            break

    if not selected_sections:
        for row in candidates:
            if row.get("chapter_no"):
                selected_chapters.append(str(row["chapter_no"]))
            if len(selected_chapters) >= keep_top:
                break

    selected_chapters = list(dict.fromkeys(selected_chapters))
    selected_sections = list(dict.fromkeys(selected_sections))

    max_score = max([row.get("score", 0) for row in candidates], default=0)
    confidence = "high" if max_score >= 4 else ("medium" if max_score >= 2 else "low")

    return {
        "chapters": selected_chapters[:keep_top],
        "sections": selected_sections[:keep_top],
        "confidence": confidence,
        "rationale": "Lexical overlap fallback",
    }


def sanitize_scope_against_catalog(scope: Dict[str, Any], catalog: Dict[str, Any], keep_top: int) -> Tuple[Dict[str, Any], List[str]]:
    scope = dict(scope or {})
    warnings: List[str] = []

    section_rows = catalog.get("sections", [])
    chapter_rows = catalog.get("chapters", [])

    valid_sections = {str(row.get("section_no")): str(row.get("chapter_no") or "") for row in section_rows if row.get("section_no")}
    valid_chapters = {
        str(row.get("chapter_no"))
        for row in chapter_rows
        if row.get("chapter_no") and chapter_has_sections(row)
    }

    requested_chapters = [str(value) for value in scope.get("chapters") or []]
    requested_sections = [str(value) for value in scope.get("sections") or []]

    filtered_sections = [value for value in requested_sections if value in valid_sections]
    filtered_chapters = [value for value in requested_chapters if value in valid_chapters]

    if requested_sections and not filtered_sections:
        warnings.append("Requested sections were not present in catalog; removed invalid section scope.")
    if requested_chapters and not filtered_chapters:
        warnings.append("Requested chapters had no valid sections in catalog; removed invalid chapter scope.")

    if filtered_sections and not filtered_chapters:
        for section_no in filtered_sections:
            chapter_no = valid_sections.get(section_no)
            if chapter_no and chapter_no in valid_chapters:
                filtered_chapters.append(chapter_no)

    if not filtered_sections and filtered_chapters:
        auto_sections = [
            str(row.get("section_no"))
            for row in section_rows
            if str(row.get("chapter_no") or "") in set(filtered_chapters) and row.get("section_no")
        ]
        auto_sections = list(dict.fromkeys(auto_sections))
        if auto_sections:
            filtered_sections = auto_sections[: max(1, keep_top)]
            warnings.append("Scope widened to valid sections under selected chapters.")

    filtered_chapters = list(dict.fromkeys(filtered_chapters))[: max(1, keep_top)]
    filtered_sections = list(dict.fromkeys(filtered_sections))[: max(1, keep_top)]

    scope["chapters"] = filtered_chapters
    scope["sections"] = filtered_sections
    scope["scope_warnings"] = warnings
    return scope, warnings


def rerank_scope_with_llm(
    unit_query: str,
    unit_topics: List[str],
    candidates: List[Dict[str, Any]],
    model_id: str,
    selected_model_name: str,
    keep_top: int,
) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None

    allowed_chapters = sorted({str(row.get("chapter_no")) for row in candidates if row.get("chapter_no")})
    allowed_sections = sorted({str(row.get("section_no")) for row in candidates if row.get("section_no")})
    compact_candidates = [
        {
            "type": row.get("type"),
            "chapter_no": row.get("chapter_no"),
            "section_no": row.get("section_no"),
            "title": row.get("title"),
            "summary_text": row.get("summary_text", "")[:500],
            "score": row.get("score"),
        }
        for row in candidates
    ]

    prompt = f"""
You are a retrieval scope selector for syllabus-guided question generation.
Select the most relevant chapters and sections for the target unit.

Return strict JSON only:
{{
  "chapters": ["..."],
  "sections": ["..."],
  "confidence": "high|medium|low",
  "rationale": "short"
}}

Rules:
- Prefer sections over chapters when specific matches exist.
- Use only chapter_no from: {json.dumps(allowed_chapters)}
- Use only section_no from: {json.dumps(allowed_sections)}
- Keep at most {keep_top} chapters and {keep_top} sections.

Unit Query: {unit_query}
Unit Topics: {json.dumps(unit_topics, ensure_ascii=False)}
Candidates: {json.dumps(compact_candidates, ensure_ascii=False)}
""".strip()

    payload = invoke_model_json(model_id, selected_model_name, prompt, max_tokens=900)
    if not payload:
        return None

    raw_chapters = payload.get("chapters", [])
    raw_sections = payload.get("sections", [])
    chapters = [str(value) for value in raw_chapters if str(value) in set(allowed_chapters)]
    sections = [str(value) for value in raw_sections if str(value) in set(allowed_sections)]
    confidence = str(payload.get("confidence") or "low").lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    return {
        "chapters": list(dict.fromkeys(chapters))[:keep_top],
        "sections": list(dict.fromkeys(sections))[:keep_top],
        "confidence": confidence,
        "rationale": str(payload.get("rationale") or ""),
    }


def build_unit_scope_map(
    units: List[Any],
    unit_meta_map: Dict[str, Dict[str, Any]],
    catalog: Dict[str, Any],
    model_id: str,
    selected_model_name: str,
    candidates_k: int,
    final_k: int,
) -> Dict[str, Dict[str, Any]]:
    scope_map: Dict[str, Dict[str, Any]] = {}
    for unit in units:
        unit_key = str(unit)
        meta = unit_meta_map.get(unit_key, {})
        unit_name = str(meta.get("unit_name") or "").strip()
        unit_topics = meta.get("topics") or []
        query_text = "; ".join(unit_topics) if unit_topics else (unit_name or f"Unit {unit_key}")

        lexical_candidates = build_lexical_candidates(query_text, catalog, keep_top=candidates_k)
        scope = build_scope_fallback(lexical_candidates, keep_top=final_k)
        reranked = rerank_scope_with_llm(
            query_text,
            unit_topics,
            lexical_candidates,
            model_id=model_id,
            selected_model_name=selected_model_name,
            keep_top=final_k,
        )
        if reranked:
            scope = reranked
        scope, scope_warnings = sanitize_scope_against_catalog(scope, catalog, keep_top=final_k)
        scope["page_ranges"] = build_scope_page_ranges(scope, catalog)
        scope["query"] = query_text
        if scope_warnings:
            existing_rationale = str(scope.get("rationale") or "").strip()
            warning_text = "; ".join(scope_warnings)
            scope["rationale"] = f"{existing_rationale} | {warning_text}" if existing_rationale else warning_text
        scope_map[unit_key] = scope
    return scope_map


def docs_to_context_entries(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for doc in docs:
        page_start = _to_int(doc.get("page_start"))
        page_end = _to_int(doc.get("page_end"))
        if page_start is not None and page_end is not None:
            page_range = f"{page_start}-{page_end}"
        else:
            page_range = str(doc.get("page_range") or "?")
        entries.append(
            {
                "source_type": "chunk",
                "id": doc_identity(doc),
                "text": doc.get("chunk_text", ""),
                "chapter_no": doc.get("chapter_no"),
                "section_no": doc.get("section_no"),
                "page_start": page_start,
                "page_end": page_end,
                "page_range": page_range,
            }
        )
    return entries


def kg_rows_to_context_entries(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for row in rows:
        entries.append(
            {
                "source_type": "kg_node",
                "id": f"{row.get('node_label')}|{row.get('node_name')}",
                "text": row.get("description") or row.get("node_name") or "",
                "node_label": row.get("node_label"),
                "node_name": row.get("node_name"),
                "chapter_no": row.get("chapter_no"),
                "section_no": row.get("section_no"),
                "page_start": _to_int(row.get("page_start")),
                "page_end": _to_int(row.get("page_end")),
            }
        )
    return entries


def build_kg_traversal_debug(
    query: str,
    requested_scope: Dict[str, Any],
    rows: List[Dict[str, Any]],
    book_id: Optional[str],
    stage: Optional[str] = None,
    attempts: Optional[List[Dict[str, Any]]] = None,
    effective_scope: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    terms = query_terms_for_kg(query)
    top_rows = rows[: min(len(rows), 12)]
    matched_nodes = []
    for row in top_rows:
        matched_nodes.append(
            {
                "node_label": row.get("node_label"),
                "node_name": row.get("node_name"),
                "chapter_no": row.get("chapter_no"),
                "section_no": row.get("section_no"),
                "page_start": row.get("page_start"),
                "page_end": row.get("page_end"),
            }
        )

    return {
        "query": query,
        "book_id": book_id,
        "requested_scope": {
            "chapters": requested_scope.get("chapters", []),
            "sections": requested_scope.get("sections", []),
            "confidence": requested_scope.get("confidence", "low"),
        },
        "effective_scope": effective_scope or {"chapter_nos": [], "section_nos": []},
        "stage": stage or "unknown",
        "attempts": attempts or [],
        "warnings": warnings or [],
        "terms": terms,
        "traversal_steps": [
            "MATCH (:Book {book_id})-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(:Section)",
            "OPTIONAL MATCH section mentions + subsection mentions content nodes (resilient subsection join)",
            "FILTER by scope (chapter_nos / section_nos) when provided",
            "FILTER by query terms on name/description/statement/expression/steps",
            "RETURN matching nodes ordered by chapter/section/page",
            "If empty, broaden scope in KG-only stages and finally use chapter/section summaries",
        ],
        "rows_retrieved": len(rows),
        "matched_nodes_sample": matched_nodes,
    }


def format_context_text(entries: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for entry in entries:
        if entry.get("source_type") == "kg_node":
            lines.append(
                f"[KG:{entry.get('node_label')}:{entry.get('node_name')} | CH:{entry.get('chapter_no')} "
                f"| SEC:{entry.get('section_no')} | PAGES:{entry.get('page_start')}-{entry.get('page_end')}] "
                f"{entry.get('text', '')}"
            )
        else:
            lines.append(
                f"[DOC_ID:{entry.get('id')} | PAGES:{entry.get('page_range')}] "
                f"{entry.get('text', '')}"
            )
    return "\n\n".join(lines)


def retrieve_embedding(query: str, k: int, course_id: Optional[str], scope: Dict[str, Any]) -> Dict[str, Any]:
    vector_docs = vector_search_docs(
        query,
        k,
        course_id=course_id,
        index_override=retrieval_index_name,
        source_filter=retrieval_source_filter,
    )
    filtered_docs = apply_scope_filter(vector_docs, scope)
    final_docs = dedupe_docs(filtered_docs, take_top=merge_top_k)
    return {
        "mode": "embedding",
        "used_fallback": False,
        "fallback_reason": "",
        "scope": scope,
        "context_entries": docs_to_context_entries(final_docs),
        "debug": {"vector_hits": len(vector_docs), "filtered_hits": len(filtered_docs), "final_hits": len(final_docs)},
    }


def retrieve_hybrid(query: str, k: int, course_id: Optional[str], scope: Dict[str, Any]) -> Dict[str, Any]:
    bm25_docs = bm25_search_docs(
        query,
        k,
        course_id=course_id,
        index_override=retrieval_index_name,
        source_filter=retrieval_source_filter,
    )
    vector_docs = vector_search_docs(
        query,
        k,
        course_id=course_id,
        index_override=retrieval_index_name,
        source_filter=retrieval_source_filter,
    )

    bm25_docs = apply_scope_filter(bm25_docs, scope)
    vector_docs = apply_scope_filter(vector_docs, scope)

    bm25_take = int(k * ratio)
    final_docs = merge_and_dedupe_docs(bm25_docs, vector_docs, take_top=merge_top_k, bm25_take=bm25_take)

    return {
        "mode": "hybrid",
        "used_fallback": False,
        "fallback_reason": "",
        "scope": scope,
        "context_entries": docs_to_context_entries(final_docs),
        "debug": {
            "bm25_hits": len(bm25_docs),
            "vector_hits": len(vector_docs),
            "final_hits": len(final_docs),
            "bm25_take": bm25_take,
        },
    }


def retrieve_kg(query: str, scope: Dict[str, Any]) -> Dict[str, Any]:
    if not questindex_helpers.get("ok"):
        return {
            "mode": "kg",
            "used_fallback": False,
            "fallback_reason": f"QuestIndex helpers unavailable: {questindex_helpers.get('error')}",
            "scope": scope,
            "context_entries": [],
            "debug": {
                "kg_rows": 0,
                "final_hits": 0,
                "stage": "empty",
                "attempts": [{"stage": "strict_scope", "row_count": 0}],
                "effective_scope": {"chapter_nos": [], "section_nos": []},
                "warnings": [f"QuestIndex helpers unavailable: {questindex_helpers.get('error')}"],
                "row_count_per_stage": {"strict_scope": 0},
                "traversal": build_kg_traversal_debug(
                    query,
                    scope,
                    [],
                    quest_book_id,
                    stage="empty",
                    attempts=[{"stage": "strict_scope", "row_count": 0}],
                    effective_scope={"chapter_nos": [], "section_nos": []},
                    warnings=[f"QuestIndex helpers unavailable: {questindex_helpers.get('error')}"],
                ),
            },
        }

    if not quest_book_id:
        return {
            "mode": "kg",
            "used_fallback": False,
            "fallback_reason": "No QuestIndex book selected.",
            "scope": scope,
            "context_entries": [],
            "debug": {
                "kg_rows": 0,
                "final_hits": 0,
                "stage": "empty",
                "attempts": [{"stage": "strict_scope", "row_count": 0}],
                "effective_scope": {"chapter_nos": [], "section_nos": []},
                "warnings": ["No QuestIndex book selected."],
                "row_count_per_stage": {"strict_scope": 0},
                "traversal": build_kg_traversal_debug(
                    query,
                    scope,
                    [],
                    quest_book_id,
                    stage="empty",
                    attempts=[{"stage": "strict_scope", "row_count": 0}],
                    effective_scope={"chapter_nos": [], "section_nos": []},
                    warnings=["No QuestIndex book selected."],
                ),
            },
        }

    try:
        if "retrieve_kg_context_v2" in questindex_helpers:
            kg_payload = questindex_helpers["retrieve_kg_context_v2"](
                quest_book_id,
                chapter_nos=scope.get("chapters") or [],
                section_nos=scope.get("sections") or [],
                query=query,
                limit=merge_top_k * 2,
            )
            rows = kg_payload.get("rows", [])
            stage = kg_payload.get("stage", "strict_scope")
            attempts = kg_payload.get("attempts", [])
            effective_scope = kg_payload.get("effective_scope", {"chapter_nos": [], "section_nos": []})
            warnings = kg_payload.get("warnings", [])
        else:
            rows = questindex_helpers["retrieve_kg_context"](
                quest_book_id,
                chapter_nos=scope.get("chapters") or [],
                section_nos=scope.get("sections") or [],
                query=query,
                limit=merge_top_k * 2,
            )
            stage = "strict_scope"
            attempts = [{"stage": "strict_scope", "row_count": len(rows)}]
            effective_scope = {
                "chapter_nos": scope.get("chapters") or [],
                "section_nos": scope.get("sections") or [],
            }
            warnings = []
    except Exception as exc:
        return {
            "mode": "kg",
            "used_fallback": False,
            "fallback_reason": f"KG retrieval error: {exc}",
            "scope": scope,
            "context_entries": [],
            "debug": {
                "kg_rows": 0,
                "final_hits": 0,
                "stage": "empty",
                "attempts": [{"stage": "strict_scope", "row_count": 0}],
                "effective_scope": {"chapter_nos": [], "section_nos": []},
                "warnings": [f"KG retrieval error: {exc}"],
                "row_count_per_stage": {"strict_scope": 0},
                "traversal": build_kg_traversal_debug(
                    query,
                    scope,
                    [],
                    quest_book_id,
                    stage="empty",
                    attempts=[{"stage": "strict_scope", "row_count": 0}],
                    effective_scope={"chapter_nos": [], "section_nos": []},
                    warnings=[f"KG retrieval error: {exc}"],
                ),
            },
        }

    entries = kg_rows_to_context_entries(rows[:merge_top_k])
    row_count_per_stage = {str(item.get("stage")): int(item.get("row_count", 0)) for item in attempts}
    fallback_reason = ""
    if warnings:
        fallback_reason = " ; ".join(warnings)
    return {
        "mode": "kg",
        "used_fallback": False,
        "fallback_reason": fallback_reason,
        "scope": scope,
        "context_entries": entries,
        "debug": {
            "kg_rows": len(rows),
            "final_hits": len(entries),
            "stage": stage,
            "attempts": attempts,
            "effective_scope": effective_scope,
            "warnings": warnings,
            "row_count_per_stage": row_count_per_stage,
            "traversal": build_kg_traversal_debug(
                query,
                scope,
                rows,
                quest_book_id,
                stage=stage,
                attempts=attempts,
                effective_scope=effective_scope,
                warnings=warnings,
            ),
        },
    }


def retrieve_for_unit(
    query: str,
    requested_mode: str,
    course_id: Optional[str],
    scope: Dict[str, Any],
) -> Dict[str, Any]:
    mode = requested_mode.lower()
    if mode == "embedding":
        return retrieve_embedding(query, num_hits, course_id, scope)

    if mode == "kg":
        return retrieve_kg(query, scope)

    return retrieve_hybrid(query, num_hits, course_id, scope)

if st.button("GENERATE QUESTION PAPER"):
   
    if "qn_matrix" not in st.session_state or st.session_state.qn_matrix.empty:
        st.error("No mapping found in st.session_state.qn_matrix — run Step 1.5 first.")
        st.stop()

    qn_df = st.session_state.qn_matrix.copy()
    generated_items = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    retrieval_diagnostics: List[Dict[str, Any]] = []

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
    unit_meta_map: Dict[str, Dict[str, Any]] = {}
    for unit_meta in st.session_state.get("units_parsed", []):
        unit_no = unit_meta.get("unit_no")
        if unit_no is None:
            continue
        unit_meta_map[str(unit_no)] = {
            "unit_name": unit_meta.get("unit_name") or "",
            "topics": unit_meta.get("topics") or [],
        }

    all_units = sorted(list({str(value) for value in qn_df["Unit"].unique()}), key=unit_sort_key)

    effective_retrieval_filter = (retrieval_course_filter or "").strip() or None
    index_health = compute_index_health(
        effective_retrieval_filter,
        index_override=retrieval_index_name,
        source_filter=retrieval_source_filter,
    )
    if retrieval_mode in {"hybrid", "embedding"}:
        if index_health.get("chunk_id_pct", 0.0) < 80.0:
            st.warning(
                f"Index health warning: chunk_id coverage is {index_health.get('chunk_id_pct')}%. "
                "Deterministic dedupe will fallback to chunk hash where needed."
            )
        if index_health.get("page_numeric_pct", 0.0) < 50.0:
            st.info(
                f"Index has limited numeric page metadata ({index_health.get('page_numeric_pct')}%). "
                "Scope filtering will use legacy page_range parsing fallback."
            )

    catalog: Dict[str, Any] = {"chapters": [], "sections": []}
    unit_scope_map: Dict[str, Dict[str, Any]] = {}
    if quest_book_id and questindex_helpers.get("ok"):
        try:
            catalog = questindex_helpers["get_catalog"](quest_book_id)
            if catalog.get("chapters") or catalog.get("sections"):
                unit_scope_map = build_unit_scope_map(
                    all_units,
                    unit_meta_map,
                    catalog,
                    model_id=model_config["id"],
                    selected_model_name=selected_model,
                    candidates_k=int(scope_candidates_k),
                    final_k=int(scope_final_k),
                )
        except Exception as exc:
            st.warning(f"Failed to build QuestIndex scope map; retrieval continues without structure scoping. ({exc})")

    # iterate units deterministically
    for unit in all_units:
        st.write(f"---\n### Running Process on Unit {unit}")
        # collect topics (if present)
        unit_meta = unit_meta_map.get(str(unit), {})
        unit_topics = unit_meta.get("topics") or []
        unit_name = str(unit_meta.get("unit_name") or "")

        # Build query from topics
        query = "; ".join(unit_topics) if unit_topics else (unit_name or f"Unit {unit}")
        st.write(f"**Query:** {query}")

        scope = unit_scope_map.get(str(unit), {"chapters": [], "sections": [], "confidence": "low", "rationale": "No map", "page_ranges": []})
        retrieval_result = retrieve_for_unit(
            query=query,
            requested_mode=retrieval_mode,
            course_id=effective_retrieval_filter,
            scope=scope,
        )

        if retrieval_result.get("mode") == "kg":
            kg_debug = retrieval_result.get("debug", {})
            traversal_payload = {
                "unit": unit,
                "query": query,
                "fallback_reason": retrieval_result.get("fallback_reason", ""),
                "traversal": kg_debug.get("traversal", {}),
                "stage": kg_debug.get("stage", "unknown"),
                "attempts": kg_debug.get("attempts", []),
                "effective_scope": kg_debug.get("effective_scope", {}),
                "warnings": kg_debug.get("warnings", []),
                "row_count_per_stage": kg_debug.get("row_count_per_stage", {}),
                "kg_rows": kg_debug.get("kg_rows", 0),
                "context_entries_used": kg_debug.get("final_hits", 0),
            }
            with st.expander(f"KG traversal response - Unit {unit}", expanded=False):
                st.json(traversal_payload)

        context_entries = retrieval_result.get("context_entries", [])
        context_text = format_context_text(context_entries)
        if not context_text.strip():
            context_text = f"Unit topics only: {'; '.join(unit_topics)}"

        # Build mapping rows for this unit and expand subdivisions
        rows_for_unit = qn_df[qn_df["Unit"] == str(unit)].to_dict(orient="records")
        mapping_expanded = expand_subdivisions(rows_for_unit)

        # Build the per-unit prompt by filling placeholders
        prompt_filled = PROMPT_TEMPLATE.format(
            course_id=(subject_code),
            subject_name=subject_name,
            unit_no=unit,
            unit_topics=json.dumps(unit_topics),
            mapped_chapters=json.dumps(scope.get("chapters", [])),
            mapped_sections=json.dumps(scope.get("sections", [])),
            mapping_confidence=scope.get("confidence", "low"),
            retrieval_mode_used=retrieval_result.get("mode", retrieval_mode),
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
            retrieval_diagnostics.append(
                {
                    "unit": unit,
                    "query": query,
                    "requested_mode": retrieval_mode,
                    "retrieval_index": retrieval_index_name,
                    "retrieval_source_filter": retrieval_source_filter,
                    "retrieval_course_filter": effective_retrieval_filter,
                    "final_mode": retrieval_result.get("mode"),
                    "used_fallback": retrieval_result.get("used_fallback", False),
                    "fallback_reason": retrieval_result.get("fallback_reason", ""),
                    "scope": {
                        "chapters": scope.get("chapters", []),
                        "sections": scope.get("sections", []),
                        "confidence": scope.get("confidence", "low"),
                        "rationale": scope.get("rationale", ""),
                        "scope_warnings": scope.get("scope_warnings", []),
                    },
                    "effective_scope": retrieval_result.get("debug", {}).get("effective_scope", {}),
                    "stage": retrieval_result.get("debug", {}).get("stage", ""),
                    "attempts": retrieval_result.get("debug", {}).get("attempts", []),
                    "row_count_per_stage": retrieval_result.get("debug", {}).get("row_count_per_stage", {}),
                    "warnings": retrieval_result.get("debug", {}).get("warnings", []),
                    "context_count": len(context_entries),
                    "debug": retrieval_result.get("debug", {}),
                }
            )
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
    st.write(f"- Retrieval mode: {retrieval_mode_label}")
    st.write(f"- Retrieval index: {retrieval_index_name}")
    st.write(f"- Chunk source filter: {retrieval_source_filter or 'Any'}")
    st.write(f"- Course/Book filter: {effective_retrieval_filter or 'None'}")
    st.write(f"- Index health (sample={index_health.get('sampled', 0)}): chunk_id={index_health.get('chunk_id_pct', 0)}%, numeric_pages={index_health.get('page_numeric_pct', 0)}%")

    with st.expander("Retrieval Diagnostics"):
        st.json(
            {
                "quest_book_id": quest_book_id,
                "retrieval_mode": retrieval_mode,
                "index_health": index_health,
                "unit_scope_map": unit_scope_map,
                "unit_diagnostics": retrieval_diagnostics,
            }
        )

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

    st.dataframe(filtered_df, use_container_width=True, height=450)
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
    render_qp_pdf(qp_data, template_name="template2.html", title="Preview and Download Question Paper PDF")
