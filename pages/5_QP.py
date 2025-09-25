
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
# ---------------- Environment & simple config --------------------------------
load_dotenv()

OUT_DIR = os.environ.get("AUQA_OUT_DIR", "/tmp/auqa_output")
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# AWS / Textract config
region = os.environ.get("AWS_REGION", "us-east-1")
bucket_name = os.environ.get("S3_BUCKET", "")   # set in .env or Streamlit UI
textract = boto3.client("textract", region_name=region)
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


def normalize_ratios(e: float, m: float, h: float):
    s = e + m + h
    if s <= 0:
        return 1/3, 1/3, 1/3
    return e/s, m/s, h/s


def integer_distribute_equal(total: int, groups: int):
    """Distribute `total` items as evenly as possible across `groups` buckets."""
    if groups <= 0:
        return []
    base = total // groups
    rem = total % groups
    dist = [base + (1 if i < rem else 0) for i in range(groups)]
    return dist


# ---------------- Environment & Clients -------------------------------------
load_dotenv()

OUT_DIR = os.environ.get("AUQA_OUT_DIR", "/tmp/auqa_output")
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

region = os.environ.get("AWS_REGION", "us-east-1")
bucket_name = os.environ.get("S3_BUCKET", "my-bucket")
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
# ---------------- Streamlit UI ----------------------------------------------
st.set_page_config(layout="wide")
st.title("📘 AUQA: Minimal Question Paper Generator (Up to Step 1.5)")

# Basic metadata
subject_code = st.text_input("Subject Code", "CN")
subject_name = st.text_input("Subject Name", "Computer Networks")
department = st.text_input("Department", "Computer Technology")
semester = st.text_input("Semester", "VII / VIII")
exam_session = st.text_input("Exam Session (e.g. NOV/DEC 2025)", "NOV/DEC 2025")

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
                        topics = unit.get("topics", [])
                        if topics:
                            for t in topics:
                                st.markdown(f"  - {t}")
                        else:
                            st.markdown("  _(No topics found)_")

                except Exception as e:
                    st.error(f"Failed to parse JSON: {e}")
                    st.text_area("Model raw output", generated_text, height=300)
            else:
                st.warning("Model config missing; saving raw text only.")
                st.session_state.raw_text = full_text
    except Exception as e:
        st.error(f"Textract error: {e}")
import streamlit as st
import pandas as pd

# ---------------- Step 1.5: Question mapping (16 rows fixed) -----------
st.set_page_config(layout="wide")
st.title("📘 AUQA: Question Mapping (End-Sem, 100 Marks)")

# CO / BL options (fallback if not parsed from syllabus)
co_options = [c.get("id", f"CO{i+1}") for i, c in enumerate(st.session_state.get("COs", []))]
if not co_options:
    co_options = ["CO1", "CO2", "CO3", "CO4", "CO5"]
bl_options = ["L1", "L2", "L3", "L4", "L5", "L6"]
unit_options = [str(i) for i in range(1, 6)]

# Build exactly 16 fixed questions (Part A=10, Part B=5, Part C=1)
def build_16_qns():
    rows = []
    # Part A: 10 × 2 marks
    for q in range(1, 11):
        rows.append({"QNo": q, "Section": "Part A", "Marks": 2})
    # Part B: 5 × 13 marks
    for q in range(11, 16):
        rows.append({"QNo": q, "Section": "Part B", "Marks": 13})
    # Part C: 1 × 15 marks
    rows.append({"QNo": 16, "Section": "Part C", "Marks": 15})
    return pd.DataFrame(rows, index=range(1, 17))  # force 16 rows only

if "qn_matrix" not in st.session_state or st.session_state.qn_matrix.empty:
    df = build_16_qns()
    n = len(df)
    df["CO"] = [co_options[i % len(co_options)] for i in range(n)]
    df["BL"] = [bl_options[i % len(bl_options)] for i in range(n)]
    df["Unit"] = [unit_options[i % len(unit_options)] for i in range(n)]
    st.session_state.qn_matrix = df

# Data editor (strictly 16 rows)
st.markdown("Edit CO, BL, and Unit for each question below (Marks & QNo fixed). Exactly 16 rows shown.")
edited = st.data_editor(
    st.session_state.qn_matrix,
    num_rows=16,  # force 16 rows
    column_config={
        "QNo": st.column_config.NumberColumn("QNo", disabled=True),
        "Section": st.column_config.TextColumn("Section", disabled=True),
        "Marks": st.column_config.NumberColumn("Marks", disabled=True),
        "CO": st.column_config.SelectboxColumn("CO", options=co_options),
        "BL": st.column_config.SelectboxColumn("BL", options=bl_options),
        "Unit": st.column_config.SelectboxColumn("Unit", options=unit_options),
    },
    use_container_width=True,
)
st.session_state.qn_matrix = edited.copy()

# ---------------- Summary & Graphs ----------------
st.markdown("### Summary & Distribution")

total_marks = int(edited["Marks"].sum())
if total_marks == 100:
    st.success("✅ Grand total = 100 marks")
else:
    st.error(f"❌ Grand total = {total_marks} (expected 100)")
    
if(1==3):
    # Distribution by CO
    co_marks = edited.groupby("CO")["Marks"].sum().reindex(co_options, fill_value=0)
    st.write("**CO-wise Marks Distribution**")
    st.bar_chart(co_marks)

    # Distribution by Bloom’s level
    bl_marks = edited.groupby("BL")["Marks"].sum().reindex(bl_options, fill_value=0)
    st.write("**Bloom’s Level Marks Distribution**")
    st.bar_chart(bl_marks)

    # Distribution by Unit
    unit_marks = edited.groupby("Unit")["Marks"].sum().reindex(unit_options, fill_value=0)
    st.write("**Unit-wise Marks Distribution**")
    st.bar_chart(unit_marks)


# Percentages
co_marks = edited.groupby("CO")["Marks"].sum().reindex(co_options, fill_value=0)
st.write("**CO % of Total (100 marks)**")
percent_by_co = (co_marks / total_marks * 100).round(1)
st.table(percent_by_co.to_frame("Percentage"))

bl_marks = edited.groupby("BL")["Marks"].sum().reindex(bl_options, fill_value=0)
st.write("**BL % of Total (100 marks)**")
percent_by_bl = (bl_marks / total_marks * 100).round(1)
st.table(percent_by_bl.to_frame("Percentage"))

unit_marks = edited.groupby("Unit")["Marks"].sum().reindex(unit_options, fill_value=0)
st.write("**Unit % of Total (100 marks)**")
percent_by_unit = (unit_marks / total_marks * 100).round(1)
st.table(percent_by_unit.to_frame("Percentage"))

# Download
col1, col2 = st.columns(2)
with col1:
    st.download_button("Download Mapping (CSV)", edited.to_csv(index=False), file_name="qn_mapping.csv", mime="text/csv")
with col2:
    st.download_button("Download Mapping (JSON)", edited.to_json(orient="records", indent=2), file_name="qn_mapping.json", mime="application/json")


data = { "roll": ["2","0","2","3","5","0","3","5","5","5"], "title": "ANNA UNIVERSITY (UNIVERSITY DEPARTMENTS)", "subtitle": "B.E. /B.Tech / B. Arch (Full Time) - END SEMESTER EXAMINATIONS, NOV / DEC 2024", "course": "COMPUTER SCIENCE AND ENGINEERING\nIII Semester\nCS23303 DIGITAL SYSTEM DESIGN", "duration": "3hrs", "maxmarks": 100, "cos": [ {"co":"CO1","desc":"Use theorems and K-maps to simplify Boolean functions."}, {"co":"CO2","desc":"Design, analyze and Implement combinational circuits."}, {"co":"CO3","desc":"Design, analyze and implement sequential circuits."}, {"co":"CO4","desc":"Design digital circuits using MSI chips and PLDs."}, {"co":"CO5","desc":"Use HDL to build digital systems."} ], "partA": [ {"no":1,"q":"What are self-complementing codes? Give example.","marks":2,"co":1,"bl":2}, {"no":2,"q":"Simplify the following Boolean expressions to a minimum number of literals (a) ABC + A'B + ABC' (b) (x + y)' (x' + y)","marks":2,"co":1,"bl":3}, {"no":3,"q":"Implement pqr+qr'+p'q using NOR only.","marks":2,"co":2,"bl":3}, {"no":4,"q":"Implement Full adder using MUX.","marks":2,"co":2,"bl":3}, {"no":5,"q":"Show that a Johnson counter with n flip-flops produces a sequence of 2n states. List the 10 states produced with five flip-flops and the Boolean terms of each of the 10 AND gate outputs.","marks":2,"co":3,"bl":4}, {"no":6,"q":"How many flip-flop will be complemented in a 10-bit binary ripple counter to reach the next count after the count 1001100111?","marks":2,"co":3,"bl":4}, {"no":7,"q":"Check whether the asynchronous sequential circuit for Y = x'1x2 + x2y' is stable?","marks":2,"co":4,"bl":5}, {"no":8,"q":"State the closed covering condition.","marks":2,"co":4,"bl":2}, {"no":9,"q":"Obtain the 15-bit Hamming code word for the 11-bit data word 11001001010.","marks":2,"co":5,"bl":3}, {"no":10,"q":"How many 32K * 8 RAM chips are needed to provide a memory capacity of 256K bytes?","marks":2,"co":5,"bl":4} ], "partB": [ {"no":"11(a)","q":"Express the following numbers in decimal: (i) (10110.0101)2 (ii) (26.24)8","marks":13,"co":1,"bl":3}, {"no":"11(b)","q":"Implement the following four Boolean expressions with three half adders: D = A ⊕ B ⊕ C; E = A'BC + AB'C; F = ABC' + (A' + B')C; G = ABC","marks":13,"co":1,"bl":3}, {"no":"12(a)","q":"Design a 3-input majority circuit: give truth table, Boolean equation and logic diagram. Also write Verilog gate-level model.","marks":13,"co":2,"bl":6}, {"no":"12(b)","q":"Design a code converter that converts a decimal digit from BCD to 8, 4, -2, -1 code.","marks":13,"co":2,"bl":6}, {"no":"13(a)","q":"A sequential circuit has two JK flip-flops A and B... Draw logic diagram, state table, derive state equations.","marks":13,"co":3,"bl":5} ], "partC": [ {"no":16,"q":"Simplify using Tabulation method F(A,B,C,D) = Σ(0,2,4,5,6,7,8,10,13,15)","marks":15,"co":1,"bl":5} ] }
render_qp_pdf(data, template_name="qp.html", title="CS23303 — Generate Question Paper PDF")