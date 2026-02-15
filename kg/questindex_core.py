import os
import time
import json
import re
import hashlib
import threading
import difflib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import boto3
import fitz
import streamlit as st
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from neo4j import GraphDatabase
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import parallel_bulk
from requests_aws4auth import AWS4Auth

from bedrockModels import build_request


# ============================================================
# ENV & CLIENTS
# ============================================================
load_dotenv()

REGION = os.environ.get("AWS_REGION", "ap-south-1")
BUCKET = os.environ.get("S3_BUCKET", "")
NEO4J_URI = os.environ.get("NEO4J_URI", "")
NEO4J_USER = os.environ.get("NEO4J_USER", "")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
OS_DOMAIN = os.environ.get("OS_DOMAIN", "")
QUESTINDEX_VECTOR_INDEX = os.environ.get("QUESTINDEX_VECTOR_INDEX", "questindex-chunks")
EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")

BOOKS_DIR = "books"
LOCAL_CACHE_DIR = "kg/cache_questindex"

session = boto3.Session(region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
textract = boto3.client("textract", region_name=REGION)
bedrock = session.client("bedrock-runtime", region_name=REGION)
neo4j = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
opensearch = None
if OS_DOMAIN:
    try:
        credentials = session.get_credentials().get_frozen_credentials()
        awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            REGION,
            "es",
            session_token=credentials.token,
        )
        opensearch = OpenSearch(
            hosts=[{"host": OS_DOMAIN, "port": 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=60,
            max_retries=3,
            retry_on_timeout=True,
        )
    except Exception:
        opensearch = None

progress_lock = threading.Lock()


# ============================================================
# CONSTANTS
# ============================================================
ENTITY_SPECS = {
    "concepts": {"label": "Concept", "description_fields": ["description"]},
    "definitions": {"label": "Definition", "description_fields": ["description", "statement"]},
    "algorithms": {"label": "Algorithm", "description_fields": ["description", "steps"], "extra_fields": ["steps"]},
    "formulas": {"label": "Formula", "description_fields": ["description", "expression"], "extra_fields": ["expression"]},
    "theorems": {"label": "Theorem", "description_fields": ["description", "statement"], "extra_fields": ["statement"]},
    "examples": {"label": "Example", "description_fields": ["description"]},
    "properties": {"label": "Property", "description_fields": ["description"]},
    "key_terms": {"label": "KeyTerm", "description_fields": ["description"]},
    "questions": {
        "label": "Question",
        "description_fields": ["description", "question_text", "question"],
        "extra_fields": ["question_number"],
    },
    "objectives": {"label": "Objective", "description_fields": ["description"]},
    "prerequisites": {"label": "Prerequisite", "description_fields": ["description"]},
    "outcomes": {"label": "Outcome", "description_fields": ["description"]},
}

ALL_CONTENT_LABELS = sorted({spec["label"] for spec in ENTITY_SPECS.values()})

ALLOWED_SEMANTIC_REL_TYPES = {
    "DEPENDS_ON",
    "PART_OF",
    "APPLIES_TO",
    "DERIVED_FROM",
    "USES",
    "IMPLEMENTS",
    "CONTRASTS_WITH",
    "EXTENDS",
    "REQUIRES",
    "PROVES",
    "EXPLAINS",
}

FORBIDDEN_CYPHER_PATTERNS = [
    r"\bCREATE\b",
    r"\bMERGE\b",
    r"\bDELETE\b",
    r"\bDETACH\b",
    r"\bSET\b",
    r"\bREMOVE\b",
    r"\bDROP\b",
    r"\bLOAD\s+CSV\b",
    r"\bCALL\s+dbms\b",
]

LABEL_ALIASES = {
    "concept": "Concept",
    "concepts": "Concept",
    "definition": "Definition",
    "definitions": "Definition",
    "algorithm": "Algorithm",
    "algorithms": "Algorithm",
    "formula": "Formula",
    "formulas": "Formula",
    "theorem": "Theorem",
    "theorems": "Theorem",
    "example": "Example",
    "examples": "Example",
    "property": "Property",
    "properties": "Property",
    "keyterm": "KeyTerm",
    "keyterms": "KeyTerm",
    "key_term": "KeyTerm",
    "key_terms": "KeyTerm",
    "key term": "KeyTerm",
    "question": "Question",
    "questions": "Question",
    "objective": "Objective",
    "objectives": "Objective",
    "prerequisite": "Prerequisite",
    "prerequisites": "Prerequisite",
    "outcome": "Outcome",
    "outcomes": "Outcome",
}


# ============================================================
# PROMPTS
# ============================================================
TOC_PAGE_DETECT_PROMPT = """
You are given one page text from a PDF.
Decide if this page is part of a Table of Contents.

Return strict JSON only:
{
  "toc_detected": "yes" or "no",
  "reason": "short reason"
}
""".strip()

TOC_PARSE_PROMPT = """
Extract TOC entries from text. Keep up to 3 hierarchy levels only.

Return strict JSON only with this structure:
{
  "entries": [
    {
      "level": "chapter" | "section" | "subsection",
      "number": "string or null",
      "title": "string",
      "printed_page": 12 or null
    }
  ]
}
""".strip()

ENTITY_EXTRACTION_PROMPT = """
You are an extraction engine for engineering syllabus and textbook content.

Task:
- Extract as much structured content as possible from the given segment.
- The segment belongs to one anchor (Section or Subsection).
- Use only information present in text.

Output STRICT JSON matching exactly this schema:
{
  "concepts": [{"name":"", "description":"", "relationships":[]}],
  "definitions": [{"name":"", "description":"", "relationships":[]}],
  "algorithms": [{"name":"", "description":"", "steps":"", "relationships":[]}],
  "formulas": [{"name":"", "expression":"", "description":"", "relationships":[]}],
  "theorems": [{"name":"", "statement":"", "description":"", "relationships":[]}],
  "examples": [{"name":"", "description":"", "relationships":[]}],
  "properties": [{"name":"", "description":"", "relationships":[]}],
  "key_terms": [{"name":"", "description":"", "relationships":[]}],
  "questions": [{"name":"", "question_number":"", "relationships":[]}],
  "objectives": [{"name":"", "description":"", "relationships":[]}],
  "prerequisites": [{"name":"", "description":"", "relationships":[]}],
  "outcomes": [{"name":"", "description":"", "relationships":[]}],
  "relationships": [{"source":"", "source_type":"", "target":"", "target_type":"", "type":""}]
}

Rules:
- Return arrays for all keys, even if empty.
- Keep names concise and canonical.
- Do not invent unsupported relations.
- Output JSON only.
""".strip()

SUMMARY_PROMPT = """
Generate a concise technical summary for the given academic content.

Return plain text only, max 180 words.
Include key concepts, methods, and expected outcomes.
""".strip()

CYPHER_GEN_PROMPT = """
You generate safe, read-only Cypher for Neo4j.

Output strict JSON only:
{
  "intent": "",
  "cypher": "",
  "params": {"book_id":"", "chapter_no": null, "section_no": null, "limit": 20},
  "expected_columns": []
}

Rules:
- READ ONLY.
- Include parameterized scope filters using $book_id and optional $chapter_no/$section_no when provided.
- Use only MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT, UNWIND.
- No CREATE/MERGE/DELETE/SET/REMOVE/DROP.
- Always include LIMIT.
""".strip()

CYPHER_REPAIR_PROMPT = """
Repair this Cypher to be executable and safe.

Output strict JSON only:
{
  "intent": "",
  "cypher": "",
  "params": {"book_id":"", "chapter_no": null, "section_no": null, "limit": 20},
  "expected_columns": []
}

Rules:
- READ ONLY.
- Keep parameterized scope.
- No forbidden clauses.
- Keep intent equivalent.
""".strip()

ANSWER_SYNTHESIS_PROMPT = """
Create a structured JSON answer from graph query results.

Return strict JSON only:
{
  "answer": "",
  "cypher": "",
  "scope": {"book_id":"", "chapter_no":null, "section_no":null},
  "evidence": [{"path":"", "node_label":"", "node_name":"", "page_start":0, "page_end":0}],
  "confidence": "high|medium|low",
  "follow_up": []
}
""".strip()


# ============================================================
# GENERIC HELPERS
# ============================================================
def safe_json_load(text: str) -> Optional[Any]:
    try:
        value = text.strip()
        if value.startswith("```json"):
            value = value[7:]
        if value.startswith("```"):
            value = value[3:]
        if value.endswith("```"):
            value = value[:-3]
        return json.loads(value.strip())
    except Exception:
        return None


def extract_first_json_block(text: str) -> Optional[str]:
    if not text:
        return None

    start_candidates = [idx for idx in [text.find("{"), text.find("[")] if idx != -1]
    if not start_candidates:
        return None
    start = min(start_candidates)

    stack: List[str] = []
    in_string = False
    escape = False

    pairs = {"{": "}", "[": "]"}
    opens = set(pairs.keys())
    closes = set(pairs.values())

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch in opens:
            stack.append(pairs[ch])
        elif ch in closes:
            if not stack or ch != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return text[start : i + 1]

    return None


def parse_json_response(raw_text: str) -> Optional[Any]:
    if not raw_text:
        return None

    direct = safe_json_load(raw_text)
    if direct is not None:
        return direct

    extracted = extract_first_json_block(raw_text)
    if extracted is None:
        return None
    return safe_json_load(extracted)


def invoke_model_text(model_id: str, selected_model: str, prompt: str, max_output_tokens: int = 3000) -> str:
    body = build_request(model_id, selected_model, prompt, max_output_tokens)
    resp = bedrock.invoke_model(modelId=model_id, body=json.dumps(body))
    payload = json.loads(resp["body"].read())

    if isinstance(payload, dict):
        if "content" in payload and isinstance(payload["content"], list):
            parts = []
            for item in payload["content"]:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
            if parts:
                return "\n".join(parts)

        if "output" in payload:
            try:
                return payload["output"]["message"]["content"][0]["text"]
            except Exception:
                pass

        if "generation" in payload:
            return str(payload.get("generation", ""))

        if "completions" in payload and payload["completions"]:
            first = payload["completions"][0]
            if isinstance(first, dict):
                return str(first.get("data", {}).get("text", ""))

    return json.dumps(payload)


def invoke_model_json(model_id: str, selected_model: str, prompt: str, max_output_tokens: int = 3000) -> Optional[Any]:
    raw_text = invoke_model_text(model_id, selected_model, prompt, max_output_tokens=max_output_tokens)
    return parse_json_response(raw_text)


def make_id(text: str, prefix: str = "") -> str:
    return hashlib.blake2b(f"{prefix}{text}".encode("utf-8"), digest_size=12).hexdigest()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_name(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def normalize_chunk_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def make_chunk_id(book_id: str, chunk_text: str) -> str:
    # Keep chunk-id generation aligned with pages/1_Ingestion.py.
    normalized = normalize_chunk_text(chunk_text)
    key_text = normalized[:48] if len(normalized) > 48 else normalized
    raw = f"{book_id}|{key_text}"
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()


def make_chunk_hash(chunk_text: str) -> str:
    normalized = normalize_chunk_text(chunk_text)
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).hexdigest()


def normalize_heading_for_match(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\.\s]", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_rel_type(value: Any) -> Optional[str]:
    raw = normalize_name(value)
    if not raw:
        return None
    cleaned = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    return cleaned.upper() if cleaned else None


def normalize_label(value: Any) -> Optional[str]:
    key = normalize_name(value).replace("-", "_")
    if key in LABEL_ALIASES:
        return LABEL_ALIASES[key]
    return None


def to_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None


def make_content_node_id(book_id: str, label: str, name: str) -> str:
    normalized = normalize_name(name)
    return make_id(f"{book_id}|{label}|{normalized}", prefix="content|")


def _read_json_file(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_json_file(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _s3_read_json(key: str) -> Optional[Any]:
    try:
        response = s3.get_object(Bucket=BUCKET, Key=key)
        body = response["Body"].read().decode("utf-8")
        return json.loads(body)
    except Exception:
        return None


def list_books_folder_pdfs() -> List[str]:
    if not os.path.isdir(BOOKS_DIR):
        return []
    return sorted([name for name in os.listdir(BOOKS_DIR) if name.lower().endswith(".pdf")])


def resolve_books_pdf_path(filename: str) -> Optional[str]:
    if not filename:
        return None

    candidate = os.path.abspath(os.path.join(BOOKS_DIR, filename))
    books_root = os.path.abspath(BOOKS_DIR)
    if not candidate.startswith(books_root + os.sep) and candidate != books_root:
        return None
    return candidate if os.path.isfile(candidate) else None


def load_pdf_bytes(source: Any, source_kind: str) -> Tuple[bytes, str]:
    if source_kind == "books":
        pdf_path = resolve_books_pdf_path(source)
        if not pdf_path:
            raise FileNotFoundError(f"PDF not found in books folder: {source}")
        with open(pdf_path, "rb") as f:
            return f.read(), pdf_path

    if source_kind == "upload":
        if source is None:
            raise Exception("No uploaded file provided")
        return source.getvalue(), source.name

    if source_kind == "path":
        with open(source, "rb") as f:
            return f.read(), source

    raise Exception(f"Unsupported source kind: {source_kind}")


def extract_text_with_pymupdf(pdf_bytes: bytes) -> Tuple[Dict[str, List[str]], int, int]:
    pages: Dict[str, List[str]] = {}
    total_pages = 0

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        total_pages = len(doc)
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if lines:
                pages[str(page_index)] = lines

    total_chars = sum(len(" ".join(lines)) for lines in pages.values())
    return pages, total_pages, total_chars


def is_pymupdf_text_sufficient(pages: Dict[str, List[str]], total_pages: int, total_chars: int) -> bool:
    if total_pages <= 0:
        return False

    non_empty_pages = len(pages)
    coverage = non_empty_pages / total_pages
    min_char_threshold = max(300, total_pages * 20)
    return non_empty_pages > 0 and coverage >= 0.35 and total_chars >= min_char_threshold


def extract_text_with_textract_ocr(pdf_bytes: bytes, pdf_key: str) -> Dict[str, List[str]]:
    if not BUCKET:
        raise Exception("S3_BUCKET is missing in environment")

    def get_with_backoff(job_id: str, next_token: Optional[str] = None, max_attempts: int = 8) -> Dict[str, Any]:
        delay = 1.0
        for attempt in range(1, max_attempts + 1):
            try:
                if next_token:
                    return textract.get_document_text_detection(JobId=job_id, NextToken=next_token)
                return textract.get_document_text_detection(JobId=job_id)
            except ClientError as exc:
                error_code = exc.response.get("Error", {}).get("Code", "")
                is_throttled = error_code in {
                    "ProvisionedThroughputExceededException",
                    "ThrottlingException",
                    "TooManyRequestsException",
                }
                if not is_throttled or attempt == max_attempts:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 20)

        raise Exception("Textract backoff exhausted")

    s3.put_object(Bucket=BUCKET, Key=pdf_key, Body=pdf_bytes)
    job = textract.start_document_text_detection(
        DocumentLocation={"S3Object": {"Bucket": BUCKET, "Name": pdf_key}}
    )

    while True:
        result = get_with_backoff(job["JobId"])
        status = result["JobStatus"]
        if status in ["SUCCEEDED", "FAILED"]:
            break
        time.sleep(3)

    if status == "FAILED":
        raise Exception("Textract OCR processing failed")

    pages: Dict[str, List[str]] = {}
    token = None
    while True:
        resp = get_with_backoff(job["JobId"], token) if token else result
        for block in resp.get("Blocks", []):
            if block.get("BlockType") == "LINE":
                page = str(block.get("Page"))
                pages.setdefault(page, []).append(block.get("Text", ""))
        token = resp.get("NextToken")
        if not token:
            break

    return pages


def extract_pdf_text_local(pdf_source: Any, source_kind: str, book_id: str) -> Dict[str, List[str]]:
    pdf_bytes, source_name = load_pdf_bytes(pdf_source, source_kind)
    if not pdf_bytes:
        raise Exception("Selected PDF is empty")

    pdf_hash = sha256_hex(pdf_bytes)
    cache_prefix = os.path.join(LOCAL_CACHE_DIR, book_id)
    json_path = os.path.join(cache_prefix, "text.json")
    metadata_path = os.path.join(cache_prefix, "metadata.json")

    cached_metadata = _read_json_file(metadata_path)
    cached_pages = _read_json_file(json_path)
    if (
        cached_metadata
        and cached_pages
        and cached_metadata.get("book_id") == book_id
        and cached_metadata.get("pdf_sha256") == pdf_hash
    ):
        st.success(f"Using local cached text for book: {book_id}")
        return cached_pages

    pages, total_pages, total_chars = extract_text_with_pymupdf(pdf_bytes)
    extraction_method = "pymupdf"

    if not is_pymupdf_text_sufficient(pages, total_pages, total_chars):
        if BUCKET:
            st.warning("PyMuPDF text appears sparse. Trying Textract OCR fallback...")
            try:
                pdf_key = f"KG_QUESTINDEX/{book_id}/source.pdf"
                pages = extract_text_with_textract_ocr(pdf_bytes, pdf_key)
                extraction_method = "textract_ocr_fallback"
            except Exception as ocr_error:
                if not pages:
                    raise Exception(f"No text extracted from selected PDF. Textract fallback failed: {ocr_error}")
                extraction_method = "pymupdf_partial"
        else:
            extraction_method = "pymupdf_partial"

    if not pages:
        raise Exception("No text extracted from selected PDF")

    _write_json_file(json_path, pages)
    _write_json_file(
        metadata_path,
        {
            "book_id": book_id,
            "pdf_filename": os.path.basename(source_name),
            "pdf_sha256": pdf_hash,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "page_count": len(pages),
            "extraction_method": extraction_method,
        },
    )

    st.success(f"Processed and cached {len(pages)} pages for {book_id} via {extraction_method}")
    return pages


def extract_pdf_text_s3(pdf_source: Any, source_kind: str, book_id: str) -> Dict[str, List[str]]:
    if not BUCKET:
        raise Exception("S3_BUCKET is missing in environment")

    pdf_bytes, _ = load_pdf_bytes(pdf_source, source_kind)
    if not pdf_bytes:
        raise Exception("Selected PDF is empty")

    pdf_hash = sha256_hex(pdf_bytes)
    cache_prefix = f"KG_QUESTINDEX/{book_id}"
    json_key = f"{cache_prefix}/text.json"
    metadata_key = f"{cache_prefix}/metadata.json"
    pdf_key = f"{cache_prefix}/source.pdf"

    cached_metadata = _s3_read_json(metadata_key)
    cached_pages = _s3_read_json(json_key)
    if (
        cached_metadata
        and cached_pages
        and cached_metadata.get("book_id") == book_id
        and cached_metadata.get("pdf_sha256") == pdf_hash
    ):
        st.success(f"Using S3 cached text for book: {book_id}")
        return cached_pages

    pages, total_pages, total_chars = extract_text_with_pymupdf(pdf_bytes)
    extraction_method = "pymupdf"

    if not is_pymupdf_text_sufficient(pages, total_pages, total_chars):
        st.info("PyMuPDF text appears sparse. Falling back to Textract OCR...")
        pages = extract_text_with_textract_ocr(pdf_bytes, pdf_key)
        extraction_method = "textract_ocr"

    if not pages:
        raise Exception("No text extracted from selected PDF")

    s3.put_object(Bucket=BUCKET, Key=json_key, Body=json.dumps(pages), ContentType="application/json")
    s3.put_object(
        Bucket=BUCKET,
        Key=metadata_key,
        Body=json.dumps(
            {
                "book_id": book_id,
                "pdf_sha256": pdf_hash,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "page_count": len(pages),
                "extraction_method": extraction_method,
            }
        ),
        ContentType="application/json",
    )

    st.success(f"Processed and cached {len(pages)} pages for {book_id} via {extraction_method}")
    return pages


def extract_pdf_text(pdf_source: Any, source_kind: str, book_id: str, extraction_mode: str) -> Dict[str, List[str]]:
    if extraction_mode == "s3":
        return extract_pdf_text_s3(pdf_source, source_kind, book_id)
    return extract_pdf_text_local(pdf_source, source_kind, book_id)


def sort_page_keys(keys: List[str]) -> List[str]:
    return sorted(keys, key=lambda k: (0, int(k)) if str(k).isdigit() else (1, str(k)))


def filter_pages_by_range(
    pages: Dict[str, List[str]], start_page: int, end_page: int
) -> Tuple[Dict[str, List[str]], Optional[Dict[str, int]]]:
    numeric_pages = [int(k) for k in pages.keys() if str(k).isdigit()]
    if not numeric_pages:
        return pages, None

    min_page = min(numeric_pages)
    max_page = max(numeric_pages)
    start = max(start_page, min_page)
    end = max_page if end_page <= 0 else min(end_page, max_page)

    if start > end:
        return {}, {
            "available_min": min_page,
            "available_max": max_page,
            "applied_start": start,
            "applied_end": end,
        }

    filtered = {k: v for k, v in pages.items() if str(k).isdigit() and start <= int(k) <= end}
    return filtered, {
        "available_min": min_page,
        "available_max": max_page,
        "applied_start": start,
        "applied_end": end,
    }


def pages_to_text_map(pages: Dict[str, List[str]]) -> Dict[int, str]:
    text_map: Dict[int, str] = {}
    for key in sort_page_keys(list(pages.keys())):
        if not str(key).isdigit():
            continue
        text_map[int(key)] = "\n".join(pages[key])
    return text_map


def get_page_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


# ============================================================
# TOC + STRUCTURE BUILDING
# ============================================================
def roman_to_int(value: str) -> Optional[int]:
    value = value.upper().strip()
    if not value or not re.fullmatch(r"[IVXLCDM]+", value):
        return None

    roman_map = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    prev = 0
    for ch in reversed(value):
        cur = roman_map[ch]
        if cur < prev:
            total -= cur
        else:
            total += cur
        prev = cur
    return total if total > 0 else None


def toc_regex_score(page_text: str) -> int:
    text = page_text or ""
    score = 0

    if re.search(r"\b(table\s+of\s+contents|contents)\b", text, re.IGNORECASE):
        score += 3

    numbered_lines = re.findall(
        r"(?im)^\s*(?:chapter\s+\d+|\d+(?:\.\d+){0,3})\s+.{3,}\s+\d{1,4}\s*$",
        text,
    )
    score += min(4, len(numbered_lines))

    dotted_lines = re.findall(r"(?im)^\s*.+\.{3,}\s*\d{1,4}\s*$", text)
    score += min(3, len(dotted_lines))

    return score


def llm_toc_page_detector(page_text: str, model_id: str, selected_model: str) -> bool:
    prompt = (
        f"{TOC_PAGE_DETECT_PROMPT}\n\n"
        f"Page Text:\n{page_text[:7000]}"
    )
    payload = invoke_model_json(model_id, selected_model, prompt, max_output_tokens=300)
    if not isinstance(payload, dict):
        return False
    return normalize_name(payload.get("toc_detected")) == "yes"


def detect_toc_pages(
    page_text_map: Dict[int, str],
    toc_scan_pages: int,
    model_id: str,
    selected_model: str,
) -> List[int]:
    candidate_pages: List[int] = []
    sorted_pages = sorted(page_text_map.keys())

    for page_no in sorted_pages[:toc_scan_pages]:
        score = toc_regex_score(page_text_map[page_no])
        if score >= 4:
            candidate_pages.append(page_no)

    if candidate_pages:
        first = min(candidate_pages)
        contiguous = []
        for page_no in sorted_pages:
            if page_no < first:
                continue
            if page_no > first + 12:
                break
            score = toc_regex_score(page_text_map[page_no])
            if score >= 2:
                contiguous.append(page_no)
            elif contiguous:
                break
        return contiguous or candidate_pages

    llm_candidates = []
    for page_no in sorted_pages[: min(toc_scan_pages, 12)]:
        score = toc_regex_score(page_text_map[page_no])
        if score >= 2 or page_no <= 4:
            if llm_toc_page_detector(page_text_map[page_no], model_id, selected_model):
                llm_candidates.append(page_no)

    if not llm_candidates:
        return []

    start = min(llm_candidates)
    toc_pages = []
    for page_no in sorted_pages:
        if page_no < start:
            continue
        if page_no > start + 12:
            break
        score = toc_regex_score(page_text_map[page_no])
        if score >= 2:
            toc_pages.append(page_no)
        elif page_no in llm_candidates:
            toc_pages.append(page_no)
        elif toc_pages:
            break

    return toc_pages or llm_candidates


def merge_toc_lines(lines: List[str]) -> List[str]:
    cleaned = [line.strip() for line in lines if line and line.strip()]
    merged: List[str] = []
    i = 0
    while i < len(cleaned):
        line = cleaned[i]
        if (
            i + 1 < len(cleaned)
            and re.fullmatch(r"\d{1,4}", cleaned[i + 1])
            and not re.search(r"\d{1,4}\s*$", line)
        ):
            line = f"{line} {cleaned[i + 1]}"
            i += 1
        merged.append(line)
        i += 1
    return merged


def _chapter_no_from_raw(raw_no: Optional[str], fallback_index: int) -> str:
    if not raw_no:
        return str(fallback_index)

    raw = raw_no.strip().strip(".:")
    if raw.isdigit():
        return str(int(raw))

    maybe_roman = roman_to_int(raw)
    if maybe_roman is not None:
        return str(maybe_roman)

    return str(fallback_index)


def _normalize_number_token(number: str) -> Optional[str]:
    if not number:
        return None
    cleaned = re.sub(r"[^0-9\.]", "", number)
    cleaned = re.sub(r"\.{2,}", ".", cleaned).strip(".")
    if not cleaned:
        return None
    parts = [p for p in cleaned.split(".") if p]
    if not parts:
        return None
    if not all(part.isdigit() for part in parts):
        return None
    return ".".join(str(int(part)) for part in parts)


def parse_toc_entries_regex(raw_lines: List[str]) -> List[Dict[str, Any]]:
    lines = merge_toc_lines(raw_lines)
    entries: List[Dict[str, Any]] = []
    seen = set()
    chapter_fallback = 1

    for order, line in enumerate(lines):
        line_clean = re.sub(r"\s+", " ", line).strip()
        if not line_clean:
            continue
        if re.fullmatch(r"(contents|table of contents|part\s+\w+)", line_clean, flags=re.IGNORECASE):
            continue

        chapter_match = re.match(
            r"(?i)^\s*(?:chapter|unit)\s+([0-9IVXLC]+)\s*[:\.-]?\s*(.*?)(?:\s+(\d{1,4}))?\s*$",
            line_clean,
        )
        if chapter_match:
            chapter_no = _chapter_no_from_raw(chapter_match.group(1), chapter_fallback)
            chapter_fallback = to_int(chapter_no) + 1 if chapter_no.isdigit() else chapter_fallback + 1
            title = chapter_match.group(2).strip() or f"Chapter {chapter_no}"
            printed_page = to_int(chapter_match.group(3))
            entry = {
                "level": "chapter",
                "chapter_no": chapter_no,
                "section_no": None,
                "subsection_no": None,
                "title": title,
                "printed_page": printed_page,
                "order": order,
            }
            key = (entry["level"], entry["chapter_no"], normalize_name(entry["title"]))
            if key not in seen:
                entries.append(entry)
                seen.add(key)
            continue

        numbered_match = re.match(
            r"^\s*(\d+(?:\.\d+){0,4})\s+(.+?)(?:\s+(\d{1,4}))?\s*$",
            line_clean,
        )
        if numbered_match:
            number = _normalize_number_token(numbered_match.group(1) or "")
            if not number:
                continue
            title = numbered_match.group(2).strip(" .:-")
            if not title:
                continue
            printed_page = to_int(numbered_match.group(3))
            parts = number.split(".")

            if len(parts) == 1:
                level = "chapter"
                chapter_no = parts[0]
                section_no = None
                subsection_no = None
            elif len(parts) == 2:
                level = "section"
                chapter_no = parts[0]
                section_no = ".".join(parts[:2])
                subsection_no = None
            else:
                level = "subsection"
                chapter_no = parts[0]
                section_no = ".".join(parts[:2])
                subsection_no = ".".join(parts[:3])

            entry = {
                "level": level,
                "chapter_no": chapter_no,
                "section_no": section_no,
                "subsection_no": subsection_no,
                "title": title,
                "printed_page": printed_page,
                "order": order,
            }

            key = (
                entry["level"],
                entry.get("chapter_no"),
                entry.get("section_no"),
                entry.get("subsection_no"),
                normalize_name(entry["title"]),
            )
            if key not in seen:
                entries.append(entry)
                seen.add(key)
            continue

        section_word_match = re.match(
            r"(?i)^\s*section\s+(\d+(?:\.\d+){1,3})\s*[:\.-]?\s*(.*?)(?:\s+(\d{1,4}))?\s*$",
            line_clean,
        )
        if section_word_match:
            number = _normalize_number_token(section_word_match.group(1) or "")
            if not number:
                continue
            parts = number.split(".")
            title = section_word_match.group(2).strip() or f"Section {number}"
            printed_page = to_int(section_word_match.group(3))
            level = "section" if len(parts) == 2 else "subsection"
            entry = {
                "level": level,
                "chapter_no": parts[0],
                "section_no": ".".join(parts[:2]),
                "subsection_no": ".".join(parts[:3]) if len(parts) >= 3 else None,
                "title": title,
                "printed_page": printed_page,
                "order": order,
            }
            key = (
                entry["level"],
                entry.get("chapter_no"),
                entry.get("section_no"),
                entry.get("subsection_no"),
                normalize_name(entry["title"]),
            )
            if key not in seen:
                entries.append(entry)
                seen.add(key)

    return sorted(entries, key=lambda x: x.get("order", 0))


def parse_toc_entries_llm(toc_text: str, model_id: str, selected_model: str) -> List[Dict[str, Any]]:
    prompt = (
        f"{TOC_PARSE_PROMPT}\n\n"
        f"TOC Text:\n{toc_text[:18000]}"
    )
    payload = invoke_model_json(model_id, selected_model, prompt, max_output_tokens=2000)
    if not isinstance(payload, dict):
        return []

    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        return []

    normalized: List[Dict[str, Any]] = []
    order = 0
    for item in entries:
        if not isinstance(item, dict):
            continue
        level = normalize_name(item.get("level"))
        number = _normalize_number_token(str(item.get("number") or ""))
        title = str(item.get("title") or "").strip()
        printed_page = to_int(item.get("printed_page"))

        if level not in {"chapter", "section", "subsection"}:
            if number:
                parts = number.split(".")
                if len(parts) == 1:
                    level = "chapter"
                elif len(parts) == 2:
                    level = "section"
                else:
                    level = "subsection"
            else:
                continue

        chapter_no = None
        section_no = None
        subsection_no = None

        if number:
            parts = number.split(".")
            chapter_no = parts[0]
            if len(parts) >= 2:
                section_no = ".".join(parts[:2])
            if len(parts) >= 3:
                subsection_no = ".".join(parts[:3])

        if not title:
            if level == "chapter" and chapter_no:
                title = f"Chapter {chapter_no}"
            elif level == "section" and section_no:
                title = f"Section {section_no}"
            elif level == "subsection" and subsection_no:
                title = f"Subsection {subsection_no}"
            else:
                continue

        normalized.append(
            {
                "level": level,
                "chapter_no": chapter_no,
                "section_no": section_no,
                "subsection_no": subsection_no,
                "title": title,
                "printed_page": printed_page,
                "order": order,
            }
        )
        order += 1

    return normalized


def score_heading_match(entry: Dict[str, Any], page_text: str) -> float:
    text_norm = normalize_heading_for_match(page_text)
    if not text_norm:
        return 0.0

    title_norm = normalize_heading_for_match(entry.get("title", ""))
    if not title_norm:
        return 0.0

    number = entry.get("subsection_no") or entry.get("section_no") or entry.get("chapter_no")
    number_norm = normalize_heading_for_match(number or "")

    score = 0.0
    if title_norm in text_norm:
        score += 8.0

    ratio = difflib.SequenceMatcher(None, title_norm[:150], text_norm[:1200]).ratio()
    score += ratio * 3.0

    title_tokens = [tok for tok in title_norm.split() if len(tok) > 2][:8]
    overlap = sum(1 for tok in title_tokens if tok in text_norm)
    score += min(4.0, overlap * 0.6)

    if number_norm:
        if re.search(rf"\b{re.escape(number_norm)}\b", text_norm):
            score += 2.0

    return score


def find_heading_page_by_fuzzy(entry: Dict[str, Any], page_text_map: Dict[int, str]) -> Optional[int]:
    best_page = None
    best_score = 0.0

    for page_no in sorted(page_text_map.keys()):
        score = score_heading_match(entry, page_text_map[page_no])
        if score > best_score:
            best_score = score
            best_page = page_no

    if best_score >= 3.2:
        return best_page
    return None


def infer_heading_page_with_llm(
    entry: Dict[str, Any],
    page_text_map: Dict[int, str],
    model_id: str,
    selected_model: str,
) -> Optional[int]:
    title = entry.get("title", "")
    number = entry.get("subsection_no") or entry.get("section_no") or entry.get("chapter_no") or ""

    scored = []
    for page_no, page_text in page_text_map.items():
        scored.append((score_heading_match(entry, page_text), page_no, page_text))
    scored.sort(reverse=True, key=lambda x: x[0])

    candidates = scored[:20]
    snippets = []
    for _, page_no, page_text in candidates:
        lines = get_page_lines(page_text)[:6]
        snippet = " ".join(lines)
        snippets.append(f"Page {page_no}: {snippet[:300]}")

    prompt = f"""
Find the most likely start page for this heading.

Heading number: {number}
Heading title: {title}

Candidate page snippets:
{chr(10).join(snippets)}

Return strict JSON only:
{{
  "page": <integer or null>,
  "confidence": "high|medium|low"
}}
"""

    payload = invoke_model_json(model_id, selected_model, prompt, max_output_tokens=250)
    if not isinstance(payload, dict):
        return None

    page = to_int(payload.get("page"))
    if page in page_text_map:
        return page
    return None


def assign_physical_pages_to_entries(
    entries: List[Dict[str, Any]],
    page_text_map: Dict[int, str],
    model_id: str,
    selected_model: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not entries:
        return entries, {"offset": None, "unresolved": 0}

    total_pages = max(page_text_map.keys()) if page_text_map else 0

    offset_candidates: List[int] = []
    for entry in entries[:50]:
        printed_page = entry.get("printed_page")
        if printed_page is None:
            continue
        fuzzy_page = find_heading_page_by_fuzzy(entry, page_text_map)
        if fuzzy_page is not None:
            offset_candidates.append(fuzzy_page - printed_page)

    offset = None
    if offset_candidates:
        offset = Counter(offset_candidates).most_common(1)[0][0]

    for entry in entries:
        printed_page = entry.get("printed_page")
        if printed_page is not None and offset is not None:
            mapped = printed_page + offset
            if 1 <= mapped <= total_pages:
                entry["page_start"] = mapped
                continue
        entry["page_start"] = None

    unresolved = [entry for entry in entries if entry.get("page_start") is None]

    for entry in unresolved:
        fuzzy_page = find_heading_page_by_fuzzy(entry, page_text_map)
        if fuzzy_page is not None:
            entry["page_start"] = fuzzy_page

    unresolved = [entry for entry in entries if entry.get("page_start") is None]
    for entry in unresolved[:20]:
        inferred_page = infer_heading_page_with_llm(entry, page_text_map, model_id, selected_model)
        if inferred_page is not None:
            entry["page_start"] = inferred_page

    # forward/backward fill for unresolved entries
    ordered = sorted(entries, key=lambda x: x.get("order", 0))
    last_page = 1
    for entry in ordered:
        if entry.get("page_start") is None:
            entry["page_start"] = last_page
        else:
            last_page = int(entry["page_start"])

    next_page = total_pages if total_pages > 0 else 1
    for entry in reversed(ordered):
        if entry.get("page_start") is None:
            entry["page_start"] = next_page
        else:
            next_page = int(entry["page_start"])

    for entry in entries:
        entry["page_start"] = max(1, min(total_pages if total_pages > 0 else 1, int(entry["page_start"])))

    unresolved_count = sum(1 for entry in entries if entry.get("page_start") is None)
    return entries, {"offset": offset, "unresolved": unresolved_count}


def _sort_no_key(value: Optional[str]) -> Tuple[int, ...]:
    if not value:
        return (10_000_000,)
    try:
        return tuple(int(part) for part in value.split("."))
    except Exception:
        return (10_000_000,)


def _assign_end_pages(items: List[Dict[str, Any]], total_pages: int) -> None:
    if not items:
        return

    ordered = sorted(items, key=lambda x: (x.get("page_start", 1), _sort_no_key(x.get("sort_no"))))
    for idx, item in enumerate(ordered):
        next_start = ordered[idx + 1].get("page_start", total_pages + 1) if idx + 1 < len(ordered) else total_pages + 1
        start_page = item.get("page_start", 1)
        end_page = max(start_page, next_start - 1)
        item["page_end"] = min(total_pages, end_page)


def _extract_raw_chapter_no(entry: Dict[str, Any]) -> Optional[str]:
    chapter_no = str(entry.get("chapter_no") or "").strip()
    if chapter_no:
        return chapter_no

    section_no = str(entry.get("section_no") or "").strip()
    if section_no and "." in section_no:
        return section_no.split(".")[0]

    subsection_no = str(entry.get("subsection_no") or "").strip()
    if subsection_no and "." in subsection_no:
        return subsection_no.split(".")[0]
    return None


def _extract_raw_section_no(entry: Dict[str, Any]) -> Optional[str]:
    section_no = str(entry.get("section_no") or "").strip()
    if section_no:
        return section_no

    subsection_no = str(entry.get("subsection_no") or "").strip()
    if subsection_no and subsection_no.count(".") >= 1:
        return ".".join(subsection_no.split(".")[:2])
    return None


def _normalize_structure_order(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        entries,
        key=lambda x: (
            to_int(x.get("page_start")) or 1,
            to_int(x.get("order")) or 0,
            normalize_name(x.get("title")),
        ),
    )


def normalize_structure_entries(entries: List[Dict[str, Any]], total_pages: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    dropped_entries_count = 0

    safe_total_pages = max(1, int(total_pages or 1))
    last_page = 1

    for idx, entry in enumerate(_normalize_structure_order(entries)):
        level = normalize_name(entry.get("level"))
        if level not in {"chapter", "section", "subsection"}:
            dropped_entries_count += 1
            continue

        title = re.sub(r"\s+", " ", str(entry.get("title") or "").strip())
        if not title:
            dropped_entries_count += 1
            continue

        page_start = to_int(entry.get("page_start"))
        if page_start is None:
            page_start = to_int(entry.get("printed_page"))
        if page_start is None:
            page_start = last_page

        page_start = max(1, min(safe_total_pages, int(page_start)))
        last_page = page_start

        raw_chapter_no = _extract_raw_chapter_no(entry)
        raw_section_no = _extract_raw_section_no(entry)
        raw_subsection_no = str(entry.get("subsection_no") or "").strip() or None

        cleaned.append(
            {
                "level": level,
                "title": title,
                "page_start": page_start,
                "order": to_int(entry.get("order")) if to_int(entry.get("order")) is not None else idx,
                "raw_chapter_no": raw_chapter_no,
                "raw_section_no": raw_section_no,
                "raw_subsection_no": raw_subsection_no,
            }
        )

    if not cleaned:
        fallback = [
            {
                "level": "chapter",
                "title": "Chapter 1",
                "page_start": 1,
                "order": 0,
                "raw_chapter_no": None,
                "raw_section_no": None,
                "raw_subsection_no": None,
                "chapter_no": "1",
            },
            {
                "level": "section",
                "title": "Section 1.1",
                "page_start": 1,
                "order": 1,
                "raw_chapter_no": None,
                "raw_section_no": None,
                "raw_subsection_no": None,
                "chapter_no": "1",
                "section_no": "1.1",
            },
        ]
        return fallback, {
            "structure_version": "v2_normalized",
            "dropped_entries_count": dropped_entries_count,
            "relabelled_chapters_count": 0,
            "relabelled_sections_count": 0,
            "orphan_sections_fixed": 1,
            "orphan_subsections_fixed": 0,
            "normalized_entry_count": len(fallback),
            "normalized_chapter_count": 1,
            "normalized_section_count": 1,
            "normalized_subsection_count": 0,
        }

    normalized_entries: List[Dict[str, Any]] = []
    relabelled_chapter_pairs = set()
    relabelled_section_pairs = set()
    orphan_sections_fixed = 0
    orphan_subsections_fixed = 0

    sectionish = [row for row in cleaned if row["level"] in {"section", "subsection"}]

    if sectionish:
        chapter_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in sectionish:
            raw_chapter = row.get("raw_chapter_no")
            if not raw_chapter:
                raw_chapter = f"__auto_ch_{row['page_start']}_{row['order']}"
                if row["level"] == "section":
                    orphan_sections_fixed += 1
                else:
                    orphan_subsections_fixed += 1
            chapter_groups[raw_chapter].append(row)

        chapter_rows = []
        for raw_chapter, group in chapter_groups.items():
            min_page = min(item["page_start"] for item in group)
            min_order = min(item.get("order", 0) for item in group)
            chapter_title = None
            chapter_candidates = [
                item
                for item in cleaned
                if item["level"] == "chapter" and str(item.get("raw_chapter_no") or "") == str(raw_chapter)
            ]
            if chapter_candidates:
                chapter_candidates = sorted(chapter_candidates, key=lambda x: (x["page_start"], x.get("order", 0)))
                chapter_title = chapter_candidates[0]["title"]
            if not chapter_title:
                chapter_title = f"Chapter {len(chapter_rows) + 1}"

            chapter_rows.append(
                {
                    "raw_chapter_no": None if str(raw_chapter).startswith("__auto_ch_") else str(raw_chapter),
                    "title": chapter_title,
                    "page_start": min_page,
                    "order": min_order,
                    "group": group,
                }
            )

        chapter_rows.sort(key=lambda x: (x["page_start"], x["order"], normalize_name(x["title"])))

        out_order = 0
        for chapter_index, chapter_row in enumerate(chapter_rows, start=1):
            canonical_chapter_no = str(chapter_index)
            raw_chapter_no = chapter_row.get("raw_chapter_no")
            if raw_chapter_no and raw_chapter_no != canonical_chapter_no:
                relabelled_chapter_pairs.add((raw_chapter_no, canonical_chapter_no))

            normalized_entries.append(
                {
                    "level": "chapter",
                    "chapter_no": canonical_chapter_no,
                    "section_no": None,
                    "subsection_no": None,
                    "raw_chapter_no": raw_chapter_no,
                    "raw_section_no": None,
                    "raw_subsection_no": None,
                    "title": chapter_row["title"],
                    "page_start": chapter_row["page_start"],
                    "order": out_order,
                }
            )
            out_order += 1

            section_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for row in chapter_row["group"]:
                section_key = row.get("raw_section_no")
                if not section_key:
                    section_key = f"__auto_sec_{canonical_chapter_no}_{row['page_start']}_{row['order']}"
                    if row["level"] == "section":
                        orphan_sections_fixed += 1
                    else:
                        orphan_subsections_fixed += 1
                section_groups[section_key].append(row)

            sorted_section_groups = sorted(
                section_groups.items(),
                key=lambda kv: (
                    min(item["page_start"] for item in kv[1]),
                    min(item.get("order", 0) for item in kv[1]),
                    kv[0],
                ),
            )

            for section_index, (raw_section_key, section_group) in enumerate(sorted_section_groups, start=1):
                canonical_section_no = f"{canonical_chapter_no}.{section_index}"
                raw_section_no = None if str(raw_section_key).startswith("__auto_sec_") else str(raw_section_key)
                if raw_section_no and raw_section_no != canonical_section_no:
                    relabelled_section_pairs.add((raw_section_no, canonical_section_no))

                section_title_candidates = [item for item in section_group if item["level"] == "section"]
                if section_title_candidates:
                    section_title = sorted(
                        section_title_candidates, key=lambda x: (x["page_start"], x.get("order", 0))
                    )[0]["title"]
                else:
                    section_title = f"Section {canonical_section_no}"

                section_page_start = min(item["page_start"] for item in section_group)

                normalized_entries.append(
                    {
                        "level": "section",
                        "chapter_no": canonical_chapter_no,
                        "section_no": canonical_section_no,
                        "subsection_no": None,
                        "raw_chapter_no": raw_chapter_no,
                        "raw_section_no": raw_section_no,
                        "raw_subsection_no": None,
                        "title": section_title,
                        "page_start": section_page_start,
                        "order": out_order,
                    }
                )
                out_order += 1

                subsection_rows = [item for item in section_group if item["level"] == "subsection"]
                subsection_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                for subsection in subsection_rows:
                    raw_sub_key = subsection.get("raw_subsection_no")
                    if not raw_sub_key:
                        raw_sub_key = f"__auto_sub_{canonical_section_no}_{subsection['page_start']}_{subsection['order']}"
                        orphan_subsections_fixed += 1
                    subsection_groups[raw_sub_key].append(subsection)

                sorted_subsection_groups = sorted(
                    subsection_groups.items(),
                    key=lambda kv: (
                        min(item["page_start"] for item in kv[1]),
                        min(item.get("order", 0) for item in kv[1]),
                        kv[0],
                    ),
                )

                for sub_index, (raw_sub_key, sub_group) in enumerate(sorted_subsection_groups, start=1):
                    canonical_subsection_no = f"{canonical_section_no}.{sub_index}"
                    raw_subsection_no = None if str(raw_sub_key).startswith("__auto_sub_") else str(raw_sub_key)
                    sub_title = sorted(sub_group, key=lambda x: (x["page_start"], x.get("order", 0)))[0]["title"]
                    sub_page_start = min(item["page_start"] for item in sub_group)

                    normalized_entries.append(
                        {
                            "level": "subsection",
                            "chapter_no": canonical_chapter_no,
                            "section_no": canonical_section_no,
                            "subsection_no": canonical_subsection_no,
                            "raw_chapter_no": raw_chapter_no,
                            "raw_section_no": raw_section_no,
                            "raw_subsection_no": raw_subsection_no,
                            "title": sub_title,
                            "page_start": sub_page_start,
                            "order": out_order,
                        }
                    )
                    out_order += 1

    else:
        chapter_candidates = [row for row in cleaned if row["level"] == "chapter"]
        if not chapter_candidates:
            chapter_candidates = [
                {
                    "title": "Chapter 1",
                    "page_start": 1,
                    "order": 0,
                    "raw_chapter_no": None,
                }
            ]

        chapter_candidates = sorted(
            chapter_candidates, key=lambda x: (x["page_start"], x.get("order", 0), normalize_name(x.get("title")))
        )
        deduped = []
        seen_keys = set()
        for row in chapter_candidates:
            key = (row["page_start"], normalize_name(row.get("title")))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(row)

        out_order = 0
        for chapter_index, row in enumerate(deduped, start=1):
            canonical_chapter_no = str(chapter_index)
            raw_chapter_no = row.get("raw_chapter_no")
            if raw_chapter_no and raw_chapter_no != canonical_chapter_no:
                relabelled_chapter_pairs.add((raw_chapter_no, canonical_chapter_no))

            normalized_entries.append(
                {
                    "level": "chapter",
                    "chapter_no": canonical_chapter_no,
                    "section_no": None,
                    "subsection_no": None,
                    "raw_chapter_no": raw_chapter_no,
                    "raw_section_no": None,
                    "raw_subsection_no": None,
                    "title": row.get("title") or f"Chapter {canonical_chapter_no}",
                    "page_start": row.get("page_start", 1),
                    "order": out_order,
                }
            )
            out_order += 1

            normalized_entries.append(
                {
                    "level": "section",
                    "chapter_no": canonical_chapter_no,
                    "section_no": f"{canonical_chapter_no}.1",
                    "subsection_no": None,
                    "raw_chapter_no": raw_chapter_no,
                    "raw_section_no": None,
                    "raw_subsection_no": None,
                    "title": f"Section {canonical_chapter_no}.1",
                    "page_start": row.get("page_start", 1),
                    "order": out_order,
                }
            )
            out_order += 1
            orphan_sections_fixed += 1

    normalized_entries = _normalize_structure_order(normalized_entries)
    for idx, row in enumerate(normalized_entries):
        row["order"] = idx

    chapter_count = sum(1 for row in normalized_entries if row["level"] == "chapter")
    section_count = sum(1 for row in normalized_entries if row["level"] == "section")
    subsection_count = sum(1 for row in normalized_entries if row["level"] == "subsection")

    diagnostics = {
        "structure_version": "v2_normalized",
        "dropped_entries_count": dropped_entries_count,
        "relabelled_chapters_count": len(relabelled_chapter_pairs),
        "relabelled_sections_count": len(relabelled_section_pairs),
        "orphan_sections_fixed": orphan_sections_fixed,
        "orphan_subsections_fixed": orphan_subsections_fixed,
        "normalized_entry_count": len(normalized_entries),
        "normalized_chapter_count": chapter_count,
        "normalized_section_count": section_count,
        "normalized_subsection_count": subsection_count,
    }
    return normalized_entries, diagnostics


def build_structure_from_entries(entries: List[Dict[str, Any]], total_pages: int) -> Dict[str, List[Dict[str, Any]]]:
    chapters_map: Dict[str, Dict[str, Any]] = {}
    sections_map: Dict[str, Dict[str, Any]] = {}
    subsections_map: Dict[str, Dict[str, Any]] = {}

    # First pass: chapters
    for entry in sorted(entries, key=lambda x: x.get("order", 0)):
        if entry.get("level") == "chapter":
            chapter_no = entry.get("chapter_no") or "1"
            chapters_map.setdefault(
                chapter_no,
                {
                    "chapter_no": chapter_no,
                    "title": entry.get("title") or f"Chapter {chapter_no}",
                    "raw_chapter_no": entry.get("raw_chapter_no"),
                    "page_start": entry.get("page_start", 1),
                    "sort_no": chapter_no,
                },
            )

    # Ensure chapters from section/subsection identifiers
    for entry in entries:
        chapter_no = entry.get("chapter_no")
        if chapter_no:
            chapters_map.setdefault(
                chapter_no,
                {
                    "chapter_no": chapter_no,
                    "title": f"Chapter {chapter_no}",
                    "raw_chapter_no": entry.get("raw_chapter_no"),
                    "page_start": entry.get("page_start", 1),
                    "sort_no": chapter_no,
                },
            )

    if not chapters_map:
        chapters_map["1"] = {
            "chapter_no": "1",
            "title": "Chapter 1",
            "raw_chapter_no": None,
            "page_start": 1,
            "sort_no": "1",
        }

    # Sections
    for entry in sorted(entries, key=lambda x: x.get("order", 0)):
        if entry.get("level") != "section":
            continue
        section_no = entry.get("section_no")
        if not section_no:
            continue
        chapter_no = entry.get("chapter_no") or section_no.split(".")[0]
        sections_map.setdefault(
            section_no,
            {
                "section_no": section_no,
                "title": entry.get("title") or f"Section {section_no}",
                "chapter_no": chapter_no,
                "raw_section_no": entry.get("raw_section_no"),
                "page_start": entry.get("page_start", chapters_map.get(chapter_no, {}).get("page_start", 1)),
                "sort_no": section_no,
            },
        )

    # Create fallback sections if none
    if not sections_map:
        for chapter_no, chapter in chapters_map.items():
            fallback_section_no = f"{chapter_no}.1"
            sections_map[fallback_section_no] = {
                "section_no": fallback_section_no,
                "title": f"Section {fallback_section_no}",
                "chapter_no": chapter_no,
                "raw_section_no": None,
                "page_start": chapter.get("page_start", 1),
                "sort_no": fallback_section_no,
            }

    # Subsections
    for entry in sorted(entries, key=lambda x: x.get("order", 0)):
        if entry.get("level") != "subsection":
            continue
        subsection_no = entry.get("subsection_no")
        if not subsection_no:
            continue
        section_no = entry.get("section_no") or ".".join(subsection_no.split(".")[:2])
        chapter_no = entry.get("chapter_no") or section_no.split(".")[0]

        # Ensure parent section exists
        if section_no not in sections_map:
            sections_map[section_no] = {
                "section_no": section_no,
                "title": f"Section {section_no}",
                "chapter_no": chapter_no,
                "raw_section_no": entry.get("raw_section_no"),
                "page_start": entry.get("page_start", chapters_map.get(chapter_no, {}).get("page_start", 1)),
                "sort_no": section_no,
            }

        subsections_map.setdefault(
            subsection_no,
            {
                "subsection_no": subsection_no,
                "title": entry.get("title") or f"Subsection {subsection_no}",
                "section_no": section_no,
                "chapter_no": chapter_no,
                "raw_subsection_no": entry.get("raw_subsection_no"),
                "page_start": entry.get("page_start", sections_map[section_no].get("page_start", 1)),
                "sort_no": subsection_no,
            },
        )

    chapters = sorted(chapters_map.values(), key=lambda x: (_sort_no_key(x["chapter_no"]), x.get("page_start", 1)))
    sections = sorted(sections_map.values(), key=lambda x: (_sort_no_key(x["section_no"]), x.get("page_start", 1)))
    subsections = sorted(subsections_map.values(), key=lambda x: (_sort_no_key(x["subsection_no"]), x.get("page_start", 1)))

    _assign_end_pages(chapters, total_pages)

    # section end pages within each chapter
    sections_by_chapter: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for section in sections:
        sections_by_chapter[section["chapter_no"]].append(section)

    for chapter in chapters:
        chapter_no = chapter["chapter_no"]
        chapter_sections = sections_by_chapter.get(chapter_no, [])
        chapter_sections.sort(key=lambda x: (x.get("page_start", chapter["page_start"]), _sort_no_key(x["section_no"])))

        for idx, section in enumerate(chapter_sections):
            start_page = max(chapter["page_start"], section.get("page_start", chapter["page_start"]))
            next_start = (
                chapter_sections[idx + 1].get("page_start", chapter["page_end"] + 1)
                if idx + 1 < len(chapter_sections)
                else chapter["page_end"] + 1
            )
            section["page_start"] = start_page
            section["page_end"] = min(chapter["page_end"], max(start_page, next_start - 1))

    # subsection end pages within each section
    subsections_by_section: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for subsection in subsections:
        subsections_by_section[subsection["section_no"]].append(subsection)

    section_lookup = {section["section_no"]: section for section in sections}
    for section_no, group in subsections_by_section.items():
        parent = section_lookup.get(section_no)
        if not parent:
            continue

        group.sort(key=lambda x: (x.get("page_start", parent["page_start"]), _sort_no_key(x["subsection_no"])))
        for idx, subsection in enumerate(group):
            start_page = max(parent["page_start"], subsection.get("page_start", parent["page_start"]))
            next_start = (
                group[idx + 1].get("page_start", parent["page_end"] + 1)
                if idx + 1 < len(group)
                else parent["page_end"] + 1
            )
            subsection["page_start"] = start_page
            subsection["page_end"] = min(parent["page_end"], max(start_page, next_start - 1))

    for chapter in chapters:
        chapter.pop("sort_no", None)
    for section in sections:
        section.pop("sort_no", None)
    for subsection in subsections:
        subsection.pop("sort_no", None)

    return {
        "chapters": chapters,
        "sections": sections,
        "subsections": subsections,
    }


def infer_structure_without_toc(page_text_map: Dict[int, str]) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    seen = set()
    order = 0

    chapter_patterns = [
        re.compile(r"(?i)^\s*(chapter|unit)\s+([0-9IVXLC]+)\s*[:\.-]?\s*(.+)?$"),
        re.compile(r"^\s*(\d+)\s+(.+)$"),
    ]
    section_pattern = re.compile(r"^\s*(\d+\.\d+)\s+(.+)$")
    subsection_pattern = re.compile(r"^\s*(\d+\.\d+\.\d+)\s+(.+)$")

    for page_no in sorted(page_text_map.keys()):
        lines = get_page_lines(page_text_map[page_no])[:12]
        for line in lines:
            line_clean = re.sub(r"\s+", " ", line).strip()
            if not line_clean:
                continue

            sm = subsection_pattern.match(line_clean)
            if sm:
                num = _normalize_number_token(sm.group(1))
                if num:
                    key = ("subsection", num)
                    if key not in seen:
                        seen.add(key)
                        entries.append(
                            {
                                "level": "subsection",
                                "chapter_no": num.split(".")[0],
                                "section_no": ".".join(num.split(".")[:2]),
                                "subsection_no": ".".join(num.split(".")[:3]),
                                "title": sm.group(2).strip(),
                                "printed_page": None,
                                "order": order,
                                "page_start": page_no,
                            }
                        )
                        order += 1
                continue

            sm = section_pattern.match(line_clean)
            if sm:
                num = _normalize_number_token(sm.group(1))
                if num:
                    key = ("section", num)
                    if key not in seen:
                        seen.add(key)
                        entries.append(
                            {
                                "level": "section",
                                "chapter_no": num.split(".")[0],
                                "section_no": ".".join(num.split(".")[:2]),
                                "subsection_no": None,
                                "title": sm.group(2).strip(),
                                "printed_page": None,
                                "order": order,
                                "page_start": page_no,
                            }
                        )
                        order += 1
                continue

            for pattern in chapter_patterns:
                cm = pattern.match(line_clean)
                if not cm:
                    continue

                if pattern is chapter_patterns[0]:
                    chapter_no = _chapter_no_from_raw(cm.group(2), len(entries) + 1)
                    title = (cm.group(3) or "").strip() or f"Chapter {chapter_no}"
                else:
                    num = _normalize_number_token(cm.group(1) or "")
                    if not num or "." in num:
                        continue
                    chapter_no = num
                    title = cm.group(2).strip()

                key = ("chapter", chapter_no)
                if key in seen:
                    break
                seen.add(key)
                entries.append(
                    {
                        "level": "chapter",
                        "chapter_no": chapter_no,
                        "section_no": None,
                        "subsection_no": None,
                        "title": title,
                        "printed_page": None,
                        "order": order,
                        "page_start": page_no,
                    }
                )
                order += 1
                break

    if not entries:
        entries = [
            {
                "level": "chapter",
                "chapter_no": "1",
                "section_no": None,
                "subsection_no": None,
                "title": "Chapter 1",
                "printed_page": None,
                "order": 0,
                "page_start": min(page_text_map.keys()) if page_text_map else 1,
            },
            {
                "level": "section",
                "chapter_no": "1",
                "section_no": "1.1",
                "subsection_no": None,
                "title": "Section 1.1",
                "printed_page": None,
                "order": 1,
                "page_start": min(page_text_map.keys()) if page_text_map else 1,
            },
        ]

    total_pages = max(page_text_map.keys()) if page_text_map else 1
    normalized_entries, normalization_diag = normalize_structure_entries(entries, total_pages)
    structure = build_structure_from_entries(normalized_entries, total_pages)

    return {
        "structure": structure,
        "diagnostics": {
            "mode": "fallback_heading_inference",
            "toc_pages": [],
            "toc_entry_count": len(entries),
            "fallback_entry_count": len(entries),
            "offset": None,
            "unresolved": 0,
            "structure_version": normalization_diag.get("structure_version", "v2_normalized"),
            "normalization": normalization_diag,
        },
    }


def build_structure_from_toc_or_fallback(
    pages: Dict[str, List[str]],
    model_id: str,
    selected_model: str,
    toc_scan_pages: int = 25,
) -> Dict[str, Any]:
    page_text_map = pages_to_text_map(pages)
    if not page_text_map:
        return {
            "structure": {"chapters": [], "sections": [], "subsections": []},
            "diagnostics": {"mode": "empty", "toc_pages": []},
        }

    toc_pages = detect_toc_pages(page_text_map, toc_scan_pages, model_id, selected_model)

    if not toc_pages:
        return infer_structure_without_toc(page_text_map)

    toc_lines: List[str] = []
    for page_no in toc_pages:
        toc_lines.extend(get_page_lines(page_text_map[page_no]))

    entries = parse_toc_entries_regex(toc_lines)

    # If regex parse is weak, use LLM parse and merge
    if len(entries) < 4:
        llm_entries = parse_toc_entries_llm("\n".join(toc_lines), model_id, selected_model)
        existing_keys = {
            (
                item.get("level"),
                item.get("chapter_no"),
                item.get("section_no"),
                item.get("subsection_no"),
                normalize_name(item.get("title")),
            )
            for item in entries
        }
        for item in llm_entries:
            key = (
                item.get("level"),
                item.get("chapter_no"),
                item.get("section_no"),
                item.get("subsection_no"),
                normalize_name(item.get("title")),
            )
            if key not in existing_keys:
                item["order"] = len(entries)
                entries.append(item)
                existing_keys.add(key)

    if not entries:
        return infer_structure_without_toc(page_text_map)

    entries, map_diag = assign_physical_pages_to_entries(entries, page_text_map, model_id, selected_model)
    total_pages = max(page_text_map.keys())
    normalized_entries, normalization_diag = normalize_structure_entries(entries, total_pages)
    structure = build_structure_from_entries(normalized_entries, total_pages)

    if not structure.get("sections"):
        return infer_structure_without_toc(page_text_map)

    diagnostics = {
        "mode": "toc_first",
        "toc_pages": toc_pages,
        "toc_entry_count": len(entries),
        "fallback_entry_count": 0,
        "offset": map_diag.get("offset"),
        "unresolved": map_diag.get("unresolved"),
        "structure_version": normalization_diag.get("structure_version", "v2_normalized"),
        "normalization": normalization_diag,
        "toc_entries_preview": entries[:25],
        "normalized_entries_preview": normalized_entries[:25],
    }

    return {"structure": structure, "diagnostics": diagnostics}


# ============================================================
# SEGMENT EXTRACTION
# ============================================================
def split_page_windows(start_page: int, end_page: int, pages_per_chunk: int, overlap: int) -> List[Tuple[int, int]]:
    windows: List[Tuple[int, int]] = []

    if start_page > end_page:
        return windows

    window_size = max(1, pages_per_chunk)
    step = max(1, window_size - max(0, overlap))

    current = start_page
    while current <= end_page:
        chunk_end = min(end_page, current + window_size - 1)
        windows.append((current, chunk_end))
        if chunk_end >= end_page:
            break
        current += step

    return windows


def build_extraction_segments(
    structure: Dict[str, List[Dict[str, Any]]],
    total_pages: int,
    pages_per_chunk: int,
    overlap: int,
) -> List[Dict[str, Any]]:
    sections = structure.get("sections", [])
    subsections = structure.get("subsections", [])

    subsection_by_section: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for subsection in subsections:
        subsection_by_section[subsection["section_no"]].append(subsection)

    base_segments: List[Dict[str, Any]] = []

    # Use subsections as primary extraction units
    for subsection in subsections:
        base_segments.append(
            {
                "anchor_type": "Subsection",
                "anchor_no": subsection["subsection_no"],
                "chapter_no": subsection["chapter_no"],
                "section_no": subsection["section_no"],
                "title": subsection["title"],
                "page_start": subsection["page_start"],
                "page_end": subsection["page_end"],
            }
        )

    # Add sections that have no subsection children
    for section in sections:
        if subsection_by_section.get(section["section_no"]):
            continue
        base_segments.append(
            {
                "anchor_type": "Section",
                "anchor_no": section["section_no"],
                "chapter_no": section["chapter_no"],
                "section_no": section["section_no"],
                "title": section["title"],
                "page_start": section["page_start"],
                "page_end": section["page_end"],
            }
        )

    if not base_segments:
        base_segments.append(
            {
                "anchor_type": "Section",
                "anchor_no": "1.1",
                "chapter_no": "1",
                "section_no": "1.1",
                "title": "Section 1.1",
                "page_start": 1,
                "page_end": total_pages,
            }
        )

    segments: List[Dict[str, Any]] = []
    for base in base_segments:
        windows = split_page_windows(base["page_start"], base["page_end"], pages_per_chunk, overlap)
        if not windows:
            windows = [(base["page_start"], base["page_end"])]

        for idx, (start_page, end_page) in enumerate(windows, start=1):
            segments.append(
                {
                    **base,
                    "segment_id": f"{base['anchor_type']}::{base['anchor_no']}::{idx}",
                    "chunk_index": idx,
                    "page_start": start_page,
                    "page_end": end_page,
                }
            )

    return segments


def build_segment_text(page_text_map: Dict[int, str], start_page: int, end_page: int) -> str:
    parts = []
    for page_no in range(start_page, end_page + 1):
        if page_no in page_text_map:
            parts.append(f"[Page {page_no}]\n{page_text_map[page_no]}")
    return "\n\n".join(parts)


def get_text_embedding(text: str) -> List[float]:
    payload = {"inputText": str(text or "")[:12000]}
    response = bedrock.invoke_model(modelId=EMBED_MODEL_ID, body=json.dumps(payload))
    data = json.loads(response["body"].read())
    embedding = data.get("embedding", [])
    return embedding if isinstance(embedding, list) else []


def ensure_questindex_vector_index(index_name: str, dimension: int) -> None:
    if opensearch is None:
        raise Exception("OpenSearch client unavailable. Set OS_DOMAIN and AWS auth.")

    if opensearch.indices.exists(index=index_name):
        return

    body = {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "book_id": {"type": "keyword"},
                "course_id": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "chunk_hash": {"type": "keyword"},
                "segment_id": {"type": "keyword"},
                "anchor_type": {"type": "keyword"},
                "anchor_no": {"type": "keyword"},
                "chapter_no": {"type": "keyword"},
                "section_no": {"type": "keyword"},
                "page_range": {"type": "keyword"},
                "page_start": {"type": "integer"},
                "page_end": {"type": "integer"},
                "chunk_text": {"type": "text"},
                "source": {"type": "keyword"},
                "created_at": {"type": "date"},
                "vector_field": {"type": "knn_vector", "dimension": int(dimension)},
            }
        },
    }
    opensearch.indices.create(index=index_name, body=body)


def index_questindex_segments_opensearch(
    segments: List[Dict[str, Any]],
    page_text_map: Dict[int, str],
    book_id: str,
    index_name: str,
) -> Dict[str, Any]:
    if opensearch is None:
        return {
            "indexed": 0,
            "errors": 0,
            "skipped": len(segments),
            "index_name": index_name,
            "error": "OpenSearch is not configured.",
        }

    if not segments:
        return {"indexed": 0, "errors": 0, "skipped": 0, "index_name": index_name}

    rows: List[Dict[str, Any]] = []
    embed_errors = 0
    progress_bar = st.progress(0.0)
    status = st.empty()
    total = len(segments)

    for idx, segment in enumerate(segments, start=1):
        status.text(f"Generating embeddings for segment {idx}/{total}")
        segment_text = build_segment_text(page_text_map, segment["page_start"], segment["page_end"])
        if not segment_text.strip():
            progress_bar.progress(min(1.0, idx / total))
            continue

        try:
            embedding = get_text_embedding(segment_text)
            if not embedding:
                embed_errors += 1
                progress_bar.progress(min(1.0, idx / total))
                continue

            chunk_id = make_chunk_id(book_id, segment_text)
            rows.append(
                {
                    "book_id": book_id,
                    "course_id": book_id,
                    "chunk_id": chunk_id,
                    "chunk_hash": make_chunk_hash(segment_text),
                    "segment_id": segment.get("segment_id"),
                    "anchor_type": segment.get("anchor_type"),
                    "anchor_no": segment.get("anchor_no"),
                    "chapter_no": segment.get("chapter_no"),
                    "section_no": segment.get("section_no"),
                    "page_range": f"{segment.get('page_start')}-{segment.get('page_end')}",
                    "page_start": segment.get("page_start"),
                    "page_end": segment.get("page_end"),
                    "chunk_text": segment_text,
                    "vector_field": embedding,
                    "source": "8_QuestIndex",
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
        except Exception:
            embed_errors += 1

        progress_bar.progress(min(1.0, idx / total))

    if not rows:
        status.text("Embedding generation failed for all segments")
        return {
            "indexed": 0,
            "errors": embed_errors,
            "skipped": total,
            "index_name": index_name,
            "error": "No embeddings generated.",
        }

    try:
        ensure_questindex_vector_index(index_name, len(rows[0]["vector_field"]))
    except Exception as exc:
        return {
            "indexed": 0,
            "errors": embed_errors,
            "skipped": total,
            "index_name": index_name,
            "error": f"Failed to prepare OpenSearch index: {exc}",
        }

    actions = []
    for row in rows:
        actions.append(
            {
                "_op_type": "index",
                "_index": index_name,
                "_id": row["chunk_id"],
                "_source": row,
            }
        )

    bulk_errors = 0
    indexed = 0
    try:
        for ok, _ in parallel_bulk(opensearch, actions, thread_count=1, chunk_size=50):
            if ok:
                indexed += 1
            else:
                bulk_errors += 1
    except Exception as exc:
        return {
            "indexed": indexed,
            "errors": embed_errors + bulk_errors + 1,
            "skipped": max(0, total - len(rows)),
            "index_name": index_name,
            "error": f"OpenSearch bulk indexing failed: {exc}",
        }

    # Ensure newly indexed documents are visible to immediate verification checks.
    try:
        opensearch.indices.refresh(index=index_name)
    except Exception:
        pass

    status.text("✅ OpenSearch vector indexing complete")
    return {
        "indexed": indexed,
        "errors": embed_errors + bulk_errors,
        "skipped": max(0, total - len(rows)),
        "index_name": index_name,
        "vector_dim": len(rows[0]["vector_field"]),
    }


def verify_questindex_embeddings(index_name: str, book_id: str, sample_size: int = 5) -> Dict[str, Any]:
    if opensearch is None:
        return {"verified": False, "error": "OpenSearch client unavailable."}

    try:
        source_value = "8_QuestIndex"
        book_id_text = str(book_id)

        def _book_clause() -> Dict[str, Any]:
            return {
                "bool": {
                    "should": [
                        {"term": {"book_id": book_id_text}},
                        {"term": {"book_id.keyword": book_id_text}},
                        {"match_phrase": {"book_id": book_id_text}},
                    ],
                    "minimum_should_match": 1,
                }
            }

        def _source_clause() -> Dict[str, Any]:
            return {
                "bool": {
                    "should": [
                        {"term": {"source": source_value}},
                        {"term": {"source.keyword": source_value}},
                        {"match_phrase": {"source": source_value}},
                    ],
                    "minimum_should_match": 1,
                }
            }

        # Keep verification deterministic right after bulk indexing.
        index_refreshed = False
        try:
            opensearch.indices.refresh(index=index_name)
            index_refreshed = True
        except Exception:
            index_refreshed = False

        exact_filter_query = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"book_id": book_id_text}},
                        {"term": {"source": source_value}},
                    ]
                }
            }
        }
        tolerant_filter_query = {
            "query": {
                "bool": {
                    "must": [
                        _book_clause(),
                        _source_clause(),
                    ]
                }
            }
        }

        total_docs = opensearch.count(index=index_name, body=exact_filter_query).get("count", 0)
        verification_mode = "exact_term"
        if total_docs == 0:
            total_docs = opensearch.count(index=index_name, body=tolerant_filter_query).get("count", 0)
            verification_mode = "tolerant_match"

        total_docs_book_only = opensearch.count(
            index=index_name,
            body={"query": _book_clause()},
        ).get("count", 0)
        total_docs_source_any_book = opensearch.count(
            index=index_name,
            body={"query": _source_clause()},
        ).get("count", 0)

        indexed_book_ids_for_source: List[str] = []
        try:
            agg_resp = opensearch.search(
                index=index_name,
                body={
                    "size": 0,
                    "query": _source_clause(),
                    "aggs": {"books": {"terms": {"field": "book_id", "size": 20}}},
                },
            )
            buckets = agg_resp.get("aggregations", {}).get("books", {}).get("buckets", [])
            indexed_book_ids_for_source = [str(bucket.get("key")) for bucket in buckets if bucket.get("key") is not None]
        except Exception:
            indexed_book_ids_for_source = []

        active_query = exact_filter_query["query"] if verification_mode == "exact_term" else tolerant_filter_query["query"]
        sample_query = {
            "size": max(1, sample_size),
            "_source": ["chunk_id", "vector_field", "page_range", "book_id", "segment_id"],
            "query": active_query,
        }
        sample_hits = opensearch.search(index=index_name, body=sample_query).get("hits", {}).get("hits", [])

        sampled = len(sample_hits)
        vectors_present = 0
        vector_dim = None
        probe_vector = None
        sample_chunk_ids: List[str] = []

        for hit in sample_hits:
            src = hit.get("_source", {})
            sample_chunk_ids.append(str(src.get("chunk_id") or hit.get("_id") or ""))
            vector = src.get("vector_field")
            if isinstance(vector, list) and len(vector) > 0:
                vectors_present += 1
                if vector_dim is None:
                    vector_dim = len(vector)
                    probe_vector = vector

        knn_probe_hits = 0
        knn_probe_error = None
        if probe_vector:
            try:
                knn_query = {
                    "size": 3,
                    "query": {
                        "bool": {
                            "filter": [_book_clause()],
                            "must": {"knn": {"vector_field": {"vector": probe_vector, "k": 3}}},
                        }
                    },
                }
                knn_probe_hits = len(
                    opensearch.search(index=index_name, body=knn_query).get("hits", {}).get("hits", [])
                )
            except Exception as exc:
                knn_probe_error = str(exc)

        kg_chunk_id_matches = 0
        if sample_chunk_ids:
            try:
                with neo4j.session() as neo_session:
                    kg_chunk_id_matches = int(
                        neo_session.run(
                            """
                            MATCH (n {book_id: $book_id})
                            WHERE n.chunk_id IN $chunk_ids
                            RETURN count(n) AS matched
                            """,
                            book_id=book_id,
                            chunk_ids=sample_chunk_ids,
                        ).single()["matched"]
                    )
            except Exception:
                kg_chunk_id_matches = 0

        return {
            "verified": vectors_present > 0,
            "verified_book_id": book_id_text,
            "verified_source": source_value,
            "total_docs_for_book": total_docs,
            "total_docs_book_any_source": total_docs_book_only,
            "total_docs_source_any_book": total_docs_source_any_book,
            "indexed_book_ids_for_source": indexed_book_ids_for_source,
            "sampled_docs": sampled,
            "sampled_with_vector": vectors_present,
            "vector_dim": vector_dim,
            "sample_chunk_ids": sample_chunk_ids,
            "kg_chunk_id_matches": kg_chunk_id_matches,
            "knn_probe_hits": knn_probe_hits,
            "knn_probe_error": knn_probe_error,
            "verification_mode": verification_mode,
            "index_refreshed": index_refreshed,
        }
    except Exception as exc:
        return {"verified": False, "error": str(exc)}


def extract_entities_for_segment(
    segment: Dict[str, Any],
    page_text_map: Dict[int, str],
    book_id: str,
    model_id: str,
    selected_model: str,
) -> Dict[str, Any]:
    segment_text = build_segment_text(page_text_map, segment["page_start"], segment["page_end"])
    segment_record = dict(segment)
    segment_record["chunk_id"] = make_chunk_id(book_id, segment_text)
    segment_record["chunk_hash"] = make_chunk_hash(segment_text)
    if not segment_text.strip():
        result = {key: [] for key in ENTITY_SPECS.keys()}
        result["relationships"] = []
        return {"segment": segment_record, "extracted": result}

    prompt = (
        f"{ENTITY_EXTRACTION_PROMPT}\n\n"
        f"Anchor Type: {segment['anchor_type']}\n"
        f"Anchor No: {segment['anchor_no']}\n"
        f"Chapter: {segment['chapter_no']}\n"
        f"Section: {segment['section_no']}\n"
        f"Pages: {segment['page_start']}-{segment['page_end']}\n\n"
        f"Segment Text:\n{segment_text[:26000]}"
    )

    payload = invoke_model_json(model_id, selected_model, prompt, max_output_tokens=3500)
    if not isinstance(payload, dict):
        payload = {}

    normalized: Dict[str, Any] = {}
    for key in ENTITY_SPECS.keys():
        value = payload.get(key, [])
        normalized[key] = value if isinstance(value, list) else []

    rels = payload.get("relationships", [])
    normalized["relationships"] = rels if isinstance(rels, list) else []

    return {"segment": segment_record, "extracted": normalized}


def process_segments_concurrently(
    segments: List[Dict[str, Any]],
    page_text_map: Dict[int, str],
    book_id: str,
    model_id: str,
    selected_model: str,
    max_workers: int,
) -> List[Dict[str, Any]]:
    if not segments:
        return []

    results: List[Dict[str, Any]] = []
    progress_bar = st.progress(0.0)
    status = st.empty()

    completed = 0
    total = len(segments)

    def update_progress() -> None:
        nonlocal completed
        with progress_lock:
            completed += 1
            progress_bar.progress(min(1.0, completed / total))
            status.text(f"Extracting entities from segment {completed}/{total}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_entities_for_segment, segment, page_text_map, book_id, model_id, selected_model): segment
            for segment in segments
        }

        for future in as_completed(futures):
            segment = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                st.warning(f"Segment extraction failed for {segment.get('segment_id')}: {exc}")
            finally:
                update_progress()

    status.text("Entity extraction complete")
    return results


# ============================================================
# EXTRACTION CONSOLIDATION
# ============================================================
def pick_description(item: Dict[str, Any], key: str) -> str:
    spec = ENTITY_SPECS.get(key, {})
    for field in spec.get("description_fields", ["description"]):
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_content_lookup(content_by_label: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, set]]:
    by_label: Dict[str, Dict[str, set]] = {label: {} for label in ALL_CONTENT_LABELS}
    by_any: Dict[str, set] = {}

    for label, nodes in content_by_label.items():
        for node in nodes:
            node_name = normalize_name(node.get("name"))
            node_id = node.get("node_id")
            if not node_name or not node_id:
                continue

            by_label.setdefault(label, {}).setdefault(node_name, set()).add(node_id)
            by_any.setdefault(node_name, set()).add(node_id)

    return {"by_label": by_label, "by_any": by_any}


def resolve_node_id(name: str, label: Optional[str], lookup: Dict[str, Dict[str, set]]) -> Optional[str]:
    normalized_name = normalize_name(name)
    if not normalized_name:
        return None

    if label:
        ids = lookup["by_label"].get(label, {}).get(normalized_name, set())
    else:
        ids = lookup["by_any"].get(normalized_name, set())

    if len(ids) == 1:
        return next(iter(ids))
    return None


def normalize_relationship_candidate(
    raw: Any,
    default_source: Optional[str] = None,
    default_source_label: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if isinstance(raw, str):
        if not raw.strip() or not default_source:
            return None
        return {
            "source": default_source,
            "source_type": default_source_label,
            "target": raw.strip(),
            "target_type": None,
            "type": "RELATED_TO",
        }

    if not isinstance(raw, dict):
        return None

    source = raw.get("source") or raw.get("from") or default_source
    target = raw.get("target") or raw.get("to")
    rel_type = raw.get("type") or raw.get("relation") or raw.get("predicate") or "RELATED_TO"

    if isinstance(source, dict):
        source_name = source.get("name") or source.get("title")
        source_label = normalize_label(source.get("type") or source.get("label"))
    else:
        source_name = source
        source_label = normalize_label(raw.get("source_type") or raw.get("source_label") or default_source_label)

    if isinstance(target, dict):
        target_name = target.get("name") or target.get("title")
        target_label = normalize_label(target.get("type") or target.get("label"))
    else:
        target_name = target
        target_label = normalize_label(raw.get("target_type") or raw.get("target_label"))

    if not source_name or not target_name:
        return None

    return {
        "source": str(source_name),
        "source_type": source_label,
        "target": str(target_name),
        "target_type": target_label,
        "type": str(rel_type),
    }


def consolidate_extraction_results(
    extraction_results: List[Dict[str, Any]],
    book_id: str,
) -> Dict[str, Any]:
    content_node_map: Dict[str, Dict[str, Dict[str, Any]]] = {label: {} for label in ALL_CONTENT_LABELS}
    relationship_candidates: List[Dict[str, Any]] = []

    for result in extraction_results:
        segment = result.get("segment", {})
        extracted = result.get("extracted", {})

        for key, spec in ENTITY_SPECS.items():
            label = spec["label"]
            nodes = extracted.get(key, [])
            if not isinstance(nodes, list):
                continue

            for node in nodes:
                if not isinstance(node, dict):
                    continue

                name = str(node.get("name") or "").strip()
                if not name:
                    continue

                node_id = make_content_node_id(book_id, label, name)
                description = pick_description(node, key)

                base_record = {
                    "node_id": node_id,
                    "name": name,
                    "book_id": book_id,
                    "description": description,
                    "anchor_type": segment.get("anchor_type"),
                    "anchor_no": segment.get("anchor_no"),
                    "chapter_no": segment.get("chapter_no"),
                    "section_no": segment.get("section_no"),
                    "page_start": segment.get("page_start"),
                    "page_end": segment.get("page_end"),
                    "chunk_id": segment.get("chunk_id") or segment.get("segment_id"),
                    "statement": node.get("statement"),
                    "expression": node.get("expression"),
                    "steps": node.get("steps"),
                    "question_number": node.get("question_number"),
                }

                existing = content_node_map[label].get(node_id)
                if existing:
                    if len(description) > len(existing.get("description", "")):
                        existing["description"] = description
                    for opt_field in ["statement", "expression", "steps", "question_number"]:
                        if not existing.get(opt_field) and base_record.get(opt_field):
                            existing[opt_field] = base_record.get(opt_field)
                    existing["page_start"] = min(existing.get("page_start") or base_record["page_start"], base_record["page_start"])
                    existing["page_end"] = max(existing.get("page_end") or base_record["page_end"], base_record["page_end"])
                else:
                    content_node_map[label][node_id] = base_record

                node_relationships = node.get("relationships", [])
                if isinstance(node_relationships, list):
                    for rel in node_relationships:
                        normalized = normalize_relationship_candidate(rel, default_source=name, default_source_label=label)
                        if normalized:
                            relationship_candidates.append(normalized)

        top_level_rels = extracted.get("relationships", [])
        if isinstance(top_level_rels, list):
            for rel in top_level_rels:
                normalized = normalize_relationship_candidate(rel)
                if normalized:
                    relationship_candidates.append(normalized)

    content_by_label = {label: list(nodes.values()) for label, nodes in content_node_map.items()}

    lookup = build_content_lookup(content_by_label)

    semantic_relationships: List[Dict[str, Any]] = []
    rel_seen = set()
    skipped = 0

    for rel in relationship_candidates:
        source_label = normalize_label(rel.get("source_type"))
        target_label = normalize_label(rel.get("target_type"))

        source_id = resolve_node_id(rel.get("source", ""), source_label, lookup)
        target_id = resolve_node_id(rel.get("target", ""), target_label, lookup)

        if not source_id or not target_id:
            skipped += 1
            continue

        normalized_type = normalize_rel_type(rel.get("type")) or "RELATED_TO"
        if normalized_type in ALLOWED_SEMANTIC_REL_TYPES:
            edge_type = normalized_type
            rel_type_raw = None
        else:
            edge_type = "RELATED_TO"
            rel_type_raw = normalized_type

        dedup_key = (source_id, target_id, edge_type, rel_type_raw or "")
        if dedup_key in rel_seen:
            continue
        rel_seen.add(dedup_key)

        semantic_relationships.append(
            {
                "source_node_id": source_id,
                "target_node_id": target_id,
                "edge_type": edge_type,
                "rel_type_raw": rel_type_raw,
            }
        )

    mention_count = sum(len(nodes) for nodes in content_by_label.values())

    return {
        "content_by_label": content_by_label,
        "semantic_relationships": semantic_relationships,
        "relationship_report": {
            "relationship_candidates": len(relationship_candidates),
            "semantic_insertable": len(semantic_relationships),
            "relationships_skipped": skipped,
            "mentions_expected": mention_count,
        },
    }


# ============================================================
# SUMMARIES
# ============================================================
def summarize_text_with_model(
    text: str,
    context_hint: str,
    model_id: str,
    selected_model: str,
) -> str:
    prompt = (
        f"{SUMMARY_PROMPT}\n\n"
        f"Context: {context_hint}\n\n"
        f"Text:\n{text[:18000]}"
    )

    try:
        response = invoke_model_text(model_id, selected_model, prompt, max_output_tokens=400)
        cleaned = response.strip()
        if cleaned:
            return cleaned
    except Exception:
        pass

    fallback = re.sub(r"\s+", " ", text[:1200]).strip()
    return fallback[:800] if fallback else "Summary unavailable."


def generate_summaries(
    structure: Dict[str, List[Dict[str, Any]]],
    consolidated: Dict[str, Any],
    page_text_map: Dict[int, str],
    book_id: str,
    model_id: str,
    selected_model: str,
    skip_section_summaries: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    section_summaries: List[Dict[str, Any]] = []
    chapter_summaries: List[Dict[str, Any]] = []
    section_count = 0 if skip_section_summaries else len(structure.get("sections", []))
    chapter_count = len(structure.get("chapters", []))
    total_summaries = section_count + chapter_count
    progress_bar = st.progress(0.0) if total_summaries > 0 else None
    status = st.empty() if total_summaries > 0 else None
    completed = 0

    def _advance(message: str) -> None:
        nonlocal completed
        if progress_bar is None or status is None:
            return
        completed += 1
        progress_bar.progress(min(1.0, completed / total_summaries))
        status.text(message)

    content_by_label = consolidated.get("content_by_label", {})
    all_nodes: List[Dict[str, Any]] = []
    for nodes in content_by_label.values():
        all_nodes.extend(nodes)

    section_entities: Dict[str, List[str]] = defaultdict(list)
    for node in all_nodes:
        anchor_type = node.get("anchor_type")
        anchor_no = node.get("anchor_no")
        if not anchor_no:
            continue
        if anchor_type == "Section":
            section_entities[anchor_no].append(node.get("name", ""))
        elif anchor_type == "Subsection":
            section_no = node.get("section_no")
            if section_no:
                section_entities[section_no].append(node.get("name", ""))

    # Section summaries
    if not skip_section_summaries:
        for section in structure.get("sections", []):
            text = build_segment_text(page_text_map, section["page_start"], section["page_end"])
            entities = sorted({name for name in section_entities.get(section["section_no"], []) if name})
            context_hint = (
                f"Section {section['section_no']} - {section['title']}"
                + (f" | Entities: {', '.join(entities[:20])}" if entities else "")
            )
            summary_text = summarize_text_with_model(text, context_hint, model_id, selected_model)

            section_summaries.append(
                {
                    "summary_id": make_id(f"{book_id}|section|{section['section_no']}", prefix="summary|"),
                    "book_id": book_id,
                    "level": "section",
                    "owner_no": section["section_no"],
                    "text": summary_text,
                    "page_start": section["page_start"],
                    "page_end": section["page_end"],
                    "model_id": model_id,
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            _advance(
                f"Generating summaries {completed + 1}/{total_summaries} "
                f"(section {section['section_no']})"
            )

    # Chapter summaries
    sections_by_chapter: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for section in structure.get("sections", []):
        sections_by_chapter[section["chapter_no"]].append(section)

    for chapter in structure.get("chapters", []):
        chapter_sections = sections_by_chapter.get(chapter["chapter_no"], [])
        section_summary_texts = [
            f"Section {s['owner_no']}: {s['text']}"
            for s in section_summaries
            if s["owner_no"] in {sec["section_no"] for sec in chapter_sections}
        ]

        if section_summary_texts:
            chapter_context = "\n\n".join(section_summary_texts)
        else:
            chapter_context = build_segment_text(page_text_map, chapter["page_start"], chapter["page_end"])

        context_hint = f"Chapter {chapter['chapter_no']} - {chapter['title']}"
        chapter_summary_text = summarize_text_with_model(chapter_context, context_hint, model_id, selected_model)

        chapter_summaries.append(
            {
                "summary_id": make_id(f"{book_id}|chapter|{chapter['chapter_no']}", prefix="summary|"),
                "book_id": book_id,
                "level": "chapter",
                "owner_no": chapter["chapter_no"],
                "text": chapter_summary_text,
                "page_start": chapter["page_start"],
                "page_end": chapter["page_end"],
                "model_id": model_id,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        _advance(
            f"Generating summaries {completed + 1}/{total_summaries} "
            f"(chapter {chapter['chapter_no']})"
        )

    if status is not None:
        status.text("✅ Summary extraction complete")

    return {"chapter_summaries": chapter_summaries, "section_summaries": section_summaries}


# ============================================================
# NEO4J WRITE
# ============================================================
def setup_constraints() -> None:
    constraint_queries = [
        "CREATE CONSTRAINT book_id_unique IF NOT EXISTS FOR (b:Book) REQUIRE b.book_id IS UNIQUE",
        "CREATE CONSTRAINT chapter_unique IF NOT EXISTS FOR (c:Chapter) REQUIRE (c.book_id, c.chapter_no) IS UNIQUE",
        "CREATE CONSTRAINT section_unique IF NOT EXISTS FOR (s:Section) REQUIRE (s.book_id, s.section_no) IS UNIQUE",
        "CREATE CONSTRAINT subsection_unique IF NOT EXISTS FOR (ss:Subsection) REQUIRE (ss.book_id, ss.subsection_no) IS UNIQUE",
        "CREATE CONSTRAINT summary_unique IF NOT EXISTS FOR (sm:Summary) REQUIRE sm.summary_id IS UNIQUE",
    ]

    for label in ALL_CONTENT_LABELS:
        constraint_name = f"{label.lower()}_unique"
        constraint_queries.append(
            f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS FOR (n:{label}) REQUIRE n.node_id IS UNIQUE"
        )

    with neo4j.session() as session:
        for query in constraint_queries:
            try:
                session.run(query)
            except Exception:
                # Existing conflicting constraints can be safely ignored.
                pass


def delete_book_subgraph(book_id: str) -> int:
    with neo4j.session() as session:
        total = session.run("MATCH (n {book_id: $book_id}) RETURN count(n) AS total", book_id=book_id).single()["total"]
        session.run("MATCH (n {book_id: $book_id}) DETACH DELETE n", book_id=book_id)
    return int(total)


def insert_questindex_graph(
    book_id: str,
    structure: Dict[str, List[Dict[str, Any]]],
    consolidated: Dict[str, Any],
    summaries: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    removed = delete_book_subgraph(book_id)

    chapters = structure.get("chapters", [])
    sections = structure.get("sections", [])
    subsections = structure.get("subsections", [])

    content_by_label = consolidated.get("content_by_label", {})
    semantic_relationships = consolidated.get("semantic_relationships", [])

    chapter_summaries = summaries.get("chapter_summaries", [])
    section_summaries = summaries.get("section_summaries", [])

    with neo4j.session() as session:
        session.execute_write(
            _insert_graph_tx,
            book_id,
            chapters,
            sections,
            subsections,
            content_by_label,
            semantic_relationships,
            chapter_summaries,
            section_summaries,
        )

    stats = {
        "replaced_nodes": removed,
        "structure_version": "v2_normalized",
        "chapters": len(chapters),
        "sections": len(sections),
        "subsections": len(subsections),
        "semantic_relationships": len(semantic_relationships),
        "chapter_summaries": len(chapter_summaries),
        "section_summaries": len(section_summaries),
    }

    for label in ALL_CONTENT_LABELS:
        stats[label] = len(content_by_label.get(label, []))

    stats.update(consolidated.get("relationship_report", {}))
    return stats


def _insert_graph_tx(
    tx,
    book_id: str,
    chapters: List[Dict[str, Any]],
    sections: List[Dict[str, Any]],
    subsections: List[Dict[str, Any]],
    content_by_label: Dict[str, List[Dict[str, Any]]],
    semantic_relationships: List[Dict[str, Any]],
    chapter_summaries: List[Dict[str, Any]],
    section_summaries: List[Dict[str, Any]],
) -> None:
    tx.run(
        """
        MERGE (b:Book {book_id: $book_id})
        SET b.structure_version = 'v2_normalized'
        """,
        book_id=book_id,
    )

    tx.run(
        """
        UNWIND $rows AS row
        MERGE (c:Chapter {book_id: $book_id, chapter_no: row.chapter_no})
        SET c.title = row.title,
            c.raw_chapter_no = row.raw_chapter_no,
            c.page_start = row.page_start,
            c.page_end = row.page_end
        WITH c
        MATCH (b:Book {book_id: $book_id})
        MERGE (b)-[:HAS_CHAPTER]->(c)
        """,
        book_id=book_id,
        rows=chapters,
    )

    tx.run(
        """
        UNWIND $rows AS row
        MERGE (s:Section {book_id: $book_id, section_no: row.section_no})
        SET s.title = row.title,
            s.chapter_no = row.chapter_no,
            s.raw_section_no = row.raw_section_no,
            s.page_start = row.page_start,
            s.page_end = row.page_end
        WITH s, row
        MATCH (c:Chapter {book_id: $book_id, chapter_no: row.chapter_no})
        MERGE (c)-[:HAS_SECTION]->(s)
        """,
        book_id=book_id,
        rows=sections,
    )

    tx.run(
        """
        UNWIND $rows AS row
        MERGE (ss:Subsection {book_id: $book_id, subsection_no: row.subsection_no})
        SET ss.title = row.title,
            ss.section_no = row.section_no,
            ss.chapter_no = row.chapter_no,
            ss.raw_subsection_no = row.raw_subsection_no,
            ss.page_start = row.page_start,
            ss.page_end = row.page_end
        WITH ss, row
        MATCH (s:Section {book_id: $book_id, section_no: row.section_no})
        MERGE (s)-[:HAS_SUBSECTION]->(ss)
        """,
        book_id=book_id,
        rows=subsections,
    )

    # Content nodes + mentions
    for label, rows in content_by_label.items():
        if not rows:
            continue

        tx.run(
            f"""
            UNWIND $rows AS row
            MERGE (n:{label} {{node_id: row.node_id}})
            SET n.name = row.name,
                n.book_id = $book_id,
                n.description = row.description,
                n.anchor_type = row.anchor_type,
                n.anchor_no = row.anchor_no,
                n.chapter_no = row.chapter_no,
                n.section_no = row.section_no,
                n.page_start = row.page_start,
                n.page_end = row.page_end,
                n.chunk_id = row.chunk_id,
                n.statement = row.statement,
                n.expression = row.expression,
                n.steps = row.steps,
                n.question_number = row.question_number
            """,
            book_id=book_id,
            rows=rows,
        )

        tx.run(
            f"""
            UNWIND $rows AS row
            WITH row
            WHERE row.anchor_type = 'Section'
            MATCH (s:Section {{book_id: $book_id, section_no: row.anchor_no}})
            MATCH (n:{label} {{node_id: row.node_id}})
            MERGE (s)-[:MENTIONS]->(n)
            """,
            book_id=book_id,
            rows=rows,
        )

        tx.run(
            f"""
            UNWIND $rows AS row
            WITH row
            WHERE row.anchor_type = 'Subsection'
            MATCH (ss:Subsection {{book_id: $book_id, subsection_no: row.anchor_no}})
            MATCH (n:{label} {{node_id: row.node_id}})
            MERGE (ss)-[:MENTIONS]->(n)
            """,
            book_id=book_id,
            rows=rows,
        )

    # Semantic relationships
    grouped_edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rel in semantic_relationships:
        grouped_edges[rel["edge_type"]].append(rel)

    for edge_type, rows in grouped_edges.items():
        if edge_type == "RELATED_TO":
            tx.run(
                """
                UNWIND $rows AS row
                MATCH (source {node_id: row.source_node_id})
                MATCH (target {node_id: row.target_node_id})
                MERGE (source)-[:RELATED_TO {rel_type_raw: row.rel_type_raw}]->(target)
                """,
                rows=rows,
            )
            continue

        tx.run(
            f"""
            UNWIND $rows AS row
            MATCH (source {{node_id: row.source_node_id}})
            MATCH (target {{node_id: row.target_node_id}})
            MERGE (source)-[:{edge_type}]->(target)
            """,
            rows=rows,
        )

    # Summaries
    if chapter_summaries:
        tx.run(
            """
            UNWIND $rows AS row
            MERGE (sm:Summary {summary_id: row.summary_id})
            SET sm.book_id = $book_id,
                sm.level = row.level,
                sm.owner_no = row.owner_no,
                sm.text = row.text,
                sm.page_start = row.page_start,
                sm.page_end = row.page_end,
                sm.model_id = row.model_id,
                sm.generated_at = row.generated_at
            WITH sm, row
            MATCH (c:Chapter {book_id: $book_id, chapter_no: row.owner_no})
            MERGE (c)-[:HAS_SUMMARY]->(sm)
            """,
            book_id=book_id,
            rows=chapter_summaries,
        )

    if section_summaries:
        tx.run(
            """
            UNWIND $rows AS row
            MERGE (sm:Summary {summary_id: row.summary_id})
            SET sm.book_id = $book_id,
                sm.level = row.level,
                sm.owner_no = row.owner_no,
                sm.text = row.text,
                sm.page_start = row.page_start,
                sm.page_end = row.page_end,
                sm.model_id = row.model_id,
                sm.generated_at = row.generated_at
            WITH sm, row
            MATCH (s:Section {book_id: $book_id, section_no: row.owner_no})
            MERGE (s)-[:HAS_SUMMARY]->(sm)
            """,
            book_id=book_id,
            rows=section_summaries,
        )


# ============================================================
# GRAPH QUERY + GUARDED CYPHER
# ============================================================
def get_books() -> List[str]:
    query = "MATCH (b:Book) RETURN b.book_id AS book ORDER BY book"
    with neo4j.session() as session:
        return [record["book"] for record in session.run(query)]


def get_chapters(book_id: str) -> List[Dict[str, Any]]:
    query = """
    MATCH (b:Book {book_id: $book_id})-[:HAS_CHAPTER]->(c:Chapter)
    RETURN c.chapter_no AS chapter_no, c.title AS title
    ORDER BY toInteger(c.chapter_no), c.chapter_no
    """
    with neo4j.session() as session:
        return [dict(record) for record in session.run(query, book_id=book_id)]


def get_sections(book_id: str, chapter_no: Optional[str] = None) -> List[Dict[str, Any]]:
    with neo4j.session() as session:
        if chapter_no:
            query = """
            MATCH (c:Chapter {book_id: $book_id, chapter_no: $chapter_no})-[:HAS_SECTION]->(s:Section)
            RETURN s.section_no AS section_no, s.title AS title
            ORDER BY toInteger(split(s.section_no, '.')[0]), toInteger(split(s.section_no, '.')[1]), s.section_no
            """
            return [dict(record) for record in session.run(query, book_id=book_id, chapter_no=chapter_no)]

        query = """
        MATCH (:Book {book_id: $book_id})-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s:Section)
        RETURN DISTINCT s.section_no AS section_no, s.title AS title
        ORDER BY toInteger(split(s.section_no, '.')[0]), toInteger(split(s.section_no, '.')[1]), s.section_no
        """
        return [dict(record) for record in session.run(query, book_id=book_id)]


def get_book_structure_version(book_id: str) -> str:
    with neo4j.session() as session:
        row = session.run(
            """
            MATCH (b:Book {book_id: $book_id})
            RETURN coalesce(b['structure_version'], 'legacy_unknown') AS structure_version
            """,
            book_id=book_id,
        ).single()
    if not row:
        return "unknown"
    return str(row.get("structure_version") or "legacy_unknown")


def get_chapter_section_catalog(book_id: str) -> Dict[str, List[Dict[str, Any]]]:
    with neo4j.session() as session:
        chapter_rows = session.run(
            """
            MATCH (:Book {book_id: $book_id})-[:HAS_CHAPTER]->(c:Chapter)
            OPTIONAL MATCH (c)-[:HAS_SECTION]->(s:Section)
            OPTIONAL MATCH (c)-[:HAS_SUMMARY]->(sm:Summary {level: 'chapter'})
            RETURN c.chapter_no AS chapter_no,
                   c['raw_chapter_no'] AS raw_chapter_no,
                   c.title AS title,
                   c.page_start AS page_start,
                   c.page_end AS page_end,
                   count(DISTINCT s) AS section_count,
                   count(DISTINCT s) > 0 AS has_sections,
                   toInteger(c.chapter_no) AS chapter_order,
                   sm.text AS summary_text
            ORDER BY chapter_order, c.chapter_no
            """,
            book_id=book_id,
        ).data()

        section_rows = session.run(
            """
            MATCH (:Book {book_id: $book_id})-[:HAS_CHAPTER]->(c:Chapter)-[:HAS_SECTION]->(s:Section)
            OPTIONAL MATCH (s)-[:HAS_SUMMARY]->(sm:Summary {level: 'section'})
            RETURN s.section_no AS section_no,
                   s['raw_section_no'] AS raw_section_no,
                   s.title AS title,
                   s.chapter_no AS chapter_no,
                   s.page_start AS page_start,
                   s.page_end AS page_end,
                   toInteger(split(s.section_no, '.')[0]) AS section_chapter_order,
                   toInteger(split(s.section_no, '.')[1]) AS section_order,
                   sm.text AS summary_text
            ORDER BY section_chapter_order, section_order, s.section_no
            """,
            book_id=book_id,
        ).data()

    return {
        "chapters": [dict(row) for row in chapter_rows],
        "sections": [dict(row) for row in section_rows],
    }


def _extract_kg_terms(query: str) -> List[str]:
    terms = []
    for token in re.findall(r"[A-Za-z0-9_]+", str(query or "").lower()):
        cleaned = normalize_name(token)
        if cleaned and len(cleaned) >= 3:
            terms.append(cleaned)
    return sorted(set(terms))


def _safe_scope_ids(values: Optional[List[str]]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _run_kg_node_query(
    session,
    book_id: str,
    chapter_nos: List[str],
    section_nos: List[str],
    terms: List[str],
    limit: int,
) -> List[Dict[str, Any]]:
    rows = session.run(
        """
        MATCH (:Book {book_id: $book_id})-[:HAS_CHAPTER]->(c:Chapter)-[:HAS_SECTION]->(s:Section)
        WHERE ($chapter_nos = [] OR c.chapter_no IN $chapter_nos)
          AND ($section_nos = [] OR s.section_no IN $section_nos)
        OPTIONAL MATCH (s)-[:MENTIONS]->(sn)
        OPTIONAL MATCH (ss:Subsection {book_id: $book_id, section_no: s.section_no})-[:MENTIONS]->(ssn)
        WITH c, s, collect(DISTINCT sn) + collect(DISTINCT ssn) AS nodes
        UNWIND nodes AS n
        WITH c, s, n
        WHERE n IS NOT NULL
          AND (
            size($terms) = 0
            OR any(t IN $terms WHERE toLower(coalesce(n.name, "")) CONTAINS t)
            OR any(t IN $terms WHERE toLower(coalesce(n.description, "")) CONTAINS t)
            OR any(t IN $terms WHERE toLower(coalesce(n.statement, "")) CONTAINS t)
            OR any(t IN $terms WHERE toLower(coalesce(n.expression, "")) CONTAINS t)
            OR any(t IN $terms WHERE toLower(coalesce(n.steps, "")) CONTAINS t)
          )
        RETURN labels(n)[0] AS node_label,
               coalesce(n.name, n.title, "") AS node_name,
               coalesce(n.description, n.statement, n.expression, n.steps, "") AS description,
               c.chapter_no AS chapter_no,
               s.section_no AS section_no,
               coalesce(n.page_start, s.page_start, c.page_start, 0) AS page_start,
               coalesce(n.page_end, s.page_end, c.page_end, 0) AS page_end
        ORDER BY toInteger(chapter_no), toInteger(split(section_no, '.')[1]), page_start
        LIMIT $limit
        """,
        book_id=book_id,
        chapter_nos=chapter_nos,
        section_nos=section_nos,
        terms=terms,
        limit=limit,
    ).data()
    return [dict(row) for row in rows]


def _run_kg_summary_query(
    session,
    book_id: str,
    chapter_nos: List[str],
    section_nos: List[str],
    terms: List[str],
    limit: int,
) -> List[Dict[str, Any]]:
    rows = session.run(
        """
        MATCH (:Book {book_id: $book_id})-[:HAS_CHAPTER]->(c:Chapter)
        WHERE ($chapter_nos = [] OR c.chapter_no IN $chapter_nos)
        OPTIONAL MATCH (c)-[:HAS_SUMMARY]->(csm:Summary {level: 'chapter'})
        OPTIONAL MATCH (c)-[:HAS_SECTION]->(s:Section)
        WITH c, csm, s
        WHERE ($section_nos = [] OR (s IS NOT NULL AND s.section_no IN $section_nos))
        OPTIONAL MATCH (s)-[:HAS_SUMMARY]->(ssm:Summary {level: 'section'})
        WITH c, s, csm, ssm,
             CASE WHEN ssm IS NOT NULL THEN ssm.text ELSE csm.text END AS summary_text,
             CASE WHEN ssm IS NOT NULL THEN ssm.page_start ELSE c.page_start END AS page_start,
             CASE WHEN ssm IS NOT NULL THEN ssm.page_end ELSE c.page_end END AS page_end
        WHERE summary_text IS NOT NULL AND summary_text <> ""
          AND (
            size($terms) = 0
            OR any(t IN $terms WHERE toLower(summary_text) CONTAINS t)
          )
        RETURN "Summary" AS node_label,
               CASE
                 WHEN s IS NOT NULL THEN "Section " + s.section_no + " Summary"
                 ELSE "Chapter " + c.chapter_no + " Summary"
               END AS node_name,
               summary_text AS description,
               c.chapter_no AS chapter_no,
               coalesce(s.section_no, "") AS section_no,
               coalesce(page_start, 0) AS page_start,
               coalesce(page_end, 0) AS page_end
        ORDER BY toInteger(chapter_no), section_no, page_start
        LIMIT $limit
        """,
        book_id=book_id,
        chapter_nos=chapter_nos,
        section_nos=section_nos,
        terms=terms,
        limit=limit,
    ).data()

    dedup = []
    seen = set()
    for row in rows:
        key = (
            row.get("node_label"),
            row.get("node_name"),
            row.get("chapter_no"),
            row.get("section_no"),
        )
        if key in seen:
            continue
        seen.add(key)
        dedup.append(dict(row))
    return dedup


def _resolve_scope_valid_sections(
    session,
    book_id: str,
    chapter_nos: List[str],
    section_nos: List[str],
) -> Dict[str, List[str]]:
    rows = session.run(
        """
        MATCH (:Book {book_id: $book_id})-[:HAS_CHAPTER]->(c:Chapter)-[:HAS_SECTION]->(s:Section)
        WHERE ($chapter_nos = [] OR c.chapter_no IN $chapter_nos)
          AND ($section_nos = [] OR s.section_no IN $section_nos)
        RETURN c.chapter_no AS chapter_no, s.section_no AS section_no
        ORDER BY toInteger(c.chapter_no), toInteger(split(s.section_no, '.')[1]), s.section_no
        """,
        book_id=book_id,
        chapter_nos=chapter_nos,
        section_nos=section_nos,
    ).data()

    valid_chapters = []
    valid_sections = []
    seen_ch = set()
    seen_sec = set()
    for row in rows:
        chapter_no = str(row.get("chapter_no") or "").strip()
        section_no = str(row.get("section_no") or "").strip()
        if chapter_no and chapter_no not in seen_ch:
            seen_ch.add(chapter_no)
            valid_chapters.append(chapter_no)
        if section_no and section_no not in seen_sec:
            seen_sec.add(section_no)
            valid_sections.append(section_no)
    return {"chapter_nos": valid_chapters, "section_nos": valid_sections}


def retrieve_kg_context_v2(
    book_id: str,
    chapter_nos: Optional[List[str]] = None,
    section_nos: Optional[List[str]] = None,
    query: str = "",
    limit: int = 20,
) -> Dict[str, Any]:
    chapter_scope = _safe_scope_ids(chapter_nos)
    section_scope = _safe_scope_ids(section_nos)
    terms = _extract_kg_terms(query)
    safe_limit = max(1, min(100, to_int(limit) or 20))

    attempts: List[Dict[str, Any]] = []
    warnings: List[str] = []

    with neo4j.session() as session:
        # Stage 1: strict_scope
        strict_rows = _run_kg_node_query(
            session,
            book_id=book_id,
            chapter_nos=chapter_scope,
            section_nos=section_scope,
            terms=terms,
            limit=safe_limit,
        )
        attempts.append({"stage": "strict_scope", "row_count": len(strict_rows)})
        if strict_rows:
            return {
                "rows": strict_rows,
                "stage": "strict_scope",
                "attempts": attempts,
                "effective_scope": {"chapter_nos": chapter_scope, "section_nos": section_scope},
                "warnings": warnings,
            }

        # Stage 2: scope_valid_sections
        valid_scope = {"chapter_nos": [], "section_nos": []}
        if chapter_scope or section_scope:
            valid_scope = _resolve_scope_valid_sections(session, book_id, chapter_scope, section_scope)
            if not valid_scope["section_nos"]:
                warnings.append("Requested scope had no valid sections; broadening to next stage.")

            valid_rows = _run_kg_node_query(
                session,
                book_id=book_id,
                chapter_nos=valid_scope["chapter_nos"],
                section_nos=valid_scope["section_nos"],
                terms=terms,
                limit=safe_limit,
            )
        else:
            valid_rows = []
        attempts.append({"stage": "scope_valid_sections", "row_count": len(valid_rows)})
        if valid_rows:
            return {
                "rows": valid_rows,
                "stage": "scope_valid_sections",
                "attempts": attempts,
                "effective_scope": valid_scope,
                "warnings": warnings,
            }

        # Stage 3: book_wide_sections
        broad_rows = _run_kg_node_query(
            session,
            book_id=book_id,
            chapter_nos=[],
            section_nos=[],
            terms=terms,
            limit=safe_limit,
        )
        attempts.append({"stage": "book_wide_sections", "row_count": len(broad_rows)})
        if broad_rows:
            warnings.append("Scoped retrieval returned no rows; using book-wide section traversal.")
            return {
                "rows": broad_rows,
                "stage": "book_wide_sections",
                "attempts": attempts,
                "effective_scope": {"chapter_nos": [], "section_nos": []},
                "warnings": warnings,
            }

        # Stage 4: chapter_summary_fallback
        summary_rows = _run_kg_summary_query(
            session,
            book_id=book_id,
            chapter_nos=[],
            section_nos=[],
            terms=terms,
            limit=safe_limit,
        )
        attempts.append({"stage": "chapter_summary_fallback", "row_count": len(summary_rows)})
        if summary_rows:
            warnings.append("Node traversal returned no rows; using summary fallback within KG.")
            return {
                "rows": summary_rows,
                "stage": "chapter_summary_fallback",
                "attempts": attempts,
                "effective_scope": {"chapter_nos": [], "section_nos": []},
                "warnings": warnings,
            }

    attempts.append({"stage": "empty", "row_count": 0})
    warnings.append("No KG evidence found after all broadening stages.")
    return {
        "rows": [],
        "stage": "empty",
        "attempts": attempts,
        "effective_scope": {"chapter_nos": [], "section_nos": []},
        "warnings": warnings,
    }


def retrieve_kg_context(
    book_id: str,
    chapter_nos: Optional[List[str]] = None,
    section_nos: Optional[List[str]] = None,
    query: str = "",
    limit: int = 20,
) -> List[Dict[str, Any]]:
    result = retrieve_kg_context_v2(
        book_id=book_id,
        chapter_nos=chapter_nos,
        section_nos=section_nos,
        query=query,
        limit=limit,
    )
    return result.get("rows", [])


def build_schema_context(book_id: str, chapter_no: Optional[str], section_no: Optional[str]) -> Dict[str, Any]:
    with neo4j.session() as session:
        label_rows = session.run(
            """
            MATCH (n {book_id: $book_id})
            UNWIND labels(n) AS label
            RETURN label, count(*) AS count
            ORDER BY count DESC, label
            """,
            book_id=book_id,
        ).data()

        rel_rows = session.run(
            """
            MATCH (a {book_id: $book_id})-[r]->(b {book_id: $book_id})
            RETURN type(r) AS rel, count(*) AS count
            ORDER BY count DESC, rel
            """,
            book_id=book_id,
        ).data()

        chapter_rows = session.run(
            """
            MATCH (:Book {book_id: $book_id})-[:HAS_CHAPTER]->(c:Chapter)
            RETURN c.chapter_no AS chapter_no, c.title AS title, c.page_start AS page_start, c.page_end AS page_end
            ORDER BY c.chapter_no
            """,
            book_id=book_id,
        ).data()

        section_rows = session.run(
            """
            MATCH (:Book {book_id: $book_id})-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s:Section)
            RETURN DISTINCT s.section_no AS section_no, s.title AS title, s.chapter_no AS chapter_no, s.page_start AS page_start, s.page_end AS page_end
            ORDER BY s.section_no
            """,
            book_id=book_id,
        ).data()

        sample_nodes = session.run(
            """
            MATCH (n {book_id: $book_id})
            WHERE n.name IS NOT NULL
            RETURN labels(n)[0] AS label, n.name AS name, n.page_start AS page_start, n.page_end AS page_end
            ORDER BY n.page_start ASC
            LIMIT 40
            """,
            book_id=book_id,
        ).data()

    return {
        "book_id": book_id,
        "scope": {"chapter_no": chapter_no, "section_no": section_no},
        "labels": label_rows,
        "relationship_types": rel_rows,
        "chapters": chapter_rows,
        "sections": section_rows,
        "sample_nodes": sample_nodes,
    }


def default_cypher_contract(question: str, scope: Dict[str, Any]) -> Dict[str, Any]:
    chapter_filter = "AND n.chapter_no = $chapter_no" if scope.get("chapter_no") else ""
    section_filter = "AND n.section_no = $section_no" if scope.get("section_no") else ""

    cypher = f"""
MATCH (n {{book_id: $book_id}})
WHERE n.name IS NOT NULL
  AND toLower(n.name) CONTAINS toLower($query)
  {chapter_filter}
  {section_filter}
RETURN labels(n)[0] AS node_label,
       n.name AS node_name,
       n.description AS description,
       n.page_start AS page_start,
       n.page_end AS page_end,
       n.chapter_no AS chapter_no,
       n.section_no AS section_no
ORDER BY n.page_start ASC
LIMIT $limit
""".strip()

    return {
        "intent": f"Find graph nodes relevant to: {question}",
        "cypher": cypher,
        "params": {
            "book_id": scope.get("book_id"),
            "chapter_no": scope.get("chapter_no"),
            "section_no": scope.get("section_no"),
            "query": question,
            "limit": 20,
        },
        "expected_columns": [
            "node_label",
            "node_name",
            "description",
            "page_start",
            "page_end",
            "chapter_no",
            "section_no",
        ],
    }


def generate_guarded_cypher(
    question: str,
    scope: Dict[str, Any],
    schema_context: Dict[str, Any],
    model_id: str,
    selected_model: str,
) -> Dict[str, Any]:
    prompt = (
        f"{CYPHER_GEN_PROMPT}\n\n"
        f"Question: {question}\n"
        f"Scope: {json.dumps(scope, ensure_ascii=False)}\n\n"
        f"Schema Context:\n{json.dumps(schema_context, ensure_ascii=False)}"
    )

    payload = invoke_model_json(model_id, selected_model, prompt, max_output_tokens=1200)
    if not isinstance(payload, dict):
        return default_cypher_contract(question, scope)

    if "cypher" not in payload:
        return default_cypher_contract(question, scope)

    return payload


def sanitize_cypher_text(cypher: str) -> str:
    text = str(cypher or "").strip()
    if not text:
        return text

    # Fix common LLM placeholder mistakes.
    text = re.sub(r"(?i)\bbook_id\s*:\s*book_id\b", "book_id: $book_id", text)
    text = re.sub(r"(?i)\bchapter_no\s*:\s*chapter_no\b", "chapter_no: $chapter_no", text)
    text = re.sub(r"(?i)\bsection_no\s*:\s*section_no\b", "section_no: $section_no", text)

    text = re.sub(r"(?i)\bLIMIT\s+limit\b", "LIMIT $limit", text)
    text = re.sub(r"(?i)\btoLower\(\s*query\s*\)", "toLower($query)", text)

    # Equality placeholders without '$' (keep left side unchanged).
    text = re.sub(r"(?i)(=\s*)book_id\b", r"\1$book_id", text)
    text = re.sub(r"(?i)(=\s*)chapter_no\b", r"\1$chapter_no", text)
    text = re.sub(r"(?i)(=\s*)section_no\b", r"\1$section_no", text)

    return text


def enforce_limit(cypher: str, hard_limit: int = 50) -> str:
    text = str(cypher or "").strip()
    if not text:
        return f"LIMIT $limit"

    # Keep only the first LIMIT line if LLM produced duplicates.
    line_limit_pattern = re.compile(r"(?im)^\s*LIMIT\s+(\$[A-Za-z_][A-Za-z0-9_]*|\d+)\s*$")
    seen_limit = 0

    def _dedup_limit(match: re.Match) -> str:
        nonlocal seen_limit
        seen_limit += 1
        return match.group(0) if seen_limit == 1 else ""

    text = line_limit_pattern.sub(_dedup_limit, text)

    limit_pattern = re.compile(r"(?i)\bLIMIT\s+(\$[A-Za-z_][A-Za-z0-9_]*|\d+)")
    match = limit_pattern.search(text)
    if not match:
        return f"{text}\nLIMIT $limit"

    token = match.group(1)
    if token.startswith("$"):
        return text

    try:
        current_limit = int(token)
    except Exception:
        return text

    if current_limit <= hard_limit:
        return text

    start, end = match.span(1)
    return f"{text[:start]}{hard_limit}{text[end:]}"


def validate_cypher_contract(contract: Dict[str, Any], scope: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []

    if not isinstance(contract, dict):
        return {"valid": False, "errors": ["Contract is not a JSON object"]}

    cypher = sanitize_cypher_text(str(contract.get("cypher") or "").strip())
    if not cypher:
        errors.append("Missing cypher")

    if cypher.count(";") > 1 or (";" in cypher and not cypher.strip().endswith(";")):
        errors.append("Multiple statements are not allowed")

    cypher_upper = cypher.upper()
    for pattern in FORBIDDEN_CYPHER_PATTERNS:
        if re.search(pattern, cypher_upper, flags=re.IGNORECASE):
            errors.append(f"Forbidden Cypher pattern: {pattern}")

    if "$book_id" not in cypher:
        errors.append("Cypher must use parameterized $book_id")

    if scope.get("chapter_no") and "$chapter_no" not in cypher:
        errors.append("Cypher must use parameterized $chapter_no when chapter scope is selected")

    if scope.get("section_no") and "$section_no" not in cypher:
        errors.append("Cypher must use parameterized $section_no when section scope is selected")

    if not re.search(r"\bMATCH\b", cypher, flags=re.IGNORECASE):
        errors.append("Cypher must include MATCH")

    if not re.search(r"\bRETURN\b", cypher, flags=re.IGNORECASE):
        errors.append("Cypher must include RETURN")

    params = contract.get("params", {})
    if not isinstance(params, dict):
        params = {}

    params["book_id"] = scope.get("book_id")
    params["chapter_no"] = scope.get("chapter_no")
    params["section_no"] = scope.get("section_no")

    limit = to_int(params.get("limit")) or 20
    params["limit"] = max(1, min(50, limit))

    cypher = enforce_limit(cypher, hard_limit=50).rstrip(";")

    validated = {
        "valid": len(errors) == 0,
        "errors": errors,
        "cypher": cypher,
        "params": params,
        "intent": contract.get("intent", ""),
        "expected_columns": contract.get("expected_columns", []),
    }
    return validated


def get_validated_default_contract(question: str, scope: Dict[str, Any]) -> Dict[str, Any]:
    fallback = default_cypher_contract(question, scope)
    return validate_cypher_contract(fallback, scope)


def repair_cypher_contract(
    question: str,
    scope: Dict[str, Any],
    schema_context: Dict[str, Any],
    failed_contract: Dict[str, Any],
    error_text: str,
    model_id: str,
    selected_model: str,
) -> Dict[str, Any]:
    prompt = (
        f"{CYPHER_REPAIR_PROMPT}\n\n"
        f"Question: {question}\n"
        f"Scope: {json.dumps(scope, ensure_ascii=False)}\n"
        f"Previous Contract: {json.dumps(failed_contract, ensure_ascii=False)}\n"
        f"Error: {error_text}\n\n"
        f"Schema Context:\n{json.dumps(schema_context, ensure_ascii=False)}"
    )

    payload = invoke_model_json(model_id, selected_model, prompt, max_output_tokens=1200)
    if isinstance(payload, dict):
        return payload
    return failed_contract


def run_read_cypher(cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    with neo4j.session() as session:
        # Pass params as a single map to avoid collisions with run() argument names
        # (e.g., a Cypher parameter named "query").
        result = session.run(cypher, parameters=params or {})
        return [dict(record) for record in result]


def explain_read_cypher(cypher: str, params: Dict[str, Any]) -> Optional[str]:
    try:
        with neo4j.session() as session:
            session.run(f"EXPLAIN {cypher}", parameters=params or {}).consume()
        return None
    except Exception as exc:
        return str(exc)


def extract_evidence_from_rows(rows: List[Dict[str, Any]], max_items: int = 20) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []

    for row in rows[:max_items]:
        if not isinstance(row, dict):
            continue

        node_label = row.get("node_label") or row.get("label") or ""
        node_name = row.get("node_name") or row.get("name") or ""
        page_start = to_int(row.get("page_start")) or 0
        page_end = to_int(row.get("page_end")) or page_start

        if not node_name:
            for value in row.values():
                if isinstance(value, dict) and value.get("name"):
                    node_name = str(value.get("name"))
                    node_label = value.get("label") or node_label
                    page_start = to_int(value.get("page_start")) or page_start
                    page_end = to_int(value.get("page_end")) or page_end
                    break

        if not node_name:
            continue

        path = []
        if row.get("chapter_no"):
            path.append(f"Chapter {row['chapter_no']}")
        if row.get("section_no"):
            path.append(f"Section {row['section_no']}")

        evidence.append(
            {
                "path": " > ".join(path),
                "node_label": str(node_label),
                "node_name": str(node_name),
                "page_start": page_start,
                "page_end": page_end,
            }
        )

    return evidence


def synthesize_structured_answer(
    question: str,
    scope: Dict[str, Any],
    cypher: str,
    rows: List[Dict[str, Any]],
    model_id: str,
    selected_model: str,
) -> Dict[str, Any]:
    if not rows:
        return {
            "answer": "No matching graph evidence was found for the query in the selected scope.",
            "cypher": cypher,
            "scope": {
                "book_id": scope.get("book_id"),
                "chapter_no": scope.get("chapter_no"),
                "section_no": scope.get("section_no"),
            },
            "evidence": [],
            "confidence": "low",
            "follow_up": [
                "Try broader wording.",
                "Remove section filter and query at book level.",
                "Ask for a specific concept or algorithm name.",
            ],
        }

    payload_preview = json.dumps(rows[:35], ensure_ascii=False)
    prompt = (
        f"{ANSWER_SYNTHESIS_PROMPT}\n\n"
        f"Question: {question}\n"
        f"Scope: {json.dumps(scope, ensure_ascii=False)}\n"
        f"Cypher: {cypher}\n"
        f"Rows: {payload_preview}"
    )

    payload = invoke_model_json(model_id, selected_model, prompt, max_output_tokens=1200)
    if isinstance(payload, dict):
        required_keys = {"answer", "cypher", "scope", "evidence", "confidence", "follow_up"}
        if required_keys.issubset(payload.keys()):
            payload["cypher"] = cypher
            payload["scope"] = {
                "book_id": scope.get("book_id"),
                "chapter_no": scope.get("chapter_no"),
                "section_no": scope.get("section_no"),
            }
            if not isinstance(payload.get("evidence"), list):
                payload["evidence"] = extract_evidence_from_rows(rows)
            return payload

    evidence = extract_evidence_from_rows(rows)
    brief = []
    for item in evidence[:5]:
        if item.get("path"):
            brief.append(f"{item['node_name']} ({item['path']})")
        else:
            brief.append(item["node_name"])

    return {
        "answer": (
            "Relevant graph evidence was found. "
            + ("Key matches: " + ", ".join(brief) if brief else "See evidence list.")
        ),
        "cypher": cypher,
        "scope": {
            "book_id": scope.get("book_id"),
            "chapter_no": scope.get("chapter_no"),
            "section_no": scope.get("section_no"),
        },
        "evidence": evidence,
        "confidence": "medium" if evidence else "low",
        "follow_up": [
            "Ask for a deeper explanation of one evidence node.",
            "Constrain by chapter or section for higher precision.",
        ],
    }


# ============================================================
# ORCHESTRATION
# ============================================================
def build_questindex_graph(
    pages: Dict[str, List[str]],
    book_id: str,
    model_id: str,
    selected_model: str,
    toc_scan_pages: int,
    pages_per_chunk: int,
    overlap: int,
    max_workers: int,
    skip_summaries: bool = False,
    skip_section_summaries: bool = False,
    enable_vector_indexing: bool = False,
    vector_index_name: str = QUESTINDEX_VECTOR_INDEX,
) -> Dict[str, Any]:
    page_text_map = pages_to_text_map(pages)
    if not page_text_map:
        raise Exception("No valid page text available")

    structure_payload = build_structure_from_toc_or_fallback(
        pages,
        model_id=model_id,
        selected_model=selected_model,
        toc_scan_pages=toc_scan_pages,
    )

    structure = structure_payload["structure"]
    diagnostics = structure_payload["diagnostics"]

    total_pages = max(page_text_map.keys())
    segments = build_extraction_segments(structure, total_pages, pages_per_chunk, overlap)

    extraction_results = process_segments_concurrently(
        segments,
        page_text_map,
        book_id,
        model_id,
        selected_model,
        max_workers=max_workers,
    )

    consolidated = consolidate_extraction_results(extraction_results, book_id)
    if skip_summaries:
        summaries = {"chapter_summaries": [], "section_summaries": []}
    else:
        summaries = generate_summaries(
            structure,
            consolidated,
            page_text_map,
            book_id,
            model_id,
            selected_model,
            skip_section_summaries=skip_section_summaries,
        )
    summary_status = {
        "skip_summaries": bool(skip_summaries),
        "skip_section_summaries": bool(skip_section_summaries),
        "chapter_summaries": len(summaries.get("chapter_summaries", [])),
        "section_summaries": len(summaries.get("section_summaries", [])),
    }

    stats = insert_questindex_graph(book_id, structure, consolidated, summaries)
    vector_stats = None
    vector_verify = None
    if enable_vector_indexing:
        target_index = (vector_index_name or QUESTINDEX_VECTOR_INDEX).strip() or QUESTINDEX_VECTOR_INDEX
        vector_stats = index_questindex_segments_opensearch(segments, page_text_map, book_id, target_index)
        vector_verify = verify_questindex_embeddings(target_index, book_id)

    return {
        "structure": structure,
        "diagnostics": diagnostics,
        "segments": segments,
        "consolidated": consolidated,
        "summaries": summaries,
        "summary_status": summary_status,
        "stats": stats,
        "vector_stats": vector_stats,
        "vector_verify": vector_verify,
    }
