import os, time, json, re, hashlib, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import parallel_bulk
from requests_aws4auth import AWS4Auth
from bedrockModels import build_request
import fitz
from botocore.exceptions import ClientError

# ============================================================
# ENV & CLIENTS
# ============================================================
load_dotenv()
REGION = os.environ["AWS_REGION"]
BUCKET = os.environ.get("S3_BUCKET", "")
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
OS_DOMAIN = os.environ.get("OS_DOMAIN", "")
KG_VECTOR_INDEX = os.environ.get("KG_VECTOR_INDEX", "kg-chunks")
EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
BOOKS_DIR = "books"
LOCAL_CACHE_DIR = "kg/cache"

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
            session_token=credentials.token
        )
        opensearch = OpenSearch(
            hosts=[{"host": OS_DOMAIN, "port": 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=60,
            max_retries=3,
            retry_on_timeout=True
        )
    except Exception:
        opensearch = None

# Thread lock for progress updates
progress_lock = threading.Lock()

# ============================================================
# UPDATED KG SYSTEM PROMPT
# ============================================================
KG_SYSTEM_PROMPT = """
You are an information extraction engine for academic textbooks.
Extract Knowledge Graph elements from the text.

FOCUS:
- Extract Level 1 (structural) nodes
- Extract Level 2 (content) nodes
- EXPLICITLY link Level 2 nodes to their Section

LEVEL 1 NODES (Structural – no chunk_id):
- Chapter
- Section

LEVEL 2 NODES (Content – must include chunk_id AND section_no):
- Concept
- Algorithm
- Question

CRITICAL RULE:
Every Concept, Algorithm, and Question MUST belong to EXACTLY ONE Section.
If multiple sections appear, choose the MOST RELEVANT one.

For every Level 2 node, ADD:
- section_no (e.g., "4.1", "4.2.3")

RELATIONSHIPS (IMPLICIT):
- Section MENTIONS Concept
- Section MENTIONS Algorithm
- Section MENTIONS Question

DO NOT output Section→Content relationships explicitly.
Just attach section_no to each content node.

OPTIONAL CROSS-CONTENT RELATIONSHIPS:
- You may output "relationships" to connect content nodes (Concept/Algorithm/Question).
- Each relationship should include source, target, and relation type.
- Preferred keys:
  - source: string or object with name/type
  - target: string or object with name/type
  - type: relation label
  - optional: source_type, target_type
- To express TWO-WAY relationships without duplicating objects:
  - optional: bidirectional (true/false)
  - optional: inverse_type (relation label for reverse direction)
  - if bidirectional=true and inverse_type is omitted, reverse uses same type.
- Keep top-level output structure unchanged:
  chapters, sections, concepts, algorithms, questions, relationships.

OUTPUT JSON ONLY.

EXAMPLE OUTPUT:
{
  "chapters": [
    {"chapter_no": "4", "title": "Binary Trees"}
  ],
  "sections": [
    {"section_no": "4.1", "title": "Binary Search Trees", "chapter_no": "4"}
  ],
  "concepts": [
    {
      "name": "Binary Search Tree",
      "description": "Ordered binary tree",
      "section_no": "4.1"
    }
  ],
  "algorithms": [
    {
      "name": "Tree Traversal",
      "description": "Visit nodes",
      "section_no": "4.1"
    }
  ],
  "questions": [
    {
      "name": "What is the time complexity of BST search?",
      "question_number": "1",
      "section_no": "4.1"
    }
  ],
  "relationships": [
    {
      "source": "Binary Search Tree",
      "source_type": "Concept",
      "target": "Tree Traversal",
      "target_type": "Algorithm",
      "type": "USES",
      "bidirectional": true,
      "inverse_type": "USED_BY"
    }
  ]
}
"""


# ============================================================
# HELPERS
# ============================================================
def safe_json_load(text):
    try:
        text = text.strip()
        if text.startswith('```json'):
            text = text[7:]
        if text.endswith('```'):
            text = text[:-3]
        return json.loads(text)
    except:
        return None

def make_id(text, prefix=""):
    return hashlib.blake2b(f"{prefix}{text}".encode(), digest_size=8).hexdigest()

def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()

def normalize_name(value):
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def normalize_chunk_text(value):
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def make_os_chunk_id(course_id, chunk_text):
    # Keep deterministic chunk identity aligned with 1_Ingestion.py.
    normalized = normalize_chunk_text(chunk_text)
    key_text = normalized[:48] if len(normalized) > 48 else normalized
    raw = f"{course_id}|{key_text}"
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()


def make_os_chunk_hash(chunk_text):
    normalized = normalize_chunk_text(chunk_text)
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).hexdigest()


def make_content_node_id(book_id, label, name):
    normalized_name = normalize_name(name)
    return make_id(f"{book_id}|{label}|{normalized_name}")

def normalize_label(value):
    raw = normalize_name(value)
    if raw in {"concept", "concepts"}:
        return "Concept"
    if raw in {"algorithm", "algorithms", "algo", "algos"}:
        return "Algorithm"
    if raw in {"question", "questions", "q"}:
        return "Question"
    return None

def normalize_rel_type(value):
    raw = normalize_name(value)
    if not raw:
        return None
    clean = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    return clean.upper() if clean else None

def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    raw = normalize_name(value)
    return raw in {"true", "yes", "y", "1", "bidirectional", "two_way", "two-way"}

def _read_json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _write_json_file(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

def list_books_folder_pdfs():
    if not os.path.isdir(BOOKS_DIR):
        return []
    return sorted(
        [name for name in os.listdir(BOOKS_DIR) if name.lower().endswith(".pdf")]
    )

def resolve_books_pdf_path(filename):
    if not filename:
        return None
    candidate = os.path.abspath(os.path.join(BOOKS_DIR, filename))
    books_root = os.path.abspath(BOOKS_DIR)
    if not candidate.startswith(books_root + os.sep) and candidate != books_root:
        return None
    return candidate if os.path.isfile(candidate) else None

def _s3_read_json(key):
    try:
        response = s3.get_object(Bucket=BUCKET, Key=key)
        body = response["Body"].read().decode("utf-8")
        return json.loads(body)
    except Exception:
        return None

def _s3_read_bytes(key):
    try:
        response = s3.get_object(Bucket=BUCKET, Key=key)
        return response["Body"].read()
    except Exception:
        return None

def load_pdf_bytes(source, source_kind):
    if source_kind == "books":
        pdf_path = resolve_books_pdf_path(source)
        if not pdf_path:
            raise FileNotFoundError(f"PDF not found in books folder: {source}")
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        return pdf_bytes, pdf_path

    if source_kind == "upload":
        if source is None:
            raise Exception("No uploaded file provided")
        return source.getvalue(), source.name

    if source_kind == "path":
        with open(source, "rb") as f:
            pdf_bytes = f.read()
        return pdf_bytes, source

    raise Exception(f"Unsupported source kind: {source_kind}")

def _first_present(obj, keys):
    for key in keys:
        if key in obj and obj[key] not in (None, ""):
            return obj[key]
    return None

def _extract_rel_endpoint(value, fallback_label=None):
    name = value
    label = fallback_label
    if isinstance(value, dict):
        name = _first_present(value, ["name", "node", "id", "value", "text", "title"])
        nested_label = _first_present(value, ["label", "type", "node_type", "entity_type"])
        label = nested_label if nested_label else fallback_label
    return name, normalize_label(label)

def normalize_relationship(raw_relationship):
    if not isinstance(raw_relationship, dict):
        return None

    source_value = _first_present(raw_relationship, ["source", "from", "src"])
    target_value = _first_present(raw_relationship, ["target", "to", "dst"])
    rel_type_value = _first_present(raw_relationship, ["type", "relation", "predicate", "rel_type"])

    source_hint = _first_present(raw_relationship, ["source_type", "source_label", "from_type", "from_label"])
    target_hint = _first_present(raw_relationship, ["target_type", "target_label", "to_type", "to_label"])
    bidirectional_value = _first_present(raw_relationship, ["bidirectional", "two_way", "twoWay", "is_bidirectional"])
    inverse_type_value = _first_present(raw_relationship, ["inverse_type", "reverse_type", "inverse", "inverse_relation"])

    source_name, source_label = _extract_rel_endpoint(source_value, source_hint)
    target_name, target_label = _extract_rel_endpoint(target_value, target_hint)
    rel_type = normalize_rel_type(rel_type_value)
    bidirectional = parse_bool(bidirectional_value)
    inverse_rel_type = normalize_rel_type(inverse_type_value)

    if not source_name or not target_name or not rel_type:
        return None

    return {
        "source_name": normalize_name(source_name),
        "source_label": source_label,
        "target_name": normalize_name(target_name),
        "target_label": target_label,
        "rel_type": rel_type,
        "bidirectional": bidirectional,
        "inverse_rel_type": inverse_rel_type or rel_type
    }

def build_content_lookup(concepts, algorithms, questions):
    by_label = {"Concept": {}, "Algorithm": {}, "Question": {}}
    by_any = {}
    label_to_nodes = [
        ("Concept", concepts),
        ("Algorithm", algorithms),
        ("Question", questions)
    ]

    for label, nodes in label_to_nodes:
        for node in nodes:
            node_name = normalize_name(node.get("name"))
            node_id = node.get("node_id")
            if not node_name or not node_id:
                continue

            by_label[label].setdefault(node_name, set()).add(node_id)
            by_any.setdefault(node_name, set()).add(node_id)

    return {"by_label": by_label, "by_any": by_any}

def _resolve_node_id(name, label, lookup):
    if label and label in lookup["by_label"]:
        ids = lookup["by_label"][label].get(name, set())
    else:
        ids = lookup["by_any"].get(name, set())

    if len(ids) == 1:
        return next(iter(ids))
    return None

def resolve_relationships(raw_relationships, lookup):
    resolved = []
    dedup = set()
    resolved_raw_count = 0

    def add_rel(source_node_id, target_node_id, rel_type):
        rel_key = (source_node_id, target_node_id, rel_type)
        if rel_key in dedup:
            return False
        dedup.add(rel_key)
        resolved.append({
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
            "rel_type": rel_type
        })
        return True

    for raw_relationship in raw_relationships:
        normalized = normalize_relationship(raw_relationship)
        if not normalized:
            continue

        source_node_id = _resolve_node_id(
            normalized["source_name"],
            normalized["source_label"],
            lookup
        )
        target_node_id = _resolve_node_id(
            normalized["target_name"],
            normalized["target_label"],
            lookup
        )

        if not source_node_id or not target_node_id:
            continue

        added_any = add_rel(source_node_id, target_node_id, normalized["rel_type"])
        if normalized.get("bidirectional"):
            reverse_type = normalized.get("inverse_rel_type") or normalized["rel_type"]
            added_any = add_rel(target_node_id, source_node_id, reverse_type) or added_any

        if added_any:
            resolved_raw_count += 1

    return resolved, resolved_raw_count

def sort_page_keys(keys):
    return sorted(
        keys,
        key=lambda k: (0, int(k)) if str(k).isdigit() else (1, str(k))
    )

def filter_pages_by_range(pages, start_page, end_page):
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
            "applied_end": end
        }

    filtered = {
        k: v for k, v in pages.items()
        if str(k).isdigit() and start <= int(k) <= end
    }
    return filtered, {
        "available_min": min_page,
        "available_max": max_page,
        "applied_start": start,
        "applied_end": end
    }

def chunk_pages(pages, pages_per_chunk, overlap, book_id):
    chunks = []
    keys = sort_page_keys(pages.keys())
    step = max(1, pages_per_chunk - overlap)

    for i in range(0, len(keys), step):
        block = keys[i:i+pages_per_chunk]
        text = "\n\n".join("\n".join(pages[p]) for p in block)
        if text.strip():
            stable_chunk_id = make_os_chunk_id(book_id, text)
            chunks.append({
                "text": text,
                "page_range": f"{block[0]}-{block[-1]}",
                "chunk_id": stable_chunk_id,
                "pages": [int(p) for p in block if str(p).isdigit()],
            })
    return chunks


def parse_page_range(page_range):
    match = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", str(page_range or ""))
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def get_text_embedding(text):
    payload = {"inputText": str(text or "")[:12000]}
    response = bedrock.invoke_model(modelId=EMBED_MODEL_ID, body=json.dumps(payload))
    data = json.loads(response["body"].read())
    return data.get("embedding", [])


def ensure_opensearch_vector_index(index_name, dimension):
    if opensearch is None:
        raise Exception("OpenSearch client unavailable. Set OS_DOMAIN and AWS auth.")

    exists = opensearch.indices.exists(index=index_name)
    if exists:
        return

    body = {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "book_id": {"type": "keyword"},
                "course_id": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "chunk_hash": {"type": "keyword"},
                "page_range": {"type": "keyword"},
                "page_start": {"type": "integer"},
                "page_end": {"type": "integer"},
                "pages": {"type": "integer"},
                "filename": {"type": "keyword"},
                "chunk_text": {"type": "text"},
                "source": {"type": "keyword"},
                "created_at": {"type": "date"},
                "vector_field": {"type": "knn_vector", "dimension": int(dimension)},
            }
        },
    }
    opensearch.indices.create(index=index_name, body=body)


def index_chunk_embeddings_opensearch(chunks, book_id, index_name):
    if opensearch is None:
        return {
            "indexed": 0,
            "errors": 0,
            "skipped": len(chunks),
            "index_name": index_name,
            "error": "OpenSearch is not configured.",
        }

    total = len(chunks)
    if total == 0:
        return {"indexed": 0, "errors": 0, "skipped": 0, "index_name": index_name}

    vector_rows = []
    embed_errors = 0
    progress_bar = st.progress(0.0)
    status = st.empty()

    for idx, chunk in enumerate(chunks, start=1):
        status.text(f"Generating embeddings {idx}/{total}")
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        try:
            vector = get_text_embedding(text)
            if not vector:
                embed_errors += 1
                continue
            page_start, page_end = parse_page_range(chunk.get("page_range"))
            vector_rows.append({
                "book_id": book_id,
                "course_id": book_id,
                "chunk_id": chunk.get("chunk_id"),
                "chunk_hash": make_os_chunk_hash(text),
                "page_range": chunk.get("page_range"),
                "page_start": page_start,
                "page_end": page_end,
                "pages": chunk.get("pages", []),
                "filename": f"{book_id}.pdf",
                "chunk_text": text,
                "vector_field": vector,
                "source": "6_KG",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
        except Exception:
            embed_errors += 1
        progress_bar.progress(min(1.0, idx / total))

    if not vector_rows:
        status.text("Embedding generation failed for all chunks")
        return {
            "indexed": 0,
            "errors": embed_errors,
            "skipped": total,
            "index_name": index_name,
            "error": "No embeddings generated.",
        }

    try:
        ensure_opensearch_vector_index(index_name, len(vector_rows[0]["vector_field"]))
    except Exception as exc:
        return {
            "indexed": 0,
            "errors": embed_errors,
            "skipped": total,
            "index_name": index_name,
            "error": f"Failed to prepare OpenSearch index: {exc}",
        }

    actions = []
    for row in vector_rows:
        os_id = row["chunk_id"]
        actions.append({
            "_op_type": "index",
            "_index": index_name,
            "_id": os_id,
            "_source": row
        })

    bulk_errors = 0
    indexed = 0
    try:
        for ok, result in parallel_bulk(opensearch, actions, thread_count=1, chunk_size=50):
            if ok:
                indexed += 1
            else:
                bulk_errors += 1
    except Exception as exc:
        return {
            "indexed": indexed,
            "errors": embed_errors + bulk_errors + 1,
            "skipped": max(0, total - len(vector_rows)),
            "index_name": index_name,
            "error": f"OpenSearch bulk indexing failed: {exc}",
        }

    # Ensure immediate visibility for verification right after indexing.
    try:
        opensearch.indices.refresh(index=index_name)
    except Exception:
        pass

    status.text("✅ OpenSearch vector indexing complete")
    return {
        "indexed": indexed,
        "errors": embed_errors + bulk_errors,
        "skipped": max(0, total - len(vector_rows)),
        "index_name": index_name,
        "vector_dim": len(vector_rows[0]["vector_field"]),
    }


def verify_opensearch_embeddings(index_name, book_id, sample_size=5):
    if opensearch is None:
        return {"verified": False, "error": "OpenSearch client unavailable."}

    try:
        source_value = "6_KG"
        book_id_text = str(book_id)

        def _book_clause():
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

        def _source_clause():
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

        sample_query = {
            "size": max(1, sample_size),
            "_source": ["chunk_id", "vector_field", "page_range", "book_id"],
            "query": exact_filter_query["query"] if verification_mode == "exact_term" else tolerant_filter_query["query"],
        }
        sample_hits = opensearch.search(index=index_name, body=sample_query).get("hits", {}).get("hits", [])

        sampled = len(sample_hits)
        vectors_present = 0
        vector_dim = None
        probe_vector = None

        for hit in sample_hits:
            src = hit.get("_source", {})
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
                            "must": {"knn": {"vector_field": {"vector": probe_vector, "k": 3}}}
                        }
                    }
                }
                knn_probe_hits = len(
                    opensearch.search(index=index_name, body=knn_query).get("hits", {}).get("hits", [])
                )
            except Exception as exc:
                knn_probe_error = str(exc)

        return {
            "verified": vectors_present > 0,
            "total_docs_for_book": total_docs,
            "sampled_docs": sampled,
            "sampled_with_vector": vectors_present,
            "vector_dim": vector_dim,
            "knn_probe_hits": knn_probe_hits,
            "knn_probe_error": knn_probe_error,
            "verification_mode": verification_mode,
            "index_refreshed": index_refreshed,
        }
    except Exception as exc:
        return {"verified": False, "error": str(exc)}

# ============================================================
# PDF PROCESSING WITH S3 CACHING
# ============================================================
def extract_text_with_pymupdf(pdf_bytes):
    pages = {}
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

def is_pymupdf_text_sufficient(pages, total_pages, total_chars):
    if total_pages <= 0:
        return False

    non_empty_pages = len(pages)
    coverage = non_empty_pages / total_pages
    min_char_threshold = max(300, total_pages * 20)

    return non_empty_pages > 0 and coverage >= 0.35 and total_chars >= min_char_threshold

def extract_text_with_textract_ocr(pdf_bytes, pdf_key):
    if not BUCKET:
        raise Exception("S3_BUCKET is missing in environment")

    def get_textract_result_with_backoff(job_id, next_token=None, max_attempts=8):
        delay = 1.0
        for attempt in range(1, max_attempts + 1):
            try:
                if next_token:
                    return textract.get_document_text_detection(JobId=job_id, NextToken=next_token)
                return textract.get_document_text_detection(JobId=job_id)
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                is_throttled = error_code in {
                    "ProvisionedThroughputExceededException",
                    "ThrottlingException",
                    "TooManyRequestsException",
                }
                if not is_throttled or attempt == max_attempts:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 20)

    s3.put_object(Bucket=BUCKET, Key=pdf_key, Body=pdf_bytes)
    job = textract.start_document_text_detection(
        DocumentLocation={"S3Object": {"Bucket": BUCKET, "Name": pdf_key}}
    )

    while True:
        result = get_textract_result_with_backoff(job["JobId"])
        if result["JobStatus"] in ["SUCCEEDED", "FAILED"]:
            break
        time.sleep(3)

    if result["JobStatus"] == "FAILED":
        raise Exception("Textract OCR processing failed")

    pages = {}
    token = None
    while True:
        resp = get_textract_result_with_backoff(job["JobId"], token) if token else result
        for block in resp["Blocks"]:
            if block["BlockType"] == "LINE":
                page = str(block["Page"])
                pages.setdefault(page, []).append(block["Text"])
        token = resp.get("NextToken")
        if not token:
            break

    return pages

def extract_pdf_text_local(pdf_source, source_kind, book_id):
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
        st.success(f"✅ Using local cached text for book: {book_id}")
        return cached_pages

    if cached_pages and cached_metadata:
        st.info("Local cache exists but PDF changed. Reprocessing...")
    else:
        st.info("No matching local cache found. Processing PDF...")

    pages, total_pages, total_chars = extract_text_with_pymupdf(pdf_bytes)
    extraction_method = "pymupdf"
    text_is_sufficient = is_pymupdf_text_sufficient(pages, total_pages, total_chars)

    if text_is_sufficient:
        st.success(
            f"✅ Extracted text with PyMuPDF ({len(pages)}/{total_pages} pages, {total_chars} chars)"
        )
    else:
        if BUCKET:
            st.warning("Text appears sparse. Trying Textract OCR fallback...")
            try:
                pdf_key = f"KG/{book_id}/source.pdf"
                pages = extract_text_with_textract_ocr(pdf_bytes, pdf_key)
                extraction_method = "textract_ocr_fallback"
                st.success(f"✅ Extracted text with Textract OCR fallback ({len(pages)} pages)")
            except Exception as ocr_error:
                if pages:
                    extraction_method = "pymupdf_partial"
                    st.warning(f"Textract fallback failed ({ocr_error}). Using partial PyMuPDF text.")
                else:
                    raise Exception(
                        f"No text extracted from selected PDF. Textract fallback also failed: {ocr_error}"
                    )
        else:
            extraction_method = "pymupdf_partial"
            st.warning("Text appears sparse. Using available PyMuPDF text only.")

    if not pages:
        raise Exception(
            "No text extracted from selected PDF. Use 'S3 Cache + Textract OCR' backend or set S3_BUCKET for OCR fallback."
        )

    _write_json_file(json_path, pages)
    _write_json_file(metadata_path, {
        "book_id": book_id,
        "pdf_filename": os.path.basename(source_name),
        "pdf_sha256": pdf_hash,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "page_count": len(pages),
        "extraction_method": extraction_method
    })

    st.success(f"✅ Processed and cached {len(pages)} pages for {book_id} via {extraction_method}")
    return pages

def extract_pdf_text_s3(pdf_source, source_kind, book_id):
    pdf_bytes, _ = load_pdf_bytes(pdf_source, source_kind)
    if not pdf_bytes:
        raise Exception("Selected PDF is empty")

    if not BUCKET:
        raise Exception("S3_BUCKET is missing in environment")

    pdf_hash = sha256_hex(pdf_bytes)
    cache_prefix = f"KG/{book_id}"
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
        st.success(f"✅ Using S3 cached text for book: {book_id}")
        return cached_pages

    if cached_pages and cached_metadata:
        st.info("S3 cache exists but PDF changed. Reprocessing...")
    else:
        st.info("No matching S3 cache found. Processing PDF...")

    pages, total_pages, total_chars = extract_text_with_pymupdf(pdf_bytes)
    extraction_method = "pymupdf"
    if not is_pymupdf_text_sufficient(pages, total_pages, total_chars):
        st.info("PyMuPDF text appears sparse. Falling back to Textract OCR...")
        pages = extract_text_with_textract_ocr(pdf_bytes, pdf_key)
        extraction_method = "textract_ocr"

    if not pages:
        raise Exception("No text extracted from selected PDF.")

    s3.put_object(
        Bucket=BUCKET,
        Key=json_key,
        Body=json.dumps(pages),
        ContentType="application/json"
    )
    s3.put_object(
        Bucket=BUCKET,
        Key=metadata_key,
        Body=json.dumps({
            "book_id": book_id,
            "pdf_sha256": pdf_hash,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "page_count": len(pages),
            "extraction_method": extraction_method
        }),
        ContentType="application/json"
    )

    st.success(f"✅ Processed and cached {len(pages)} pages for {book_id} via {extraction_method}")
    return pages

def extract_pdf_text(pdf_source, source_kind, book_id, extraction_mode):
    if extraction_mode == "s3":
        return extract_pdf_text_s3(pdf_source, source_kind, book_id)
    return extract_pdf_text_local(pdf_source, source_kind, book_id)


# ============================================================
# CONSTRAINTS SETUP
# ============================================================
def setup_constraints():
    constraints = [
        "CREATE CONSTRAINT book_unique IF NOT EXISTS FOR (b:Book) REQUIRE b.book_id IS UNIQUE",
        "CREATE CONSTRAINT chapter_unique IF NOT EXISTS FOR (c:Chapter) REQUIRE (c.book_id, c.chapter_no) IS UNIQUE",
        "CREATE CONSTRAINT section_unique IF NOT EXISTS FOR (s:Section) REQUIRE (s.book_id, s.section_no) IS UNIQUE",
        "CREATE CONSTRAINT concept_unique IF NOT EXISTS FOR (n:Concept) REQUIRE n.node_id IS UNIQUE",
        "CREATE CONSTRAINT algo_unique IF NOT EXISTS FOR (a:Algorithm) REQUIRE a.node_id IS UNIQUE",
        "CREATE CONSTRAINT question_unique IF NOT EXISTS FOR (q:Question) REQUIRE q.node_id IS UNIQUE"
    ]

    with neo4j.session() as session:
        for constraint in constraints:
            try:
                session.run(constraint)
            except:
                pass

# ============================================================
# KG EXTRACTION
# ============================================================
def extract_kg_from_chunk(chunk, book_id, model_id, selected_model):
    """Extract KG from a single chunk"""
    try:
        prompt = f"""{KG_SYSTEM_PROMPT}

BOOK_ID: {book_id}
CHUNK_ID: {chunk['chunk_id']}

TEXT:
{chunk['text']}

Extract all relevant nodes and relationships. Focus on Level 2 content nodes."""

        body = build_request(model_id, selected_model, prompt, 3000)
        resp = bedrock.invoke_model(modelId=model_id, body=json.dumps(body))
        raw = json.loads(resp["body"].read())
        raw_text = raw["content"][0]["text"]

        # Extract JSON
        json_match = re.search(r'\{[\s\S]*\}', raw_text)
        if not json_match:
            return create_empty_kg()

        parsed = safe_json_load(json_match.group())
        if not parsed:
            return create_empty_kg()

        # Structure and add IDs
        result = {
            "chapters": parsed.get("chapters", []),
            "sections": parsed.get("sections", []),
            "concepts": parsed.get("concepts", []),
            "algorithms": parsed.get("algorithms", []),
            "questions": parsed.get("questions", []),
            "relationships": parsed.get("relationships", [])
        }

        # Add book_id to structural nodes (Level 1)
        for item in result["chapters"] + result["sections"]:
            item["book_id"] = book_id

        # Add chunk_id and node_id to content nodes (Level 2)
        for node_list, label in [
            (result["concepts"], "Concept"),
            (result["algorithms"], "Algorithm"),
            (result["questions"], "Question")
        ]:
            for node in node_list:
                if "section_no" not in node:
                    continue  # skip invalid nodes

                node_name = node.get("name")
                if not normalize_name(node_name):
                    continue  # skip nameless nodes

                node["chunk_id"] = chunk["chunk_id"]
                node["node_id"] = make_content_node_id(book_id, label, node_name)
                node["label"] = label


        return result

    except Exception as e:
        st.warning(f"Extraction failed for chunk: {e}")
        return create_empty_kg()

def create_empty_kg():
    return {
        "chapters": [], "sections": [], "concepts": [], 
        "algorithms": [], "questions": [], "relationships": []
    }

# ============================================================
# CONCURRENT KG PROCESSING
# ============================================================
def process_chunks_concurrently(chunks, book_id, model_id, selected_model, max_workers=3):
    """Process chunks concurrently with progress tracking"""
    all_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    completed = 0
    total = len(chunks)

    def update_progress():
        nonlocal completed
        with progress_lock:
            completed += 1
            progress = completed / total
            progress_bar.progress(progress)
            status_text.text(f"Processing chunk {completed}/{total}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_chunk = {
            executor.submit(extract_kg_from_chunk, chunk, book_id, model_id, selected_model): chunk 
            for chunk in chunks
        }

        # Collect results as they complete
        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            try:
                result = future.result()
                result['page_range'] = chunk['page_range']
                all_results.append(result)
                update_progress()
            except Exception as e:
                st.error(f"Failed to process chunk {chunk['page_range']}: {e}")
                update_progress()

    status_text.text("✅ All chunks processed!")
    return all_results

# ============================================================
# NEO4J INSERTION WITH DEDUPLICATION
# ============================================================
def insert_kg_batch(all_results, book_id):
    """Insert all KG results with deduplication"""

    # Collect and deduplicate
    all_chapters = {}
    all_sections = {}
    all_concepts = {}
    all_algorithms = {}
    all_questions = {}
    all_relationships = []

    for result in all_results:
        # Deduplicate chapters
        for ch in result["chapters"]:
            key = ch["chapter_no"]
            all_chapters[key] = ch

        # Deduplicate sections  
        for sec in result["sections"]:
            key = sec["section_no"]
            all_sections[key] = sec

        # Deduplicate concepts (lowercase normalization)
        for concept in result["concepts"]:
            key = normalize_name(concept.get("name"))
            if not key:
                continue
            if key not in all_concepts:
                all_concepts[key] = concept

        # Deduplicate algorithms (lowercase normalization)
        for algo in result["algorithms"]:
            key = normalize_name(algo.get("name"))
            if not key:
                continue
            if key not in all_algorithms:
                all_algorithms[key] = algo

        # Deduplicate questions (lowercase normalization)
        for q in result["questions"]:
            key = normalize_name(q.get("name"))
            if not key:
                continue
            if key not in all_questions:
                all_questions[key] = q

        # Collect relationships
        all_relationships.extend(result["relationships"])

    node_lookup = build_content_lookup(
        list(all_concepts.values()),
        list(all_algorithms.values()),
        list(all_questions.values())
    )
    resolved_relationships, resolved_raw_count = resolve_relationships(all_relationships, node_lookup)
    relationship_stats = {
        "relationships_seen": len(all_relationships),
        "relationships_inserted": len(resolved_relationships),
        "relationships_skipped": len(all_relationships) - resolved_raw_count
    }

    # Insert into Neo4j
    with neo4j.session() as session:
        session.execute_write(_insert_all_nodes, book_id, 
                            list(all_chapters.values()),
                            list(all_sections.values()),
                            list(all_concepts.values()),
                            list(all_algorithms.values()),
                            list(all_questions.values()),
                            resolved_relationships)

    return {
        "chapters": len(all_chapters),
        "sections": len(all_sections), 
        "concepts": len(all_concepts),
        "algorithms": len(all_algorithms),
        "questions": len(all_questions),
        "relationships": relationship_stats["relationships_inserted"],
        "relationships_seen": relationship_stats["relationships_seen"],
        "relationships_inserted": relationship_stats["relationships_inserted"],
        "relationships_skipped": relationship_stats["relationships_skipped"]
    }
def _insert_all_nodes(
    tx,
    book_id,
    chapters,
    sections,
    concepts,
    algorithms,
    questions,
    relationships=None
):
    """
    Insert all nodes and relationships in a single transaction.

    FIX-2:
    - Level 2 nodes MUST have section_no
    - Section -> Level 2 nodes are linked deterministically via section_no
    """

    # ------------------------------------------------------------
    # Book
    # ------------------------------------------------------------
    tx.run(
        "MERGE (b:Book {book_id: $book_id})",
        book_id=book_id
    )

    # ------------------------------------------------------------
    # Chapters (Level 1)
    # ------------------------------------------------------------
    for ch in chapters:
        tx.run(
            """
            MERGE (c:Chapter {book_id: $book_id, chapter_no: $chapter_no})
            SET c.title = $title
            WITH c
            MATCH (b:Book {book_id: $book_id})
            MERGE (b)-[:HAS_CHAPTER]->(c)
            """,
            book_id=book_id,
            chapter_no=ch["chapter_no"],
            title=ch.get("title", "")
        )

    # ------------------------------------------------------------
    # Sections (Level 1)
    # ------------------------------------------------------------
    for sec in sections:
        tx.run(
            """
            MERGE (s:Section {book_id: $book_id, section_no: $section_no})
            SET s.title = $title,
                s.chapter_no = $chapter_no
            """,
            book_id=book_id,
            section_no=sec["section_no"],
            title=sec.get("title", ""),
            chapter_no=sec.get("chapter_no", "")
        )

        if sec.get("chapter_no"):
            tx.run(
                """
                MATCH (c:Chapter {book_id: $book_id, chapter_no: $chapter_no})
                MATCH (s:Section {book_id: $book_id, section_no: $section_no})
                MERGE (c)-[:HAS_SECTION]->(s)
                """,
                book_id=book_id,
                chapter_no=sec["chapter_no"],
                section_no=sec["section_no"]
            )

    # ------------------------------------------------------------
    # Concepts (Level 2) + Section Linking
    # ------------------------------------------------------------
    for concept in concepts:
        if "section_no" not in concept:
            continue

        tx.run(
            """
            MATCH (s:Section {book_id: $book_id, section_no: $section_no})
            MERGE (c:Concept {node_id: $node_id})
            SET c.name = $name,
                c.book_id = $book_id,
                c.chunk_id = $chunk_id,
                c.description = $description
            MERGE (s)-[:MENTIONS]->(c)
            """,
            book_id=book_id,
            section_no=concept["section_no"],
            node_id=concept["node_id"],
            name=concept["name"],
            chunk_id=concept["chunk_id"],
            description=concept.get("description", "")
        )

    # ------------------------------------------------------------
    # Algorithms (Level 2) + Section Linking
    # ------------------------------------------------------------
    for algo in algorithms:
        if "section_no" not in algo:
            continue

        tx.run(
            """
            MATCH (s:Section {book_id: $book_id, section_no: $section_no})
            MERGE (a:Algorithm {node_id: $node_id})
            SET a.name = $name,
                a.book_id = $book_id,
                a.chunk_id = $chunk_id,
                a.description = $description
            MERGE (s)-[:MENTIONS]->(a)
            """,
            book_id=book_id,
            section_no=algo["section_no"],
            node_id=algo["node_id"],
            name=algo["name"],
            chunk_id=algo["chunk_id"],
            description=algo.get("description", "")
        )

    # ------------------------------------------------------------
    # Questions (Level 2) + Section Linking
    # ------------------------------------------------------------
    for q in questions:
        if "section_no" not in q:
            continue

        tx.run(
            """
            MATCH (s:Section {book_id: $book_id, section_no: $section_no})
            MERGE (q:Question {node_id: $node_id})
            SET q.name = $name,
                q.book_id = $book_id,
                q.chunk_id = $chunk_id,
                q.question_number = $question_number
            MERGE (s)-[:MENTIONS]->(q)
            """,
            book_id=book_id,
            section_no=q["section_no"],
            node_id=q["node_id"],
            name=q["name"],
            chunk_id=q["chunk_id"],
            question_number=q.get("question_number", "")
        )

    # ------------------------------------------------------------
    # Optional Content-to-Content Relationships
    # ------------------------------------------------------------
    for rel in relationships or []:
        tx.run(
            """
            MATCH (source {node_id: $source_node_id})
            MATCH (target {node_id: $target_node_id})
            MERGE (source)-[:RELATED_TO {rel_type: $rel_type}]->(target)
            """,
            source_node_id=rel["source_node_id"],
            target_node_id=rel["target_node_id"],
            rel_type=rel["rel_type"]
        )

def delete_entire_graph():
    with neo4j.session() as session:
        total = session.run("MATCH (n) RETURN count(n) AS total").single()["total"]
        session.run("MATCH (n) DETACH DELETE n")
    return total

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(layout="wide")
st.title("📚 AUQA – Enhanced KG Builder v2")

if "chunks" not in st.session_state:
    st.session_state.chunks = []

setup_constraints()
st.success("✅ Neo4j constraints ready")

# Input controls
col1, col2 = st.columns(2)
with col1:
    source_option = st.radio(
        "PDF Source",
        ["Upload", "Books Folder", "Server Path"],
        horizontal=True
    )
    uploaded_pdf = None
    selected_books_pdf = None
    pdf_local_path = ""

    if source_option == "Upload":
        uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    elif source_option == "Books Folder":
        books_pdfs = list_books_folder_pdfs()
        if books_pdfs:
            selected_books_pdf = st.selectbox("Select PDF from books/", books_pdfs)
        else:
            st.warning("No PDF files found in books/ folder.")
    else:
        pdf_local_path = st.text_input("PDF path on server", "")

with col2:
    book_id = st.text_input("Book ID", "DSA_Textbook")
    pages_per_chunk = st.slider("Pages per chunk", 1, 32, 4)
    overlap = st.slider("Overlap pages", 0, pages_per_chunk-1, 1)
    page_start = st.number_input("Start Page", min_value=1, value=1, step=1)
    page_end = st.number_input("End Page (0 = last page)", min_value=0, value=0, step=1)
    extraction_mode_label = st.radio(
        "Extraction Backend",
        ["Local (PyMuPDF, no S3)", "S3 Cache + Textract OCR"],
        index=0
    )
    enable_vector_indexing = st.checkbox(
        "Also index chunk embeddings in OpenSearch",
        value=True,
        help="Creates vector embeddings for each chunk and stores them in OpenSearch for semantic search."
    )
    vector_index_name = st.text_input("OpenSearch vector index", KG_VECTOR_INDEX)

# Model selection
with open("models.json") as f:
    MODELS = json.load(f)
model_names = [m["name"] for m in MODELS]
selected_model = st.selectbox("Choose Model", model_names)
model_id = next(m["id"] for m in MODELS if m["name"] == selected_model)

st.divider()
with st.expander("⚠️ Danger Zone: Delete Entire Neo4j Graph"):
    confirm_delete = st.checkbox("I understand this deletes all books and graph data.")
    delete_phrase = st.text_input("Type DELETE ALL to confirm", value="")
    if st.button("🗑️ Delete Entire Graph"):
        if not confirm_delete or delete_phrase.strip() != "DELETE ALL":
            st.error("Deletion blocked. Check confirmation and type DELETE ALL exactly.")
        else:
            with st.spinner("Deleting entire graph..."):
                deleted_count = delete_entire_graph()
            st.session_state.chunks = []
            st.success(f"✅ Graph cleared. Deleted {deleted_count} nodes.")

# PDF Processing
if st.button("📄 Extract PDF Text"):
    extraction_mode = "s3" if extraction_mode_label == "S3 Cache + Textract OCR" else "local"

    pdf_source = None
    source_kind = None
    if source_option == "Upload" and uploaded_pdf is not None:
        pdf_source = uploaded_pdf
        source_kind = "upload"
    elif source_option == "Books Folder" and selected_books_pdf:
        pdf_source = selected_books_pdf
        source_kind = "books"
    elif source_option == "Server Path" and pdf_local_path.strip():
        pdf_source = pdf_local_path.strip()
        source_kind = "path"

    if pdf_source is None or source_kind is None:
        st.error("Select a valid PDF source first.")
    else:
        try:
            with st.spinner("Processing PDF..."):
                pages = extract_pdf_text(pdf_source, source_kind, book_id, extraction_mode)
                pages, range_info = filter_pages_by_range(
                    pages,
                    int(page_start),
                    int(page_end)
                )

                if not pages:
                    if range_info:
                        raise Exception(
                            "No pages in selected range "
                            f"{range_info['applied_start']}-{range_info['applied_end']}. "
                            f"Available pages: {range_info['available_min']}-{range_info['available_max']}"
                        )
                    raise Exception("No pages extracted from PDF.")

                if range_info:
                    st.info(
                        "Processing page range "
                        f"{range_info['applied_start']}-{range_info['applied_end']} "
                        f"(available: {range_info['available_min']}-{range_info['available_max']})"
                    )

                st.session_state.chunks = chunk_pages(pages, pages_per_chunk, overlap, book_id)
                st.success(
                    f"✅ Created {len(st.session_state.chunks)} chunks from {len(pages)} pages"
                )
        except FileNotFoundError:
            if source_kind == "path":
                st.error(f"PDF file not found: {pdf_local_path.strip()}")
            else:
                st.error("Selected PDF file was not found.")
        except Exception as e:
            st.error(f"PDF extraction failed: {e}")
            st.info("Retry with Local backend or use a different source. For S3 mode, verify AWS credentials and S3/Textract permissions.")

# KG Building
if st.button("🚀 Build Knowledge Graph"):
    if not st.session_state.chunks:
        st.error("Extract PDF first")
    else:
        try:
            st.info("Processing chunks concurrently...")

            # Process all chunks concurrently
            all_results = process_chunks_concurrently(
                st.session_state.chunks, book_id, model_id, selected_model
            )

            # Insert into Neo4j with deduplication
            with st.spinner("Inserting into Neo4j..."):
                stats = insert_kg_batch(all_results, book_id)

            os_stats = None
            os_verify = None
            if enable_vector_indexing:
                target_index = (vector_index_name or KG_VECTOR_INDEX).strip() or KG_VECTOR_INDEX
                with st.spinner(f"Indexing chunk embeddings in OpenSearch ({target_index})..."):
                    os_stats = index_chunk_embeddings_opensearch(
                        st.session_state.chunks,
                        book_id,
                        target_index
                    )
                with st.spinner("Verifying OpenSearch embeddings..."):
                    os_verify = verify_opensearch_embeddings(target_index, book_id)

            st.success("✅ Knowledge Graph Complete!")

            # Show stats
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Chapters", stats["chapters"])
            col2.metric("Sections", stats["sections"])
            col3.metric("Concepts", stats["concepts"])
            col4.metric("Algorithms", stats["algorithms"])
            col5.metric("Questions", stats["questions"])

            r1, r2, r3 = st.columns(3)
            r1.metric("Relationships Seen", stats["relationships_seen"])
            r2.metric("Relationships Inserted", stats["relationships_inserted"])
            r3.metric("Relationships Skipped", stats["relationships_skipped"])

            if os_stats is not None:
                st.subheader("OpenSearch Vector Indexing")
                v1, v2, v3 = st.columns(3)
                v1.metric("Indexed", os_stats.get("indexed", 0))
                v2.metric("Skipped", os_stats.get("skipped", 0))
                v3.metric("Errors", os_stats.get("errors", 0))
                st.caption(f"Index: {os_stats.get('index_name', '')}")
                if os_stats.get("error"):
                    st.warning(f"Vector indexing warning: {os_stats['error']}")
                if os_verify is not None:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Verified (sample)", "Yes" if os_verify.get("verified") else "No")
                    c2.metric("Docs in index (book)", os_verify.get("total_docs_for_book", 0))
                    c3.metric("Sampled with vector", os_verify.get("sampled_with_vector", 0))
                    if os_verify.get("verified"):
                        st.success("Chunk embedding verification passed.")
                    else:
                        st.warning("Chunk embedding verification failed for sampled documents.")
                    if os_verify.get("error"):
                        st.warning(f"Embedding verification warning: {os_verify['error']}")
                    elif os_verify.get("knn_probe_error"):
                        st.info(f"kNN probe warning: {os_verify['knn_probe_error']}")
        except Exception as e:
            st.error(f"KG build failed: {e}")
