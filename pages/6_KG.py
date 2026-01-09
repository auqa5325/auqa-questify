import os, time, json, re, hashlib, uuid, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase
from bedrockModels import build_request

# ============================================================
# ENV & CLIENTS
# ============================================================
load_dotenv()
REGION = os.environ["AWS_REGION"]
BUCKET = os.environ["S3_BUCKET"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

session = boto3.Session(region_name=REGION)
textract = boto3.client("textract", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
bedrock = session.client("bedrock-runtime", region_name=REGION)
neo4j = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

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

def chunk_pages(pages, pages_per_chunk, overlap):
    chunks = []
    keys = sorted(pages.keys())
    step = max(1, pages_per_chunk - overlap)

    for i in range(0, len(keys), step):
        block = keys[i:i+pages_per_chunk]
        text = "\n\n".join("\n".join(pages[p]) for p in block)
        if text.strip():
            chunks.append({
                "text": text,
                "page_range": f"{block[0]}-{block[-1]}",
                "chunk_id": make_id(text[:100])
            })
    return chunks

# ============================================================
# PDF PROCESSING WITH S3 CACHING
# ============================================================
def extract_pdf_text(uploaded_pdf, book_id):
    """
    Extract text from PDF using book_id as deterministic S3 ID
    """

    json_key = f"KG/{book_id}.json"
    pdf_key = f"KG/{book_id}.pdf"

    # Try cache first
    try:
        response = s3.get_object(Bucket=BUCKET, Key=json_key)
        pages = json.loads(response["Body"].read())
        st.success(f"✅ Using cached text for book: {book_id}")
        return pages
    except:
        st.info("No cached text found. Processing PDF...")

    pdf_bytes = uploaded_pdf.read()

    # Upload PDF
    s3.put_object(
        Bucket=BUCKET,
        Key=pdf_key,
        Body=pdf_bytes
    )

    # Start Textract
    job = textract.start_document_text_detection(
        DocumentLocation={
            "S3Object": {"Bucket": BUCKET, "Name": pdf_key}
        }
    )

    # Wait for completion
    while True:
        result = textract.get_document_text_detection(JobId=job["JobId"])
        if result["JobStatus"] in ["SUCCEEDED", "FAILED"]:
            break
        time.sleep(3)

    if result["JobStatus"] == "FAILED":
        raise Exception("Textract processing failed")

    # Collect pages
    pages = {}
    token = None
    while True:
        resp = (
            textract.get_document_text_detection(JobId=job["JobId"], NextToken=token)
            if token else result
        )

        for block in resp["Blocks"]:
            if block["BlockType"] == "LINE":
                page = str(block["Page"])
                pages.setdefault(page, []).append(block["Text"])

        token = resp.get("NextToken")
        if not token:
            break

    # Cache extracted text
    s3.put_object(
        Bucket=BUCKET,
        Key=json_key,
        Body=json.dumps(pages),
        ContentType="application/json"
    )

    st.success(f"✅ Processed and cached {len(pages)} pages for {book_id}")
    return pages


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

                node["chunk_id"] = chunk["chunk_id"]
                node["node_id"] = make_id(f"{label}_{node['name']}")
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
            key = concept["name"].lower()  # Convert to lowercase for dedup
            if key not in all_concepts:
                all_concepts[key] = concept

        # Deduplicate algorithms (lowercase normalization)
        for algo in result["algorithms"]:
            key = algo["name"].lower()  # Convert to lowercase for dedup
            if key not in all_algorithms:
                all_algorithms[key] = algo

        # Deduplicate questions (lowercase normalization)
        for q in result["questions"]:
            key = q["name"].lower()  # Convert to lowercase for dedup
            if key not in all_questions:
                all_questions[key] = q

        # Collect relationships
        all_relationships.extend(result["relationships"])

    # Insert into Neo4j
    with neo4j.session() as session:
        session.execute_write(_insert_all_nodes, book_id, 
                            list(all_chapters.values()),
                            list(all_sections.values()),
                            list(all_concepts.values()),
                            list(all_algorithms.values()),
                            list(all_questions.values()),
                            all_relationships)

    return {
        "chapters": len(all_chapters),
        "sections": len(all_sections), 
        "concepts": len(all_concepts),
        "algorithms": len(all_algorithms),
        "questions": len(all_questions),
        "relationships": len(all_relationships)
    }
def _insert_all_nodes(
    tx,
    book_id,
    chapters,
    sections,
    concepts,
    algorithms,
    questions,
    relationships=None  # kept for compatibility, not used
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
    book_id = st.text_input("Book ID", "DSA_Textbook")
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

with col2:
    pages_per_chunk = st.slider("Pages per chunk", 1, 32, 4)
    overlap = st.slider("Overlap pages", 0, pages_per_chunk-1, 1)

# Model selection
with open("models.json") as f:
    MODELS = json.load(f)
model_names = [m["name"] for m in MODELS]
selected_model = st.selectbox("Choose Model", model_names)
model_id = next(m["id"] for m in MODELS if m["name"] == selected_model)

# PDF Processing
if st.button("📄 Extract PDF Text"):
    if not uploaded_pdf:
        st.error("Upload PDF first")
    else:
        with st.spinner("Processing PDF..."):
            pdf_uuid = str(uuid.uuid4())
            pages = extract_pdf_text(uploaded_pdf, pdf_uuid)
            st.session_state.chunks = chunk_pages(pages, pages_per_chunk, overlap)
            st.success(f"✅ Created {len(st.session_state.chunks)} chunks")

# KG Building
if st.button("🚀 Build Knowledge Graph"):
    if not st.session_state.chunks:
        st.error("Extract PDF first")
    else:
        st.info("Processing chunks concurrently...")

        # Process all chunks concurrently
        all_results = process_chunks_concurrently(
            st.session_state.chunks, book_id, model_id, selected_model
        )

        # Insert into Neo4j with deduplication
        with st.spinner("Inserting into Neo4j..."):
            stats = insert_kg_batch(all_results, book_id)

        st.success("✅ Knowledge Graph Complete!")

        # Show stats
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Chapters", stats["chapters"])
        col2.metric("Sections", stats["sections"])
        col3.metric("Concepts", stats["concepts"])
        col4.metric("Algorithms", stats["algorithms"])
        col5.metric("Questions", stats["questions"])
