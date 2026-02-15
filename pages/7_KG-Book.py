import os
import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase

# ============================================================
# ENV & NEO4J
# ============================================================
load_dotenv()

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

neo4j = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

# ============================================================
# NEO4J QUERY HELPERS
# ============================================================
def get_books():
    query = "MATCH (b:Book) RETURN b.book_id AS book ORDER BY book"
    with neo4j.session() as session:
        return [r["book"] for r in session.run(query)]


def get_node_counts(book_id):
    query = """
    MATCH (b:Book {book_id: $book_id})
    OPTIONAL MATCH (b)-[:HAS_CHAPTER]->(c:Chapter)
    OPTIONAL MATCH (c)-[:HAS_SECTION]->(s:Section)
    OPTIONAL MATCH (x {book_id: $book_id})
    RETURN
        count(DISTINCT c) AS chapters,
        count(DISTINCT s) AS sections,
        size([n IN collect(DISTINCT x) WHERE n:Concept]) AS concepts,
        size([n IN collect(DISTINCT x) WHERE n:Algorithm]) AS algorithms,
        size([n IN collect(DISTINCT x) WHERE n:Question]) AS questions
    """
    with neo4j.session() as session:
        return session.run(query, book_id=book_id).single()


def get_chapters(book_id):
    query = """
    MATCH (b:Book {book_id: $book_id})-[:HAS_CHAPTER]->(c:Chapter)
    RETURN c.chapter_no AS no, c.title AS title
    ORDER BY c.chapter_no
    """
    with neo4j.session() as session:
        return list(session.run(query, book_id=book_id))


def get_sections(book_id, chapter_no):
    query = """
    MATCH (c:Chapter {book_id: $book_id, chapter_no: $chapter_no})
          -[:HAS_SECTION]->(s:Section)
    RETURN s.section_no AS no, s.title AS title
    ORDER BY s.section_no
    """
    with neo4j.session() as session:
        return list(session.run(query, book_id=book_id, chapter_no=chapter_no))


def get_sections_with_content(book_id, chapter_no):
    query = """
    MATCH (c:Chapter {book_id: $book_id, chapter_no: $chapter_no})
          -[:HAS_SECTION]->(s:Section)
    OPTIONAL MATCH (s)-[:MENTIONS]->(n)
    WITH
        s,
        [name IN collect(DISTINCT CASE WHEN n:Concept THEN n.name END) WHERE name IS NOT NULL] AS concepts,
        [name IN collect(DISTINCT CASE WHEN n:Algorithm THEN n.name END) WHERE name IS NOT NULL] AS algorithms
    RETURN s.section_no AS no, s.title AS title, concepts, algorithms
    ORDER BY s.section_no
    """
    with neo4j.session() as session:
        rows = []
        for record in session.run(query, book_id=book_id, chapter_no=chapter_no):
            rows.append({
                "no": record["no"],
                "title": record["title"],
                "concepts": sorted(record["concepts"] or []),
                "algorithms": sorted(record["algorithms"] or [])
            })
        return rows


def get_all_questions(book_id):
    query = """
    MATCH (q:Question {book_id: $book_id})
    WITH q, trim(toString(coalesce(q.question_number, ""))) AS qno_sort
    RETURN q.name AS question, q.question_number AS qno
    ORDER BY
        CASE WHEN qno_sort =~ '^[0-9]+$' THEN 0 ELSE 1 END,
        CASE WHEN qno_sort =~ '^[0-9]+$' THEN toInteger(qno_sort) ELSE 0 END,
        toLower(qno_sort),
        toLower(coalesce(q.name, ""))
    """
    with neo4j.session() as session:
        return list(session.run(query, book_id=book_id))

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(layout="wide")
st.title("📘 AUQA — Knowledge Graph Explorer")

books = get_books()

if not books:
    st.warning("No books found in Neo4j.")
    st.stop()

book_id = st.selectbox("📚 Select Book", books)

# ------------------------------------------------------------
# NODE COUNTS
# ------------------------------------------------------------
counts = get_node_counts(book_id)

st.subheader("📊 Node Statistics")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Chapters", counts["chapters"])
c2.metric("Sections", counts["sections"])
c3.metric("Concepts", counts["concepts"])
c4.metric("Algorithms", counts["algorithms"])
c5.metric("Questions", counts["questions"])

st.divider()

# ------------------------------------------------------------
# NESTED GRAPH VIEW
# ------------------------------------------------------------
st.subheader("🧭 Knowledge Graph Structure")

chapters = get_chapters(book_id)

for ch in chapters:
    with st.expander(f"📕 Chapter {ch['no']}: {ch['title']}"):
        sections = get_sections_with_content(book_id, ch["no"])

        for sec in sections:
            with st.expander(f"📘 Section {sec['no']}: {sec['title']}"):
                concepts = sec["concepts"]
                algos = sec["algorithms"]

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**🧠 Concepts**")
                    if concepts:
                        for c in concepts:
                            st.markdown(f"- {c}")
                    else:
                        st.caption("No concepts")

                with col2:
                    st.markdown("**⚙️ Algorithms**")
                    if algos:
                        for a in algos:
                            st.markdown(f"- {a}")
                    else:
                        st.caption("No algorithms")

# ------------------------------------------------------------
# QUESTIONS VIEW
# ------------------------------------------------------------
st.divider()
st.subheader("❓ Questions")

if st.button("📋 Show All Questions in Book"):
    questions = get_all_questions(book_id)

    if not questions:
        st.info("No questions found.")
    else:
        for q in questions:
            label = f"Q{q['qno']}" if q["qno"] else "Q"
            st.markdown(f"**{label}:** {q['question']}")
