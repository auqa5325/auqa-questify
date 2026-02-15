import json
from typing import Any, Dict, List

import streamlit as st

from kg.questindex_core import (
    ALL_CONTENT_LABELS,
    QUESTINDEX_VECTOR_INDEX,
    build_questindex_graph,
    build_schema_context,
    explain_read_cypher,
    extract_pdf_text,
    filter_pages_by_range,
    generate_guarded_cypher,
    get_books,
    get_book_structure_version,
    get_chapters,
    get_sections,
    get_validated_default_contract,
    list_books_folder_pdfs,
    repair_cypher_contract,
    run_read_cypher,
    setup_constraints,
    synthesize_structured_answer,
    validate_cypher_contract,
)


st.set_page_config(layout="wide")
st.title("QuestIndex KG Builder + GraphRAG")

if "questindex_pages" not in st.session_state:
    st.session_state.questindex_pages = {}
if "questindex_last_structure" not in st.session_state:
    st.session_state.questindex_last_structure = None
if "questindex_last_diagnostics" not in st.session_state:
    st.session_state.questindex_last_diagnostics = None
if "questindex_last_relation_report" not in st.session_state:
    st.session_state.questindex_last_relation_report = None
if "questindex_last_build_stats" not in st.session_state:
    st.session_state.questindex_last_build_stats = None
if "questindex_last_vector_stats" not in st.session_state:
    st.session_state.questindex_last_vector_stats = None
if "questindex_last_vector_verify" not in st.session_state:
    st.session_state.questindex_last_vector_verify = None
if "questindex_last_rows" not in st.session_state:
    st.session_state.questindex_last_rows = None

with open("models.json", "r", encoding="utf-8") as f:
    MODELS = json.load(f)

model_names = [m["name"] for m in MODELS]

build_tab, qa_tab, diag_tab = st.tabs(["Build Graph", "Graph QA", "Diagnostics"])

with build_tab:
    st.subheader("Build QuestIndex Graph")

    col1, col2 = st.columns(2)
    with col1:
        source_option = st.radio(
            "PDF Source",
            ["Upload", "Books Folder", "Server Path"],
            horizontal=True,
            key="quest_source_option",
        )

        uploaded_pdf = None
        selected_books_pdf = None
        pdf_local_path = ""

        if source_option == "Upload":
            uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"], key="quest_upload_pdf")
        elif source_option == "Books Folder":
            books_pdfs = list_books_folder_pdfs()
            if books_pdfs:
                selected_books_pdf = st.selectbox("Select PDF from books/", books_pdfs, key="quest_books_pdf")
            else:
                st.warning("No PDF files found in books/ folder.")
        else:
            pdf_local_path = st.text_input("PDF path on server", "", key="quest_path_pdf")

    with col2:
        book_id = st.text_input("Book ID", "DBMS_QuestIndex", key="quest_book_id")
        pages_per_chunk = st.slider("Pages per chunk", 1, 24, 4, key="quest_pages_per_chunk")
        overlap = st.slider("Overlap pages", 0, pages_per_chunk - 1, min(1, pages_per_chunk - 1), key="quest_overlap")
        page_start = st.number_input("Start Page", min_value=1, value=1, step=1, key="quest_page_start")
        page_end = st.number_input("End Page (0 = last page)", min_value=0, value=0, step=1, key="quest_page_end")
        toc_scan_pages = st.slider("TOC Scan Pages", 5, 40, 25, key="quest_toc_scan")
        max_workers = st.slider("Concurrent Workers", 1, 8, 3, key="quest_workers")
        skip_summaries = st.checkbox("Skip summaries (faster build)", value=False, key="quest_skip_summaries")
        skip_section_summaries = st.checkbox(
            "Skip section summaries only (keep chapter summaries)",
            value=False,
            key="quest_skip_section_summaries",
            disabled=skip_summaries,
        )
        enable_vector_indexing = st.checkbox(
            "Also index chunk embeddings in OpenSearch",
            value=True,
            key="quest_enable_vector_index",
        )
        vector_index_name = st.text_input(
            "OpenSearch vector index",
            QUESTINDEX_VECTOR_INDEX,
            key="quest_vector_index_name",
        )
        extraction_mode_label = st.radio(
            "Extraction Backend",
            ["Local (PyMuPDF, no S3)", "S3 Cache + Textract OCR"],
            index=0,
            key="quest_extraction_backend",
        )

    selected_build_model = st.selectbox("Extraction + Summary Model", model_names, key="quest_build_model")
    model_id_build = next(m["id"] for m in MODELS if m["name"] == selected_build_model)

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Extract PDF Text", key="quest_extract_pdf"):
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
                    with st.spinner("Extracting PDF text..."):
                        pages = extract_pdf_text(pdf_source, source_kind, book_id, extraction_mode)
                        pages, range_info = filter_pages_by_range(pages, int(page_start), int(page_end))

                        if not pages:
                            if range_info:
                                raise Exception(
                                    "No pages in selected range "
                                    f"{range_info['applied_start']}-{range_info['applied_end']} "
                                    f"(available {range_info['available_min']}-{range_info['available_max']})"
                                )
                            raise Exception("No pages extracted from PDF")

                        st.session_state.questindex_pages = pages

                        if range_info:
                            st.info(
                                f"Loaded page range {range_info['applied_start']}-{range_info['applied_end']} "
                                f"(available {range_info['available_min']}-{range_info['available_max']})"
                            )

                        st.success(f"Extracted {len(pages)} pages for build")

                except FileNotFoundError:
                    st.error("PDF not found.")
                except Exception as exc:
                    st.error(f"PDF extraction failed: {exc}")

    with c2:
        if st.button("Build QuestIndex Graph", key="quest_build_graph"):
            if not st.session_state.questindex_pages:
                st.error("Extract PDF text first")
            else:
                try:
                    setup_constraints()
                    with st.spinner("Building TOC-first graph, extracting entities, writing Neo4j..."):
                        build_output = build_questindex_graph(
                            pages=st.session_state.questindex_pages,
                            book_id=book_id,
                            model_id=model_id_build,
                            selected_model=selected_build_model,
                            toc_scan_pages=toc_scan_pages,
                            pages_per_chunk=pages_per_chunk,
                            overlap=overlap,
                            max_workers=max_workers,
                            skip_summaries=skip_summaries,
                            skip_section_summaries=skip_section_summaries,
                            enable_vector_indexing=enable_vector_indexing,
                            vector_index_name=vector_index_name,
                        )

                    st.session_state.questindex_last_structure = build_output["structure"]
                    st.session_state.questindex_last_diagnostics = build_output["diagnostics"]
                    st.session_state.questindex_last_relation_report = build_output["consolidated"].get("relationship_report", {})
                    st.session_state.questindex_last_build_stats = build_output["stats"]
                    st.session_state.questindex_last_vector_stats = build_output.get("vector_stats")
                    st.session_state.questindex_last_vector_verify = build_output.get("vector_verify")

                    stats = build_output["stats"]
                    summary_status = build_output.get("summary_status", {})
                    st.success("QuestIndex graph build completed")

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Chapters", stats.get("chapters", 0))
                    m2.metric("Sections", stats.get("sections", 0))
                    m3.metric("Subsections", stats.get("subsections", 0))
                    m4.metric("Mentions Expected", stats.get("mentions_expected", 0))
                    m5.metric("Semantic Edges", stats.get("semantic_relationships", 0))

                    with st.expander("Entity Counts"):
                        entity_counts = {label: stats.get(label, 0) for label in ALL_CONTENT_LABELS}
                        st.json(entity_counts)

                    with st.expander("Build Stats"):
                        st.json(stats)

                    diagnostics_payload = build_output.get("diagnostics", {})
                    normalization_payload = diagnostics_payload.get("normalization", {})
                    if normalization_payload:
                        st.markdown("**Normalization (v2)**")
                        n1, n2, n3, n4 = st.columns(4)
                        n1.metric("Dropped entries", normalization_payload.get("dropped_entries_count", 0))
                        n2.metric("Relabelled chapters", normalization_payload.get("relabelled_chapters_count", 0))
                        n3.metric("Relabelled sections", normalization_payload.get("relabelled_sections_count", 0))
                        n4.metric("Orphans fixed", normalization_payload.get("orphan_sections_fixed", 0) + normalization_payload.get("orphan_subsections_fixed", 0))
                        st.caption(f"Structure version: {diagnostics_payload.get('structure_version', 'v2_normalized')}")

                    if summary_status.get("skip_summaries"):
                        st.info("Summary extraction skipped.")
                    elif summary_status.get("skip_section_summaries"):
                        st.success(
                            f"Chapter summaries generated: {summary_status.get('chapter_summaries', 0)}. "
                            "Section summaries skipped."
                        )
                    else:
                        st.success(
                            f"Summary extraction complete: "
                            f"{summary_status.get('chapter_summaries', 0)} chapter summaries, "
                            f"{summary_status.get('section_summaries', 0)} section summaries."
                        )

                    vector_stats = build_output.get("vector_stats")
                    vector_verify = build_output.get("vector_verify")
                    if vector_stats is not None:
                        st.subheader("OpenSearch Embedding Index")
                        v1, v2, v3 = st.columns(3)
                        v1.metric("Indexed", vector_stats.get("indexed", 0))
                        v2.metric("Skipped", vector_stats.get("skipped", 0))
                        v3.metric("Errors", vector_stats.get("errors", 0))
                        st.caption(f"Index: {vector_stats.get('index_name', '')}")
                        if vector_stats.get("error"):
                            st.warning(f"Indexing warning: {vector_stats['error']}")

                        if vector_verify is not None:
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Verified (sample)", "Yes" if vector_verify.get("verified") else "No")
                            c2.metric("Docs in index (book)", vector_verify.get("total_docs_for_book", 0))
                            c3.metric("Sampled with vector", vector_verify.get("sampled_with_vector", 0))
                            c4.metric("KG chunk-id matches", vector_verify.get("kg_chunk_id_matches", 0))
                            st.caption(
                                f"Verify scope -> book_id: {vector_verify.get('verified_book_id', '')} | "
                                f"source: {vector_verify.get('verified_source', '')} | "
                                f"mode: {vector_verify.get('verification_mode', '')}"
                            )
                            if vector_verify.get("total_docs_source_any_book", 0):
                                st.caption(
                                    f"Docs for source across books: {vector_verify.get('total_docs_source_any_book', 0)}"
                                )
                            indexed_books = vector_verify.get("indexed_book_ids_for_source") or []
                            if indexed_books:
                                st.caption("Indexed book_ids for source 8_QuestIndex:")
                                st.code("\n".join(indexed_books[:20]))
                            if vector_verify.get("sample_chunk_ids"):
                                st.caption("Sample chunk_ids (same IDs used in KG content nodes):")
                                st.code("\n".join(vector_verify.get("sample_chunk_ids", [])[:10]))
                            if vector_verify.get("verified"):
                                st.success("Chunk embedding verification passed.")
                            else:
                                st.warning("Chunk embedding verification failed for sampled documents.")
                                if (
                                    vector_verify.get("total_docs_for_book", 0) == 0
                                    and vector_verify.get("total_docs_source_any_book", 0) > 0
                                ):
                                    st.info("Likely book_id mismatch: index has 8_QuestIndex docs, but not for selected book_id.")
                            if vector_verify.get("error"):
                                st.warning(f"Embedding verification warning: {vector_verify['error']}")
                            elif vector_verify.get("knn_probe_error"):
                                st.info(f"kNN probe warning: {vector_verify['knn_probe_error']}")

                except Exception as exc:
                    st.error(f"Build failed: {exc}")


with qa_tab:
    st.subheader("GraphRAG QA (Guarded Cypher)")

    books = get_books()
    if not books:
        st.info("No books found in Neo4j")
    else:
        qa_book_id = st.selectbox("Select Book", books, key="quest_qa_book")
        qa_structure_version = get_book_structure_version(qa_book_id)
        if qa_structure_version != "v2_normalized":
            st.warning(
                f"Selected book uses structure version '{qa_structure_version}'. "
                "Rebuild this book in Page 8 to get robust normalized retrieval."
            )
        else:
            st.caption(f"Structure version: {qa_structure_version}")
        chapter_rows = get_chapters(qa_book_id)
        chapter_options = ["All"] + [f"{row['chapter_no']} — {row['title']}" for row in chapter_rows]
        chapter_choice = st.selectbox("Chapter Scope", chapter_options, key="quest_qa_chapter")
        selected_chapter_no = None
        if chapter_choice != "All":
            selected_chapter_no = chapter_choice.split(" — ", 1)[0]

        section_rows = get_sections(qa_book_id, selected_chapter_no)
        section_options = ["All"] + [f"{row['section_no']} — {row['title']}" for row in section_rows]
        section_choice = st.selectbox("Section Scope", section_options, key="quest_qa_section")
        selected_section_no = None
        if section_choice != "All":
            selected_section_no = section_choice.split(" — ", 1)[0]

        selected_qa_model = st.selectbox("Cypher + Answer Model", model_names, key="quest_qa_model")
        model_id_qa = next(m["id"] for m in MODELS if m["name"] == selected_qa_model)

        user_query = st.text_area("Ask the graph", "", height=120, key="quest_qa_query")

        if st.button("Ask Graph", key="quest_qa_ask"):
            question = user_query.strip()
            if not question:
                st.error("Enter a query")
            else:
                scope = {
                    "book_id": qa_book_id,
                    "chapter_no": selected_chapter_no,
                    "section_no": selected_section_no,
                }

                with st.spinner("Generating and executing guarded Cypher..."):
                    schema_context = build_schema_context(qa_book_id, selected_chapter_no, selected_section_no)
                    contract = generate_guarded_cypher(question, scope, schema_context, model_id_qa, selected_qa_model)
                    validated = validate_cypher_contract(contract, scope)
                    qa_notes: List[str] = []

                    if not validated["valid"]:
                        repaired = repair_cypher_contract(
                            question,
                            scope,
                            schema_context,
                            contract,
                            " ; ".join(validated["errors"]),
                            model_id_qa,
                            selected_qa_model,
                        )
                        validated = validate_cypher_contract(repaired, scope)

                    if not validated["valid"]:
                        qa_notes.append("Generated Cypher remained invalid after repair; using deterministic fallback query.")
                        validated = get_validated_default_contract(question, scope)

                    cypher = validated["cypher"]
                    params = validated["params"]
                    if "query" not in params:
                        params["query"] = question

                    # Pre-check syntax/semantics before actual execution.
                    explain_error = explain_read_cypher(cypher, params)
                    if explain_error:
                        repaired = repair_cypher_contract(
                            question,
                            scope,
                            schema_context,
                            {
                                "cypher": cypher,
                                "params": params,
                                "intent": validated.get("intent"),
                                "expected_columns": validated.get("expected_columns"),
                            },
                            explain_error,
                            model_id_qa,
                            selected_qa_model,
                        )
                        validated_repair = validate_cypher_contract(repaired, scope)
                        if validated_repair["valid"]:
                            cypher = validated_repair["cypher"]
                            params = validated_repair["params"]
                            if "query" not in params:
                                params["query"] = question
                            explain_error = explain_read_cypher(cypher, params)

                    if explain_error:
                        qa_notes.append(f"Generated Cypher failed EXPLAIN validation; fallback used. Error: {explain_error}")
                        validated = get_validated_default_contract(question, scope)
                        cypher = validated["cypher"]
                        params = validated["params"]
                        if "query" not in params:
                            params["query"] = question
                        fallback_explain_error = explain_read_cypher(cypher, params)
                        if fallback_explain_error:
                            st.error("Cypher validation failed, including fallback query.")
                            st.json(
                                {
                                    "generated_contract": contract,
                                    "generated_validation_errors": validated.get("errors", []),
                                    "fallback_explain_error": fallback_explain_error,
                                }
                            )
                            st.stop()

                    rows: List[Dict[str, Any]] = []
                    execution_error = None
                    try:
                        rows = run_read_cypher(cypher, params)
                    except Exception as exc:
                        execution_error = str(exc)

                    if execution_error:
                        qa_notes.append(f"Primary query execution failed; fallback used. Error: {execution_error}")
                        fallback_validated = get_validated_default_contract(question, scope)
                        fallback_cypher = fallback_validated["cypher"]
                        fallback_params = fallback_validated["params"]
                        if "query" not in fallback_params:
                            fallback_params["query"] = question
                        try:
                            rows = run_read_cypher(fallback_cypher, fallback_params)
                            cypher = fallback_cypher
                            params = fallback_params
                            execution_error = None
                        except Exception as fallback_exc:
                            execution_error = str(fallback_exc)

                    if execution_error:
                        st.error(f"Cypher execution failed after fallback: {execution_error}")
                        st.json({"cypher": cypher, "params": params, "notes": qa_notes})
                    else:
                        if qa_notes:
                            st.warning("\\n".join(qa_notes))

                        st.session_state.questindex_last_rows = rows
                        answer_json = synthesize_structured_answer(
                            question,
                            scope,
                            cypher,
                            rows,
                            model_id_qa,
                            selected_qa_model,
                        )
                        st.subheader("Structured JSON Response")
                        st.json(answer_json)

                        with st.expander("Cypher + Rows"):
                            st.code(cypher, language="cypher")
                            st.json({"params": params, "row_count": len(rows), "rows_preview": rows[:25], "notes": qa_notes})


with diag_tab:
    st.subheader("Diagnostics")

    diagnostics = st.session_state.get("questindex_last_diagnostics")
    structure = st.session_state.get("questindex_last_structure")
    relation_report = st.session_state.get("questindex_last_relation_report")
    build_stats = st.session_state.get("questindex_last_build_stats")
    vector_stats = st.session_state.get("questindex_last_vector_stats")
    vector_verify = st.session_state.get("questindex_last_vector_verify")

    if diagnostics:
        st.markdown("**TOC / Structure Diagnostics**")
        st.json(diagnostics)
    else:
        st.info("No build diagnostics available yet")

    if structure:
        st.markdown("**Extracted Structure**")
        ch_col, sec_col, sub_col = st.columns(3)
        with ch_col:
            st.caption("Chapters")
            st.dataframe(structure.get("chapters", []), use_container_width=True)
        with sec_col:
            st.caption("Sections")
            st.dataframe(structure.get("sections", []), use_container_width=True)
        with sub_col:
            st.caption("Subsections")
            st.dataframe(structure.get("subsections", []), use_container_width=True)

    if relation_report:
        st.markdown("**Relationship Report**")
        st.json(relation_report)

    if build_stats:
        st.markdown("**Last Build Stats**")
        st.json(build_stats)

    if vector_stats:
        st.markdown("**Vector Index Stats**")
        st.json(vector_stats)

    if vector_verify:
        st.markdown("**Vector Verification**")
        st.json(vector_verify)

    if st.session_state.get("questindex_last_rows") is not None:
        with st.expander("Last QA Rows"):
            st.json(st.session_state.questindex_last_rows[:25])
