import streamlit as st

st.set_page_config(page_title="AUQA Questify", page_icon="📘", layout="wide")

st.title("📘 AUQA Questify")
st.markdown("""
Welcome to **AUQA Questify** 🚀

Use the sidebar to navigate:
- **Ingestion** → Upload course PDFs from S3, extract with Textract, embed with Titan, and store in OpenSearch.  
- **Generation** → Search ingested docs, apply hybrid retrieval, and generate practice questions using Anthropic, Llama, or Mistral models.  
""")
