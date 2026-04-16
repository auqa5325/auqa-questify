import base64
import json
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
import streamlit as st

@st.cache_data(show_spinner=False)
def _render_pdf_bytes(html_out: str) -> bytes:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html_out, wait_until="networkidle")
        pdf_bytes = page.pdf(format="A4", print_background=True)
        browser.close()
    return pdf_bytes


def render_qp_pdf(data: dict, template_name: str = "template2.html", title: str = "CS23303 Paper → PDF"):
    """
    Render a question paper from JSON data using Jinja2 + Playwright (headless Chromium),
    preview it in Streamlit, and provide a download button.

    IMPORTANT: template must reference variables as top-level keys, e.g. {{ title }}, {{ questions }},
    not as {{ data.title }} if we call template.render(**data).
    """
    BASE = Path(__file__).parent
    templates_dir = BASE / "templates"

    env = Environment(
        loader=FileSystemLoader(templates_dir),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    st.set_page_config(page_title=title)
    st.title(f"📄 {title}")

    # Load template
    template = env.get_template(template_name)

    # Render template: UNPACK data so template sees top-level variables
    try:
        html_out = template.render(**data)
    except Exception as e:
        st.error(f"Template rendering error: {e}")
        # Show partial context for debugging
        st.write("Data keys passed to template:", list(data.keys()))
        return

    # Debug helper: show first chunk of rendered HTML so you can verify placeholders replaced
    #st.subheader("Rendered HTML (preview, first 2KB)")
    #st.code(html_out[:2048], language="html")

    # Preview PDF inline
    #st.subheader("Preview")
    try:
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError:
        st.error(
            "PDF rendering requires the optional `playwright` dependency. "
            "Install it with `python3 -m pip install playwright` and "
            "`python3 -m playwright install chromium`."
        )
        return

    with st.spinner("Rendering PDF with Playwright (Chromium)..."):
        try:
            pdf_bytes = _render_pdf_bytes(html_out)
        except Exception as e:
            st.error(f"PDF rendering failed: {e}")
            st.info("If Chromium is not installed yet, run `python3 -m playwright install chromium`.")
            return

    state_key = f"pdf_preview_visible::{template_name}::{title}"
    if state_key not in st.session_state:
        st.session_state[state_key] = False

    controls_col1, controls_col2 = st.columns(2)
    with controls_col1:
        preview_label = "Hide PDF Preview" if st.session_state[state_key] else "Show PDF Preview"
        if st.button(preview_label, key=f"{state_key}::toggle"):
            st.session_state[state_key] = not st.session_state[state_key]
    with controls_col2:
        file_stub = data.get("subject_code") or "question_paper"
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes,
            file_name=f"{file_stub}.pdf",
            mime="application/pdf",
            key=f"{state_key}::download::{hash(json.dumps(data, sort_keys=True, default=str))}"
        )

    if st.session_state[state_key]:
        b64 = base64.b64encode(pdf_bytes).decode()
        pdf_display = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="800"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

    

# Example usage:
# render_qp_pdf(exam_data_dict, template_name="question_paper_template.html", title="Sample Paper")
