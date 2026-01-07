import base64
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
from playwright.sync_api import sync_playwright
import streamlit as st

def render_qp_pdf(data: dict, template_name: str = "template3.html", title: str = "CS23303 Paper → PDF"):
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
    with st.spinner("Rendering PDF with Playwright (Chromium)..."):
                with sync_playwright() as p:
                    browser = p.chromium.launch()  # add headless=True if needed
                    page = browser.new_page()
                    # Provide a base URL so relative links (css/img) resolve
                    page.set_content(html_out, wait_until="networkidle")
                    # Generate PDF bytes. adjust margin/format as needed
                    pdf_bytes = page.pdf(format="A4", print_background=True)
                    browser.close()
    
    if st.button("PDF Preview"):
            # Embed PDF in Streamlit
            b64 = base64.b64encode(pdf_bytes).decode()
            pdf_display = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="800"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

    st.download_button(
        label="⬇️ Download PDF",
        data=pdf_bytes,
        file_name="question_paper.pdf",
        mime="application/pdf"
    )

    

# Example usage:
# render_qp_pdf(exam_data_dict, template_name="question_paper_template.html", title="Sample Paper")
