import base64
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML
import streamlit as st

def render_qp_pdf(data: dict, template_name: str = "qp.html", title: str = "CS23303 Paper → PDF"):
    """
    Render a question paper from JSON data using Jinja2 + WeasyPrint,
    preview it in Streamlit, and provide a download button.

    Args:
        data (dict): JSON-like dict with paper data
        template_name (str): name of the Jinja2 template in templates/
        title (str): Streamlit page title
    """
    BASE = Path(__file__).parent
    env = Environment(
        loader=FileSystemLoader(BASE / "templates"),
        autoescape=select_autoescape(["html", "xml"])
    )

    st.set_page_config(page_title=title)
    st.title(f"📄 {title}")

    # Render Jinja2 template
    template = env.get_template(template_name)
    html_out = template.render(data=data)

    # Preview PDF inline
    st.subheader("Preview")
    if st.button("Preview PDF"):
        with st.spinner("Rendering PDF..."):
            pdf_bytes = HTML(string=html_out, base_url=str(BASE)).write_pdf()
            b64 = base64.b64encode(pdf_bytes).decode()
            pdf_display = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="800"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes,
            file_name="question_paper.pdf",
            mime="application/pdf"
        )
