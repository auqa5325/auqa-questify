import os
import json
import pathlib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from qp_pdf_generator import render_qp_pdf   # <- this is now Playwright-based

# ---------------- Step 2: Generate Question Paper PDF ----------------
st.markdown("---")
st.header("📄 Generate Final Question Paper")

# Example data (fallback if no LLM generation is done yet)
default_data = {
"title": "ANNA UNIVERSITY (UNIVERSITY DEPARTMENTS)",
"stream": "B.E. /B. Tech / B. Arch (Full Time)",
"exam_title": "END SEMESTER EXAMINATIONS,",
"exam_session": "NOV/DEC 2025",
"course": "COMPUTER SCIENCE AND ENGINEERING",
"semester": "VII / VIII",
"subject_code": "CN",
"subject_name": "Computer Networks",
"department": "Computer Technology",
"date": "DD/MM/YYYY",
"time": "3 Hours",
"max_marks": 100,
"course_outcomes": [
{"code": "CO1", "text": "Understand fundamental networking concepts"},
{"code": "CO2", "text": "Analyze transport protocols"},
{"code": "CO3", "text": "Apply routing and addressing techniques"},
{"code": "CO4", "text": "Design error control and multiple access methods"},
{"code": "CO5", "text": "Evaluate SDN and NFV in modern networks"}
],
"questions": [
{"QNo": 1, "Section": "Part A", "Marks": 2, "Unit": "1", "CO": "CO1", "BL": "L1", "SUB": None, "Qn": "Define protocol layering in computer networks."},
{"QNo": 2, "Section": "Part A", "Marks": 2, "Unit": "2", "CO": "CO2", "BL": "L2", "SUB": None, "Qn": "Explain why UDP is preferred over TCP for DNS applications."},
{"QNo": 3, "Section": "Part A", "Marks": 2, "Unit": "3", "CO": "CO3", "BL": "L3", "SUB": None, "Qn": "Calculate the subnet address for the IP address 192.168.10.50/26."},
{"QNo": 4, "Section": "Part A", "Marks": 2, "Unit": "4", "CO": "CO4", "BL": "L1", "SUB": None, "Qn": "List the four main services that can be provided by a link-layer protocol."},
{"QNo": 5, "Section": "Part A", "Marks": 2, "Unit": "5", "CO": "CO5", "BL": "L2", "SUB": None, "Qn": "Explain the role of network functions virtualization (NFV) in modern networking."},
{"QNo": 11, "Section": "Part B", "Marks": 7, "Unit": "1", "CO": "CO1", "BL": "L3", "SUB": "a", "Qn": "Explain the five layers of the Internet protocol stack and describe the function of each layer."},
{"QNo": 11, "Section": "Part B", "Marks": 6, "Unit": "1", "CO": "CO1", "BL": "L3", "SUB": "b", "Qn": "Compare the Internet protocol stack with the OSI reference model, highlighting the key differences."},
{"QNo": 12, "Section": "Part B", "Marks": 7, "Unit": "2", "CO": "CO2", "BL": "L4", "SUB": "a", "Qn": "Analyze the advantages and disadvantages of UDP compared to TCP for multimedia applications."},
{"QNo": 12, "Section": "Part B", "Marks": 6, "Unit": "2", "CO": "CO2", "BL": "L4", "SUB": "b", "Qn": "Compare the service models provided by UDP and TCP in terms of reliability and connection management."},
{"QNo": 16, "Section": "Part C", "Marks": 8, "Unit": "1", "CO": "CO4", "BL": "L5", "SUB": "a", "Qn": "Evaluate the advantages and disadvantages of the layered architecture approach in network protocol design."},
{"QNo": 16, "Section": "Part C", "Marks": 7, "Unit": "1", "CO": "CO4", "BL": "L5", "SUB": "b", "Qn": "Justify why the Internet architecture places much of its complexity at the network edges rather than in the core."}
]
}


# Either use parsed/generated data OR fallback to default_data
qp_data = st.session_state.get(default_data)

# Render the PDF UI (uses Playwright internally now)
render_qp_pdf(default_data, template_name="template2.html", title="CS23303 — Generate Question Paper PDF")
