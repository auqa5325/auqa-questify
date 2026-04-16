import os
import mimetypes
from dotenv import load_dotenv
import streamlit as st
import boto3
import pandas as pd

load_dotenv()

# Configuration
AWS_REGION = os.environ.get("AWS_REGION")
S3_BUCKET = os.environ.get("S3_BUCKET")

st.set_page_config(page_title="S3 File Explorer", layout="wide")
st.title("📂 S3 File Explorer")

if not AWS_REGION or not S3_BUCKET:
    st.error("AWS_REGION and S3_BUCKET must be set in your environment (.env).")
    st.stop()

s3_client = boto3.client("s3", region_name=AWS_REGION)

# --- SESSION STATE ---
if "current_path" not in st.session_state:
    st.session_state.current_path = ""  # Root
if "items_limit" not in st.session_state:
    st.session_state.items_limit = 20

def navigate_to(path):
    st.session_state.current_path = path
    st.session_state.items_limit = 20  # Reset pagination on folder change
    st.rerun()

def load_more():
    st.session_state.items_limit += 20

def format_size(size_bytes):
    if size_bytes == 0: return "0 B"
    elif size_bytes < 1024: return f"{size_bytes} B"
    elif size_bytes < 1024**2: return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3: return f"{size_bytes / (1024**2):.1f} MB"
    else: return f"{size_bytes / (1024**3):.1f} GB"

# --- BREADCRUMB NAVIGATION ---
# This replaces the "Back" buttons with clickable path segments
st.subheader("Navigation")
path_parts = st.session_state.current_path.strip("/").split("/")
breadcrumb_cols = st.columns(len(path_parts) + 1)

# Root Button
if breadcrumb_cols[0].button("🏠 Root", key="bc_root"):
    navigate_to("")

# Dynamic Breadcrumbs
current_acc = ""
for i, part in enumerate(path_parts):
    if part:
        current_acc += f"/{part}"
        # We use i+1 because index 0 was the Root button
        if breadcrumb_cols[i+1].button(f" {part} ❯", key=f"bc_{i}", help=f"Go to {current_acc}"):
            navigate_to(current_acc)

st.markdown("---")

# --- FILE LISTING LOGIC ---
st.subheader("📋 Contents")

try:
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=st.session_state.current_path, Delimiter='/')

    folders = []
    files = []

    for page in pages:
        if "CommonPrefixes" in page:
            for prefix in page["CommonPrefixes"]:
                folder_path = prefix["Prefix"]
                display_name = folder_path.replace(st.session_state.current_path, "").lstrip("/")
                folders.append({
                    "Type": "📁 Folder",
                    "Name": display_name,
                    "Full Path": folder_path,
                    "Last Modified": "-",
                    "Size": "-"
                })

        if "Contents" in page:
            for obj in page["Contents"]:
                if obj["Key"] == st.session_state.current_path:
                    continue
                files.append({
                    "Type": "📄 File",
                    "Name": obj["Key"].replace(st.session_state.current_path, "").lstrip("/"),
                    "Full Path": obj["Key"],
                    "Last Modified": obj["LastModified"].strftime("%Y-%m-%d %H:%M:%S"),
                    "Size": format_size(obj["Size"])
                })

    # Sort: Folders first, then files
    all_items = folders + files
    
    # Apply Pagination Limit
    display_items = all_items[:st.session_state.items_limit]

    if display_items:
        header_col1, header_col2, header_col3, header_col4 = st.columns([1, 3, 2, 2])
        header_col1.write("**Type**")
        header_col2.write("**Name**")
        header_col3.write("**Size**")
        header_col4.write("**Action**")
        st.divider()

        for item in display_items:
            col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
            col1.write(item["Type"])
            
            if item["Type"] == "📁 Folder":
                if col2.button(f"📂 {item['Name']}", key=f"f_{item['Full Path']}"):
                    navigate_to(item["Full Path"])
            else:
                col2.write(f"📄 {item['Name']}")
            
            col3.write(item["Size"])
            
            if item["Type"] == "📄 File":
                url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{item['Full Path']}"
                if col4.button("🔗 URL", key=f"url_{item['Full Path']}"):
                    st.info(f"URL: {url}")
            else:
                col4.write("")

        # --- LOAD MORE BUTTON ---
        if len(all_items) > st.session_state.items_limit:
            if st.button(f"⬇️ Load More ({len(all_items) - st.session_state.items_limit} more items)"):
                load_more()
                st.rerun()
    else:
        st.info("This directory is empty.")

except Exception as e:
    st.error(f"Error accessing S3: {e}")

# --- UPLOAD SECTION ---
st.markdown("---")
st.subheader("📤 Upload New File")

with st.expander("Click to expand upload tool"):
    uploaded_file = st.file_uploader("Choose a file")
    
    suggested_path = st.session_state.current_path
    if suggested_path and not suggested_path.endswith("/"):
        suggested_path += "/"
    
    filename = uploaded_file.name if uploaded_file else ""
    target_key = st.text_input("S3 object key (destination path)", value=f"{suggested_path}{filename}")
    public = st.checkbox("Make object publicly readable")

    if st.button("Upload to S3"):
        if uploaded_file is not None and target_key:
            try:
                content_type, _ = mimetypes.guess_type(target_key)
                if not content_type:
                    content_type, _ = mimetypes.guess_type(uploaded_file.name)
                
                extra_args = {"ContentType": content_type} if content_type else {}
                if public:
                    extra_args["ACL"] = "public-read"

                with st.spinner("Uploading..."):
                    s3_client.upload_fileobj(uploaded_file, S3_BUCKET, target_key, ExtraArgs=extra_args)
                st.success(f"Uploaded to {target_key}")
                st.rerun()
            except Exception as e:
                st.error(f"Upload failed: {e}")
        else:
            st.warning("Please select a file and provide a key.")