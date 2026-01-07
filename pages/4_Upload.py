import os
import mimetypes
from dotenv import load_dotenv
import streamlit as st
import boto3
import pandas as pd
from datetime import datetime

load_dotenv()

# Configuration from environment (ensure these exist in your .env)
AWS_REGION = os.environ.get("AWS_REGION")
S3_BUCKET = os.environ.get("S3_BUCKET")

st.title("📤 Upload a local file to S3")
st.markdown("Choose a file from your computer and upload it directly to the configured S3 bucket.")

# Add a section for listing S3 files
st.markdown("---")
st.subheader("📋 List Files in S3 Bucket")

if not AWS_REGION or not S3_BUCKET:
    st.error("AWS_REGION and S3_BUCKET must be set in your environment (.env).")
    st.stop()

# Create S3 client for file listing
s3_client = boto3.client("s3", region_name=AWS_REGION)

# File listing functionality
if st.button("📋 List All Files in S3 Bucket"):
    try:
        st.info(f"Fetching files from bucket: {S3_BUCKET}")
        
        # List all objects in the bucket
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=S3_BUCKET)
        
        files_data = []
        total_size = 0
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    # Convert size to human readable format
                    size_bytes = obj['Size']
                    if size_bytes == 0:
                        size_str = "0 B"
                    elif size_bytes < 1024:
                        size_str = f"{size_bytes} B"
                    elif size_bytes < 1024**2:
                        size_str = f"{size_bytes/1024:.1f} KB"
                    elif size_bytes < 1024**3:
                        size_str = f"{size_bytes/(1024**2):.1f} MB"
                    else:
                        size_str = f"{size_bytes/(1024**3):.1f} GB"
                    
                    # Format last modified date
                    last_modified = obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S UTC')
                    
                    files_data.append({
                        'File Name': obj['Key'],
                        'Size': size_str,
                        'Last Modified': last_modified,
                        'Storage Class': obj.get('StorageClass', 'STANDARD')
                    })
                    total_size += size_bytes
        
        if files_data:
            # Create DataFrame and display
            df = pd.DataFrame(files_data)
            
            # Calculate total size in human readable format
            if total_size == 0:
                total_size_str = "0 B"
            elif total_size < 1024**3:
                total_size_str = f"{total_size/(1024**2):.1f} MB"
            else:
                total_size_str = f"{total_size/(1024**3):.1f} GB"
            
            st.success(f"Found {len(files_data)} files (Total size: {total_size_str})")
            
            # Display the table
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "File Name": st.column_config.TextColumn(
                        "File Name",
                        help="S3 object key (path in bucket)",
                        width="large"
                    ),
                    "Size": st.column_config.TextColumn(
                        "Size",
                        help="File size",
                        width="small"
                    ),
                    "Last Modified": st.column_config.TextColumn(
                        "Last Modified",
                        help="When the file was last updated",
                        width="medium"
                    ),
                    "Storage Class": st.column_config.TextColumn(
                        "Storage Class",
                        help="S3 storage class",
                        width="small"
                    )
                }
            )
            
            # Add download links for each file
            st.subheader("🔗 File URLs")
            for file_info in files_data:
                file_key = file_info['File Name']
                s3_url = f"s3://{S3_BUCKET}/{file_key}"
                http_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{file_key}"
                
                with st.expander(f"📄 {file_key}"):
                    st.code(f"S3 URI: {s3_url}")
                    st.code(f"HTTP URL: {http_url}")
                    
                    # Add copy buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"📋 Copy S3 URI", key=f"s3_{file_key}"):
                            st.write("S3 URI copied to clipboard!")
                    with col2:
                        if st.button(f"📋 Copy HTTP URL", key=f"http_{file_key}"):
                            st.write("HTTP URL copied to clipboard!")
        else:
            st.info("No files found in the S3 bucket.")
            
    except Exception as e:
        st.error(f"Failed to list files: {e}")

st.markdown("---")
st.subheader("📤 Upload New File")

uploaded_file = st.file_uploader("Choose a file to upload", accept_multiple_files=False)

default_key = None
if uploaded_file is not None:
    default_key = uploaded_file.name

key = st.text_input("S3 object key (path in bucket)", value=default_key or "")
public = st.checkbox("Make object publicly readable (ACL=public-read)", value=False)

if st.button("Upload"):
    if uploaded_file is None:
        st.error("Please choose a file first.")
    elif not key:
        st.error("Please provide an S3 object key (destination path).")
    else:
        try:
            # Reset file pointer and compute file size
            try:
                uploaded_file.seek(0, os.SEEK_END)
                file_size = uploaded_file.tell()
                uploaded_file.seek(0)
            except Exception:
                # fallback: read buffer length
                buf = uploaded_file.getbuffer()
                file_size = len(buf)
                uploaded_file.seek(0)

            # Detect content type
            content_type, _ = mimetypes.guess_type(key)
            if not content_type:
                # fallback to the file's underlying name
                content_type, _ = mimetypes.guess_type(uploaded_file.name)
            extra_args = {}
            if content_type:
                extra_args["ContentType"] = content_type
            if public:
                extra_args["ACL"] = "public-read"

            # Use the existing S3 client

            # Progress helper
            progress_bar = st.progress(0)
            status_text = st.empty()

            class ProgressPercentage:
                def __init__(self, size, progress_bar, status_text):
                    self._size = float(size) if size else 1.0
                    self._seen = 0
                    self._progress_bar = progress_bar
                    self._status_text = status_text

                def __call__(self, bytes_amount):
                    self._seen += bytes_amount
                    pct = int((self._seen / self._size) * 100)
                    pct = min(100, max(0, pct))
                    try:
                        self._progress_bar.progress(pct)
                        self._status_text.text(f"Uploaded {self._seen} / {int(self._size)} bytes ({pct}%)")
                    except Exception:
                        pass

            callback = ProgressPercentage(file_size, progress_bar, status_text)

            # Ensure file pointer at start
            try:
                uploaded_file.seek(0)
            except Exception:
                pass

            # Use the boto3 client's upload_fileobj (more compatible than S3Transfer.upload_fileobj)
            s3_client.upload_fileobj(
                Fileobj=uploaded_file,
                Bucket=S3_BUCKET,
                Key=key,
                ExtraArgs=extra_args or None,
                Callback=callback,
            )

            # Success
            progress_bar.progress(100)
            status_text.text("Upload complete")

            s3_url = f"s3://{S3_BUCKET}/{key}"
            http_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"
            st.success("File uploaded to S3")
            st.write("S3 URI:", s3_url)
            st.write("HTTP URL:", http_url)

            if public:
                st.info("Object uploaded with public-read ACL. The HTTP URL should be accessible if bucket policy allows it.")

        except Exception as e:
            st.error(f"Upload failed: {e}")
            raise