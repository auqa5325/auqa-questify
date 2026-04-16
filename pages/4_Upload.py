import os
import mimetypes
from dotenv import load_dotenv
import streamlit as st
import boto3

load_dotenv()

# Configuration from environment (ensure these exist in your .env)
AWS_REGION = os.environ.get("AWS_REGION")
S3_BUCKET = os.environ.get("S3_BUCKET")

st.title("📤 Upload a local file to S3")
st.markdown("Choose a file from your computer and upload it directly to the configured S3 bucket.")

if not AWS_REGION or not S3_BUCKET:
    st.error("AWS_REGION and S3_BUCKET must be set in your environment (.env).")
    st.stop()

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

            # Create boto3 session and client
            session = boto3.Session(region_name=AWS_REGION)
            s3_client = session.client("s3")

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