# AUQA Questify

Streamlit app for:

- ingesting syllabus/course PDFs from S3
- extracting text with Textract
- indexing chunks into OpenSearch
- generating questions and question papers with Amazon Bedrock

## Clone

```bash
git clone <your-repo-url>
cd auqa-questify
```

## Create `.env`

Create a `.env` file in the project root:

```env
AWS_REGION=ap-south-1
S3_BUCKET=your-s3-bucket
OS_DOMAIN=your-opensearch-domain
OS_INDEX=test-auqa
INDEX_NAME=test-auqa
AUQA_OUT_DIR=/tmp/auqa_output
```

Notes:

- `AWS_REGION`, `S3_BUCKET`, and `OS_DOMAIN` are required for the main flows.
- `OS_INDEX` is used by `pages/2_Generation.py`.
- `INDEX_NAME` is used by `pages/3_QuestionPaper.py`.
- Set both `OS_INDEX` and `INDEX_NAME` to the same OpenSearch index unless you intentionally want different ones.

## Create a Virtual Environment

```bash
python3 -m venv myvenv
source myvenv/bin/activate
python -m pip install --upgrade pip
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

Optional: install Playwright browser binaries for PDF preview/download in the Question Paper page:

```bash
python -m playwright install chromium
```

If you skip this, the app still runs, but PDF rendering will show a graceful error message instead of crashing.

## Run the App

```bash
streamlit run Home.py
```

Then open the local URL shown by Streamlit, usually:

```txt
http://localhost:8501
```

## AWS Prerequisites

Make sure your AWS credentials are available locally before running the app. For example:

- `aws configure`
- exported environment variables
- an IAM role/session if you are running in AWS

Your AWS identity needs access to:

- Amazon Bedrock
- Amazon S3
- Amazon Textract
- Amazon OpenSearch Service

## Main Pages

- `Home.py`: app entry page
- `pages/1_Ingestion.py`: ingest and index content
- `pages/2_Generation.py`: retrieve chunks and generate outputs
- `pages/3_QuestionPaper.py`: generate question papers and export PDF
- `pages/4_Upload.py`: upload flow

## Troubleshooting

If `playwright` is missing:

```bash
pip install playwright
python -m playwright install chromium
```
