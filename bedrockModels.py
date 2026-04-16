import json
from typing import Tuple, Dict, Any, Optional
import boto3
try:
    import tiktoken
except Exception:
    tiktoken = None


def count_tokens(text: str) -> int:
    """Return token count using cl100k_base if available, else fallback to word count."""
    if tiktoken:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return len(text.split())


def build_request(model_id: str, model_name: str, prompt: str, max_gen_len: int) -> Dict[str, Any]:
    """Build a request body tailored to the model family.

    Supports: amazon.nova, anthropic/claude, llama, mistral and generic models.
    The returned dict is suitable for passing to `bedrock.invoke_model(..., body=json.dumps(body))`
    for non-`converse` flows. For `converse`-style flows you should use `converse_call`.
    """
    # Amazon Nova family
    if model_id.startswith("amazon.nova"):
        return {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_gen_len, "temperature": 0.5, "topP": 0.9}
        }

    # Anthropic / Claude style
    if "anthropic" in model_id or "claude" in model_name.lower():
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_gen_len,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
        }

    # Llama family
    if "llama" in model_id:
        formatted_prompt = f"""
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
        return {"prompt": formatted_prompt, "max_gen_len": max_gen_len, "temperature": 0.5}

    # Mistral family
    if "mistral" in model_id:
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        return {"prompt": formatted_prompt, "max_tokens": max_gen_len, "temperature": 0.5}

    # Generic fallback
    return {"prompt": prompt, "max_tokens": max_gen_len, "temperature": 0.5}


def converse_call(client, model_id: str, user_message: str, max_tokens: int = 512, temperature: float = 0.5,
                  top_p: float = 0.9) -> Tuple[str, Dict[str, Any]]:
    """Call Bedrock `converse` and return (response_text, raw_response).

    Simple wrapper for chat-style single message usage.
    """
    conversation = [
        {"role": "user", "content": [{"text": user_message}]}
    ]

    try:
        resp = client.converse(
            modelId=model_id,
            messages=conversation,
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature, "topP": top_p},
        )

        # extract assistant text
        try:
            response_text = resp["output"]["message"]["content"][0]["text"]
        except Exception:
            response_text = json.dumps(resp)

        return response_text, resp

    except Exception as e:
        raise RuntimeError(f"ERROR: Can't invoke '{model_id}'. Reason: {e}") from e


# ------------------ Document understanding helpers ------------------

def converse_with_document(
    client,
    model_id: str,
    user_message: str,
    document_bytes: bytes,
    document_format: str = "pdf",
    document_name: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.5,
    top_p: float = 0.9,
) -> Tuple[str, Dict[str, Any]]:
    """Send a document together with a user message to Bedrock's `converse` API.

    - `document_bytes` should be the raw bytes of the document (PDF, DOCX, TXT, etc.).
    - The function constructs the `messages` payload including a document entry and calls `client.converse(...)`.

    Returns: (response_text, raw_response)
    Raises RuntimeError on failure with annotated message.
    """
    if not document_name:
        document_name = f"document.{document_format}"

    conversation = [
        {
            "role": "user",
            "content": [
                {"text": user_message},
                {
                    "document": {
                        "format": document_format,
                        "name": document_name,
                        "source": {"bytes": document_bytes},
                    }
                },
            ],
        }
    ]

    try:
        resp = client.converse(
            modelId=model_id,
            messages=conversation,
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature, "topP": top_p},
        )

        try:
            response_text = resp["output"]["message"]["content"][0]["text"]
        except Exception:
            response_text = json.dumps(resp)

        return response_text, resp

    except Exception as e:
        raise RuntimeError(f"ERROR: Can't invoke '{model_id}' with document. Reason: {e}") from e


def converse_with_document_from_s3(
    session_or_client,
    model_id: str,
    bucket: str,
    key: str,
    user_message: str,
    s3_client=None,
    document_format: str = "pdf",
    document_name: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.5,
    top_p: float = 0.9,
) -> Tuple[str, Dict[str, Any]]:
    """Fetch a document from S3 and call `converse_with_document`.

    `session_or_client` can be a boto3 Session or a pre-configured Bedrock client; if it's a
    Session the function will create both an S3 client and a bedrock-runtime client.
    """
    try:
        # derive s3 client
        if s3_client is None:
            if hasattr(session_or_client, "client"):
                s3 = session_or_client.client("s3")
            else:
                s3 = boto3.client("s3")
        else:
            s3 = s3_client

        # get object bytes
        obj = s3.get_object(Bucket=bucket, Key=key)
        document_bytes = obj["Body"].read()

        # derive bedrock client if needed
        if hasattr(session_or_client, "client"):
            bedrock_client = session_or_client.client("bedrock-runtime")
        else:
            bedrock_client = session_or_client

        return converse_with_document(
            bedrock_client,
            model_id=model_id,
            user_message=user_message,
            document_bytes=document_bytes,
            document_format=document_format,
            document_name=document_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    except Exception as e:
        raise RuntimeError(f"ERROR: Can't fetch document from s3://{bucket}/{key} or invoke model. Reason: {e}") from e


