"""
Test script: pass PDF natively to gpt-5.2 and let it parse it.
No preprocessing — PDF sent as-is via OpenAI file input.

Approach 1: base64 inline (input_file + file_data)
Approach 2: upload once via Files API, query with file_id (better for repeated queries)
"""

import base64
import os
import sys

PDF_PATH = "paper/CarPlanner.pdf"
QUESTION = (
    "Answer these specific questions using exact quotes and equation numbers from the paper:\n"
    "1. What is the exact formula for the consistency loss? Give the equation number.\n"
    "2. What value does the paper use for the PPO clip parameter epsilon? Which section or table?\n"
    "3. What is the entropy coefficient used during training? Exact value and where it appears.\n"
    "4. In Figure 2, what are the named components and how does data flow between them?\n"
    "If any of these are not in the paper, say so explicitly."
)

# ── Load env ──────────────────────────────────────────────────────────────────
api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
model = os.environ.get("AUTORESEARCH_INTELLIGENCE_MODEL", "gpt-5.2")

print(f"Model   : {model}")
print(f"Base URL: {base_url}")
print(f"PDF     : {PDF_PATH}  ({os.path.getsize(PDF_PATH)/1024/1024:.1f} MB)\n")

from openai import OpenAI
client = OpenAI(api_key=api_key, base_url=base_url)

# ══════════════════════════════════════════════════════════════════════════════
# Approach 1: base64 inline — PDF sent as file_data in a single request
# ══════════════════════════════════════════════════════════════════════════════
def test_base64_inline():
    print("=" * 60)
    print("Approach 1: base64 inline (responses API, input_file)")
    print("=" * 60)
    with open(PDF_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    try:
        resp = client.responses.create(
            model=model,
            input=[{
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": "CarPlanner.pdf",
                        "file_data": f"data:application/pdf;base64,{b64}",
                    },
                    {
                        "type": "input_text",
                        "text": QUESTION,
                    },
                ],
            }],
            max_output_tokens=1024,
        )
        print("SUCCESS")
        print(resp.output_text[:2000])
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Approach 2: upload once via Files API, reuse file_id across turns
# Best for repeated queries — upload cost paid once
# ══════════════════════════════════════════════════════════════════════════════
def test_file_upload():
    print("=" * 60)
    print("Approach 2: Files API upload + file_id")
    print("=" * 60)
    try:
        with open(PDF_PATH, "rb") as f:
            file_obj = client.files.create(file=f, purpose="user_data")
        file_id = file_obj.id
        print(f"Uploaded — file_id: {file_id}")

        resp = client.responses.create(
            model=model,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": file_id},
                    {"type": "input_text", "text": QUESTION},
                ],
            }],
            max_output_tokens=1024,
        )
        print("SUCCESS")
        print(resp.output_text[:2000])

        client.files.delete(file_id)
        print(f"\nCleaned up file_id: {file_id}")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("1", "all"):
        test_base64_inline()
    if which in ("2", "all"):
        test_file_upload()
