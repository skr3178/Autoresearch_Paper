"""Quick test of the z.ai GLM API connection."""
import json
import urllib.request
from intelligence_config import IntelligenceConfig, load_repo_env

import os
for v in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "AUTORESEARCH_INTELLIGENCE_MODEL"):
    os.environ.pop(v, None)

load_repo_env()
cfg = IntelligenceConfig.from_env()

print(f"model:    {cfg.model}")
print(f"base_url: {cfg.base_url}")
print(f"api_key:  {cfg.api_key[:8]}...")

body = json.dumps({
    "model": cfg.model,
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 16,
    "temperature": 0.3,
}).encode()

req = urllib.request.Request(
    f"{cfg.base_url}/chat/completions",
    data=body,
    headers={
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    },
)

with urllib.request.urlopen(req, timeout=30) as resp:
    data = json.loads(resp.read())

print("response:", data["choices"][0]["message"]["content"])
