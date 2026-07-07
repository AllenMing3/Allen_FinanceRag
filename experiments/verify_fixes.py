"""Verify fixes before running full experiment"""
import os, sys

print("=" * 50)
print("Task 1: API Key Priority")
print("=" * 50)

# Replicate the exact logic from lightrag_experiment.py
API_KEY = ""
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(env_path):
    for line in open(env_path, encoding='utf-8'):
        if line.startswith("DASHSCOPE_API_KEY="):
            API_KEY = line.split("=", 1)[1].strip()
            break

if not API_KEY:
    API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

sys_env_key = os.environ.get("DASHSCOPE_API_KEY", "NOT SET")
print(f"  System env key:  {sys_env_key[:10]}...{sys_env_key[-4:]}" if len(sys_env_key) > 14 else f"  System env key:  {sys_env_key}")
print(f"  Script API_KEY:  {API_KEY[:10]}...{API_KEY[-4:]}")
print(f"  .env file key:   sk-b7204b1...5680")

if API_KEY.endswith("5680"):
    print("  PASS: Reading from .env file")
else:
    print("  FAIL: NOT reading from .env file!")
    sys.exit(1)

# Verify dashscope global override
import dashscope
dashscope.api_key = API_KEY
os.environ["DASHSCOPE_API_KEY"] = API_KEY
print(f"  dashscope.api_key after override: {dashscope.api_key[:10]}...{dashscope.api_key[-4:]}")

# Quick API test
from dashscope import Generation
r = Generation.call(
    model="qwen-plus",
    messages=[{"role": "user", "content": "reply with just 'ok'"}],
    api_key=API_KEY,
    result_format="message",
    max_tokens=5,
)
print(f"  API call: status={r.status_code}, code={getattr(r, 'code', 'ok')}")
if r.status_code == 200:
    print("  PASS: API key works")
else:
    print(f"  FAIL: {r.code} - {r.message}")
    sys.exit(1)

print()
print("=" * 50)
print("Task 2: Tokenizer Roundtrip")
print("=" * 50)

from lightrag.utils import Tokenizer

class _SimpleInner:
    _last_text: str = ""
    def encode(self, content, **kwargs):
        self._last_text = content
        return list(range(len(content)))
    def decode(self, tokens):
        if not tokens or not self._last_text:
            return ""
        start = min(tokens)
        end = max(tokens) + 1
        return self._last_text[start:end]

class SimpleTokenizer(Tokenizer):
    def __init__(self):
        super().__init__(model_name="custom", tokenizer=_SimpleInner())

t = SimpleTokenizer()

# Test 1: basic encode/decode
text = "商汤科技发布2024年全年业绩报告"
encoded = t.encode(text)
print(f"  Text: {text}")
print(f"  encode -> {len(encoded)} tokens")

# Test 2: decode a slice (simulates chunking)
chunk = t.decode(encoded[5:15])
expected = text[5:15]
print(f"  decode(tokens[5:15]) -> '{chunk}'")
print(f"  expected:               '{expected}'")
if chunk == expected:
    print("  PASS: Roundtrip correct")
else:
    print("  FAIL: Mismatch!")
    sys.exit(1)

# Test 3: full text roundtrip
full = t.decode(encoded)
print(f"  decode(all) == original: {full == text}")
if full == text:
    print("  PASS: Full roundtrip correct")
else:
    print("  FAIL: Full roundtrip broken!")
    sys.exit(1)

print()
print("All checks passed. Ready for Task 3.")
