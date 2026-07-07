import asyncio, os, sys

# Read API key
API_KEY = ""
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
for line in open(env_path, encoding='utf-8'):
    if line.startswith("DASHSCOPE_API_KEY="):
        API_KEY = line.split("=", 1)[1].strip()
        break

print(f"API Key: {API_KEY[:10]}...{API_KEY[-4:]}")

# Test 1: Sync call
import dashscope
from dashscope import Generation

r = Generation.call(
    model="qwen-plus",
    messages=[{"role": "user", "content": "say hi"}],
    api_key=API_KEY,
    result_format="message",
    max_tokens=10,
)
print(f"Sync: status={r.status_code}, code={getattr(r, 'code', 'ok')}")

# Test 2: Async wrapper (same sync call inside async)
async def test_async():
    r2 = Generation.call(
        model="qwen-plus",
        messages=[{"role": "user", "content": "say hi"}],
        api_key=API_KEY,
        result_format="message",
        max_tokens=10,
    )
    print(f"Async-wrap: status={r2.status_code}, code={getattr(r2, 'code', 'ok')}")

asyncio.run(test_async())

# Test 3: Same function signature as our experiment
async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    from dashscope import Generation as Gen2
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages or [])
    messages.append({"role": "user", "content": prompt})
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    kwargs.pop("token_tracker", None)
    result_format = "message"
    if kwargs.pop("response_format", None):
        result_format = "json_object"
    
    print(f"  llm_func called, api_key={API_KEY[:10]}...{API_KEY[-4:]}, messages={len(messages)}")
    resp = Gen2.call(
        model="qwen-plus",
        messages=messages,
        api_key=API_KEY,
        result_format=result_format,
        max_tokens=kwargs.get("max_tokens", 4096),
        temperature=kwargs.get("temperature", 0.7),
    )
    print(f"  llm_func result: status={resp.status_code}, code={getattr(resp, 'code', 'ok')}")
    if resp.status_code != 200:
        raise Exception(f"DashScope API error: {resp.code} - {resp.message}")
    return resp.output.choices[0].message.content

async def test_func():
    result = await llm_func("say hi", system_prompt="You are helpful")
    print(f"Func: result={result[:50]}")

asyncio.run(test_func())
