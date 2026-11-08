#!/usr/bin/env bash
# ============================================================
# 启动隔离的 Phoenix 可观测性环境（.venv-obs）
# ------------------------------------------------------------
# 前置（首次）:
#   uv venv .venv-obs
#   uv pip install --python .venv-obs -r requirements-observability.txt --index https://pypi.org/simple
#   （--index 兜底：清华镜像可能缺 arize-phoenix-sqlean，需回退 PyPI 拉取）
# 启动后:
#   Phoenix UI + OTLP collector 监听 http://localhost:6006
#   主应用设 PHOENIX_ENABLED=true、PHOENIX_ENDPOINT=http://localhost:6006 即可推送 span
# ============================================================
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

phoenix=".venv-obs/bin/phoenix"
if [ ! -x "$phoenix" ]; then
    echo "Missing .venv-obs. Run first:" >&2
    echo "  uv venv .venv-obs" >&2
    echo "  uv pip install --python .venv-obs -r requirements-observability.txt --index https://pypi.org/simple" >&2
    exit 1
fi

echo "Phoenix UI + OTLP collector -> http://localhost:6006"
echo "Main app push: PHOENIX_ENABLED=true, PHOENIX_ENDPOINT=http://localhost:6006"
exec "$phoenix" serve
