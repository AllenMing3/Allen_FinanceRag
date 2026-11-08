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
# 说明: 运行时输出用英文，避免 Windows PowerShell 5.1 以 gbk 解码 UTF-8 导致中文乱码
# ============================================================
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$phoenix = ".\.venv-obs\Scripts\phoenix.exe"
if (-not (Test-Path $phoenix)) {
    Write-Error "Missing .venv-obs. Run first:  uv venv .venv-obs ; uv pip install --python .venv-obs -r requirements-observability.txt --index https://pypi.org/simple"
    exit 1
}

Write-Host "Phoenix UI + OTLP collector -> http://localhost:6006" -ForegroundColor Cyan
Write-Host "Main app push: PHOENIX_ENABLED=true, PHOENIX_ENDPOINT=http://localhost:6006" -ForegroundColor DarkGray
& $phoenix serve
