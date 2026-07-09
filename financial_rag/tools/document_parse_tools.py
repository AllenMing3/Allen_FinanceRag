"""
Document Parse Tools — 非文本文档解析能力池

将 PDF 解析和图片多模态理解下沉为可注册的工具，
供 IngestionAgent 通过 call_tool() 调用。

设计原则:
    1. 与 extraction_tools.py 对齐 — 闭包注入 LLM，FunctionDef 注册
    2. LLM-first (图片) — 用多模态模型做结构化理解，而非传统 OCR
    3. 纯本地 (PDF) — PyMuPDF 解析，无需 LLM
    4. Confidence signaling — 每个结果标注 parse_type + _confidence

工具列表:
    - describe_image_file: 用 qwen-vl 多模态模型解析图片为结构化文本
    - parse_pdf_file: 用 PyMuPDF 解析 PDF 为纯文本
"""
import logging
import os
from typing import Dict, Any, List

from financial_rag.tools.core import FunctionDef

logger = logging.getLogger(__name__)


# ===================== LLM 注入 (闭包模式) =====================

_llm_ref = {"llm": None}


def inject_document_parse_llm(llm):
    """注入 LLM 实例 — 由 create_financial_registry() 调用"""
    _llm_ref["llm"] = llm


def _get_llm():
    return _llm_ref["llm"]


# ===================== Tool 1: 图片多模态理解 =====================

def describe_image_file(image_path: str) -> Dict[str, Any]:
    """用多模态大模型（qwen-vl）解析图片内容，返回结构化文本描述。

    支持财务报表截图、K线走势图、数据表格、架构图等 AI/科技行业常见图片类型。

    Args:
        image_path: 图片文件的本地路径（支持 png/jpg/jpeg/webp）
    """
    if not image_path or not os.path.isfile(image_path):
        return {"_confidence": "none", "_error": f"文件不存在: {image_path}"}

    llm = _get_llm()
    if not llm:
        logger.warning("[describe_image_file] LLM 未注入，无法解析图片")
        return {"_confidence": "none", "_error": "LLM 未注入，图片解析不可用"}

    # 组装 domain prompt（与 extraction_tools 同样的 system + few-shot 模式）
    from financial_rag.prompts import (
        IMAGE_UNDERSTANDING_SYSTEM,
        IMAGE_UNDERSTANDING_PROMPT,
        FEW_SHOT_EXAMPLES,
    )

    system_prompt = IMAGE_UNDERSTANDING_SYSTEM
    few_shot = FEW_SHOT_EXAMPLES.get("image_understanding", "")
    if few_shot:
        system_prompt += f"\n\n以下是一些示例供参考：\n{few_shot}"
    few_shot_bad = FEW_SHOT_EXAMPLES.get("image_understanding_bad", "")
    if few_shot_bad:
        system_prompt += f"\n\n以下是错误示范，请避免：\n{few_shot_bad}"

    # 调用多模态 API
    try:
        resp = llm.describe_image(
            image_path=image_path,
            prompt=IMAGE_UNDERSTANDING_PROMPT,
        )
        description = resp.content
        if not description or not description.strip():
            return {
                "_confidence": "low",
                "_source": "vision",
                "_error": "模型返回空内容",
                "description": "",
                "vision_model": resp.model,
            }

        return {
            "_confidence": "high",
            "_source": "vision",
            "description": description.strip(),
            "vision_model": resp.model,
            "original_file": os.path.basename(image_path),
        }
    except FileNotFoundError as e:
        logger.error(f"[describe_image_file] 文件不存在: {e}")
        return {"_confidence": "none", "_error": str(e)}
    except RuntimeError as e:
        logger.warning(f"[describe_image_file] 多模态 API 失败: {e}")
        return {
            "_confidence": "none",
            "_error": f"多模态 API 失败: {e}",
            "original_file": os.path.basename(image_path),
        }


# ===================== Tool 2: PDF 文本解析 =====================

def parse_pdf_file(pdf_path: str) -> Dict[str, Any]:
    """用 PyMuPDF 解析 PDF 文件为纯文本。

    逐页提取文字，按段落拼接，适合财报、研报等 PDF 文档的文本层摄取。

    Args:
        pdf_path: PDF 文件的本地路径
    """
    if not pdf_path or not os.path.isfile(pdf_path):
        return {"_confidence": "none", "_error": f"文件不存在: {pdf_path}"}

    try:
        import pymupdf
    except ImportError:
        logger.error("[parse_pdf_file] PyMuPDF 未安装，请运行: pip install PyMuPDF")
        return {"_confidence": "none", "_error": "PyMuPDF 未安装"}

    try:
        doc = pymupdf.open(pdf_path)
        pages_text = []
        for page_num, page in enumerate(doc, 1):
            t = page.get_text()
            if t.strip():
                pages_text.append(t.strip())
        doc.close()

        if not pages_text:
            return {
                "_confidence": "low",
                "_source": "pdf",
                "text": "",
                "page_count": doc.page_count if hasattr(doc, 'page_count') else 0,
                "original_file": os.path.basename(pdf_path),
                "_warning": "PDF 无文字内容（可能是纯图片 PDF）",
            }

        full_text = "\n\n".join(pages_text)
        return {
            "_confidence": "high",
            "_source": "pdf",
            "text": full_text,
            "page_count": len(pages_text),
            "char_count": len(full_text),
            "original_file": os.path.basename(pdf_path),
        }
    except Exception as e:
        logger.warning(f"[parse_pdf_file] PDF 解析失败 {pdf_path}: {e}")
        return {
            "_confidence": "none",
            "_error": f"解析失败: {e}",
            "original_file": os.path.basename(pdf_path),
        }


# ===================== FunctionDef 列表 =====================

DOCUMENT_PARSE_TOOLS: List[FunctionDef] = [
    FunctionDef(
        name="describe_image_file",
        description="用多模态大模型（qwen-vl）解析图片内容。"
                    "支持财务报表截图、K线走势图、数据表格、架构图等 AI/科技行业常见图片类型。"
                    "返回结构化文本描述，包含关键数据、图表分析、文字转录等。"
                    "当需要从图片中提取信息时使用。",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "图片文件的本地路径（支持 png/jpg/jpeg/webp）",
                },
            },
            "required": ["image_path"],
        },
        callback=describe_image_file,
        category="data",
        tags=["图片", "多模态", "OCR", "图表", "视觉理解"],
    ),
    FunctionDef(
        name="parse_pdf_file",
        description="用 PyMuPDF 解析 PDF 文件为纯文本。"
                    "逐页提取文字，适合财报、研报等 PDF 文档的文本层摄取。"
                    "当需要从 PDF 中提取文字内容时使用。",
        parameters={
            "type": "object",
            "properties": {
                "pdf_path": {
                    "type": "string",
                    "description": "PDF 文件的本地路径",
                },
            },
            "required": ["pdf_path"],
        },
        callback=parse_pdf_file,
        category="data",
        tags=["PDF", "文档解析", "文本提取", "PyMuPDF"],
    ),
]
