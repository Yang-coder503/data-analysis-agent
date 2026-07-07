"""
文件格式与编码自动检测。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

# 常见文本编码（按概率降序）
_ENCODING_PRIORITY = [
    "utf-8", "utf-8-sig", "gbk", "gb2312", "gb18030",
    "shift-jis", "euc-kr", "big5", "latin-1", "cp1252",
]

_FORMAT_EXTENSIONS: dict[str, list[str]] = {
    "csv":     [".csv", ".tsv", ".data", ".dat"],
    "excel":   [".xlsx", ".xls", ".xlsm"],
    "json":    [".json", ".jsonl", ".ndjson"],
    "parquet": [".parquet"],
    "html":    [".html", ".htm"],
    "xml":     [".xml"],
}


def detect_format(path: str) -> str:
    """根据文件扩展名推断格式。"""
    ext = Path(path).suffix.lower()
    for fmt, exts in _FORMAT_EXTENSIONS.items():
        if ext in exts:
            return fmt
    return "unknown"


def detect_delimiter(path: str, sample_size: int = 4096) -> str:
    """尝试从文件开头推断分隔符（CSV 场景）。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            sample = f.read(sample_size)
    except (UnicodeDecodeError, OSError):
        try:
            with open(path, "r", encoding="latin-1") as f:
                sample = f.read(sample_size)
        except OSError:
            return ","

    first_line = sample.split("\n")[0]
    delimiters = [",", "\t", ";", "|", "\\s+"]

    scores = {}
    for delim in delimiters:
        if delim == "\\s+":
            count = len(re.findall(r"\\s+", first_line))
        else:
            count = first_line.count(delim)
        if count > 0:
            scores[delim] = count

    if not scores:
        return ","

    # 选出现次数最多的分隔符
    best = max(scores, key=scores.get)
    return "," if best == "\\s+" else best


def detect_encoding(path: str) -> str:
    """探测文件编码。优先 chardet，fallback 逐个尝试。"""
    try:
        import chardet
        with open(path, "rb") as f:
            raw = f.read(10000)
        result = chardet.detect(raw)
        if result["encoding"] and result["confidence"] > 0.5:
            return result["encoding"]
    except ImportError:
        pass
    except Exception:
        pass

    # fallback: 逐个尝试常见编码
    for enc in _ENCODING_PRIORITY:
        try:
            with open(path, "r", encoding=enc) as f:
                f.read(100)
            return enc
        except (UnicodeDecodeError, OSError):
            continue

    return "utf-8"


def is_text_based(path: str, sample_size: int = 1024) -> bool:
    """快速判断文件是否为文本格式。"""
    try:
        with open(path, "rb") as f:
            raw = f.read(sample_size)
        # 检查是否包含空字节（二进制文件标志）
        return b"\x00" not in raw
    except OSError:
        return False
