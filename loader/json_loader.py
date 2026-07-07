"""
JSON / JSONL / NDJSON 加载器。
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .base import LoadMeta, LoadedData


def load_json(
    path: str,
    *,
    name: Optional[str] = None,
    orient: Optional[str] = None,
    lines: Optional[bool] = None,
    **kwargs: Any,
) -> LoadedData:
    """
    加载 JSON / JSONL / NDJSON 文件。

    Parameters
    ----------
    path : str
        文件路径。
    name : str, optional
        数据集名。
    orient : str, optional
        JSON 格式方向（split / records / index / columns / values）。
        自动推断。
    lines : bool, optional
        是否按行读取（JSONL）。自动推断。
    **kwargs :
        传给 pd.read_json 的额外参数。
    """
    t0 = time.perf_counter()

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    name = name or path_obj.stem
    source = str(path_obj.resolve())
    ext = path_obj.suffix.lower()

    # JSONL / NDJSON 自动识别
    if lines is None:
        lines = ext in (".jsonl", ".ndjson")

    if lines:
        # JSONL 逐行解析
        try:
            rows = []
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        rows.append(pd.read_json(io.StringIO(line), typ="series"))
            dfs = [pd.DataFrame(rows)]
            df = pd.concat(dfs, ignore_index=True)
        except Exception as e:
            raise ValueError(f"JSONL 加载失败: {e}") from e
    else:
        df = pd.read_json(path, orient=orient, **kwargs)

    load_time_ms = (time.perf_counter() - t0) * 1000
    raw_shape = df.shape

    meta = LoadMeta(
        source=source,
        format="json",
        rows=len(df),
        cols=len(df.columns),
        memory_bytes=int(df.memory_usage(deep=True).sum()),
        load_time_ms=load_time_ms,
        raw_shape=raw_shape,
    )

    return LoadedData(df, name=name, meta=meta)


def load_jsonl(path: str, **kwargs: Any) -> LoadedData:
    """加载 JSONL 文件的快捷方式。"""
    return load_json(path, lines=True, **kwargs)
