"""
CSV / TSV / 定宽文本加载器。
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .base import LoadMeta, LoadedData
from .file_detector import detect_delimiter, detect_encoding


def load_csv(
    path: str,
    *,
    delimiter: Optional[str] = None,
    encoding: Optional[str] = None,
    name: Optional[str] = None,
    dtype: Optional[dict] = None,
    parse_dates: bool = True,
    low_memory: bool = True,
    **kwargs: Any,
) -> LoadedData:
    """
    加载 CSV / TSV 文件。

    Parameters
    ----------
    path : str
        文件路径。
    delimiter : str, optional
        分隔符。为 None 时自动检测。
    encoding : str, optional
        编码。为 None 时自动探测。
    name : str, optional
        数据集名，默认用文件名。
    dtype : dict, optional
        指定列类型。
    parse_dates : bool
        是否尝试推断日期列。默认 True。
    low_memory : bool
        是否分块读取（省内存）。默认 True。
    **kwargs :
        传给 pd.read_csv 的额外参数。
    """
    t0 = time.perf_counter()

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    name = name or path_obj.stem
    source = str(path_obj.resolve())

    # 自动探测
    if delimiter is None:
        delimiter = detect_delimiter(path)
    if encoding is None:
        encoding = detect_encoding(path)

    warns: list[str] = []

    try:
        df = pd.read_csv(
            path,
            sep=delimiter,
            encoding=encoding,
            dtype=dtype,
            low_memory=low_memory,
            on_bad_lines="warn",
            **kwargs,
        )
    except UnicodeDecodeError:
        # fallback 到 latin-1
        warns.append(f"编码 {encoding} 解码失败，回退到 latin-1")
        encoding = "latin-1"
        df = pd.read_csv(
            path,
            sep=delimiter,
            encoding=encoding,
            dtype=dtype,
            low_memory=low_memory,
            on_bad_lines="warn",
            **kwargs,
        )

    load_time_ms = (time.perf_counter() - t0) * 1000

    # 日期自动推断
    if parse_dates:
        from pandas.errors import OutOfBoundsDatetime
        for col in df.select_dtypes(include=["object"]).columns:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() > len(df) * 0.5:
                    df[col] = parsed
            except (ValueError, TypeError, OutOfBoundsDatetime):
                continue

    # 采集原始形状
    raw_shape = df.shape

    meta = LoadMeta(
        source=source,
        format="csv",
        encoding=encoding,
        rows=len(df),
        cols=len(df.columns),
        memory_bytes=int(df.memory_usage(deep=True).sum()),
        load_time_ms=load_time_ms,
        raw_shape=raw_shape,
        warnings=warns,
    )

    return LoadedData(df, name=name, meta=meta)
