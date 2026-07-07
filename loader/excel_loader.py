"""
Excel (.xlsx / .xls) 加载器。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .base import LoadMeta, LoadedData


def load_excel(
    path: str,
    *,
    sheet: Optional[str | int] = None,
    name: Optional[str] = None,
    dtype: Optional[dict] = None,
    **kwargs: Any,
) -> LoadedData:
    """
    加载 Excel 文件。

    Parameters
    ----------
    path : str
        文件路径。
    sheet : str or int, optional
        工作表名或索引（0-based）。默认加载所有 sheet 中的第一个。
    name : str, optional
        数据集名。
    dtype : dict, optional
        指定列类型。
    **kwargs :
        传给 pd.read_excel 的额外参数。
    """
    t0 = time.perf_counter()

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    name = name or path_obj.stem
    source = str(path_obj.resolve())

    # 获取所有 sheet 名
    with pd.ExcelFile(path) as xls:
        all_sheets = xls.sheet_names

    # 确定读取哪个 sheet
    if sheet is None:
        sheet = 0
    if isinstance(sheet, int):
        sheet_name = all_sheets[sheet]
    else:
        sheet_name = sheet

    warnings: list[str] = []

    try:
        df = pd.read_excel(path, sheet_name=sheet_name, dtype=dtype, **kwargs)
    except Exception as e:
        # fallback: 用 openpyxl 引擎
        warnings.append(f"默认引擎失败，回退到 openpyxl: {e}")
        df = pd.read_excel(path, sheet_name=sheet_name, dtype=dtype, engine="openpyxl", **kwargs)

    load_time_ms = (time.perf_counter() - t0) * 1000
    raw_shape = df.shape

    meta = LoadMeta(
        source=source,
        format="excel",
        rows=len(df),
        cols=len(df.columns),
        memory_bytes=int(df.memory_usage(deep=True).sum()),
        load_time_ms=load_time_ms,
        raw_shape=raw_shape,
        warnings=warnings,
        sheets=all_sheets,
    )

    return LoadedData(df, name=name, meta=meta)


def list_sheets(path: str) -> list[str]:
    """列出 Excel 文件的所有工作表名。"""
    with pd.ExcelFile(path) as xls:
        return xls.sheet_names
