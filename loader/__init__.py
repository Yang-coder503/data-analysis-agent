"""
loader 包 — 数据加载层。

统一入口 DataLoader，支持 CSV / Excel / JSON / 自动检测。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .base import ColumnInfo, LoadMeta, LoadedData, wrap_dataframe
from .file_detector import detect_format

# 导入各加载器
from .csv_loader import load_csv
from .excel_loader import load_excel, list_sheets
from .json_loader import load_json, load_jsonl


class DataLoader:
    """统一数据加载入口。"""

    _LOADER_MAP = {
        "csv": load_csv,
        "excel": load_excel,
        "json": load_json,
    }

    def __init__(self):
        raise NotImplementedError("DataLoader 是静态类，请直接使用类方法")

    @classmethod
    def from_path(cls, path: str, **kwargs: Any) -> LoadedData:
        """
        自动检测文件类型并加载。

        Parameters
        ----------
        path : str
            文件路径。
        **kwargs :
            透传给对应加载器的参数。

        Returns
        -------
        LoadedData
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        fmt = detect_format(path)

        if fmt not in cls._LOADER_MAP:
            supported = ", ".join(cls._LOADER_MAP.keys())
            raise ValueError(
                f"不支持的文件格式 {fmt!r}（路径: {path}）。"
                f" 支持: {supported}"
            )

        loader = cls._LOADER_MAP[fmt]
        return loader(path, **kwargs)

    @classmethod
    def from_csv(cls, path: str, **kwargs: Any) -> LoadedData:
        return load_csv(path, **kwargs)

    @classmethod
    def from_excel(cls, path: str, **kwargs: Any) -> LoadedData:
        return load_excel(path, **kwargs)

    @classmethod
    def from_json(cls, path: str, **kwargs: Any) -> LoadedData:
        return load_json(path, **kwargs)

    @classmethod
    def from_dataframe(
        cls,
        df: Any,
        name: str = "",
        source: str = "",
    ) -> LoadedData:
        """从现有 DataFrame 构造 LoadedData。"""
        import pandas as pd
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        return wrap_dataframe(df, name=name, source=source)


__all__ = [
    "DataLoader",
    "LoadedData",
    "LoadMeta",
    "ColumnInfo",
    "wrap_dataframe",
    "load_csv",
    "load_excel",
    "load_json",
    "list_sheets",
]
