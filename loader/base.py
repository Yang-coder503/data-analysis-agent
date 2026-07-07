"""
数据加载核心容器 - LoadedData 及其元数据结构。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class ColumnInfo:
    """单个列的概要信息。"""
    name: str
    dtype: str
    inferred_type: str
    null_count: int
    null_ratio: float
    unique_count: int
    is_numeric: bool = False
    sample_values: list[Any] = field(default_factory=list)

    def __post_init__(self):
        if self.dtype.startswith(("int", "float", "Int", "Float")):
            self.is_numeric = True


@dataclass
class LoadMeta:
    """数据来源及加载过程的元信息。"""
    source: str
    format: str = "unknown"
    encoding: Optional[str] = None
    rows: int = 0
    cols: int = 0
    memory_bytes: int = 0
    load_time_ms: float = 0.0
    raw_shape: tuple[int, int] = (0, 0)
    sheets: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def memory_display(self) -> str:
        for unit, threshold in [("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)]:
            if self.memory_bytes >= threshold:
                return f"{self.memory_bytes / threshold:.1f} {unit}"
        return f"{self.memory_bytes} B"


class LoadedData:
    """统一数据容器，装载原始 DataFrame + 元数据 + 列级信息。"""

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        name: str = "",
        meta: Optional[LoadMeta] = None,
    ):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(data).__name__}")

        self._data = data
        self.name = name or "unnamed"
        self.meta = meta or LoadMeta(source=self.name)
        self._column_info: Optional[list[ColumnInfo]] = None
        self.meta.rows, self.meta.cols = data.shape
        self.meta.memory_bytes = int(data.memory_usage(deep=True).sum())

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._data, name)

    def preview(self, n: int = 5) -> str:
        """返回前 n 行格式化预览。"""
        shape = f"{self.meta.rows} rows x {self.meta.cols} cols"
        header = f"<LoadedData {self.name!r} - {shape}>"
        return header + "\n" + self._data.head(n).to_string()

    def schema(self) -> list[ColumnInfo]:
        if self._column_info is None:
            self._column_info = self._build_column_info()
        return self._column_info

    def info(self) -> str:
        """一个可读的概要报告，方便 LLM 快速理解数据。"""
        lines = [
            f"数据集: {self.name}",
            f"来源:   {self.meta.source}",
            f"格式:   {self.meta.format}",
            f"维度:   {self.meta.rows} 行 x {self.meta.cols} 列",
            f"内存:   {self.meta.memory_display}",
            f"加载耗时: {self.meta.load_time_ms:.1f} ms",
        ]
        if self.meta.encoding:
            lines.append(f"编码:   {self.meta.encoding}")
        if self.meta.warnings:
            lines.append("警告:")
            for w in self.meta.warnings:
                lines.append(f"  . {w}")

        lines.append("")
        lines.append("列信息:")
        lines.append(f"  {'列名':<20} {'类型':<12} {'空值%':>8} {'唯一值':>8}  示例值")
        lines.append(f"  {'-'*20} {'-'*12} {'-'*8} {'-'*8}  {'-'*20}")
        for col in self.schema():
            samples = ", ".join(str(v) for v in col.sample_values[:3])
            lines.append(
                f"  {col.name:<20} {col.dtype:<12} "
                f"{col.null_ratio*100:>7.1f}% {col.unique_count:>8}  {samples}"
            )
        return "\n".join(lines)

    def _build_column_info(self) -> list[ColumnInfo]:
        infos = []
        for col_name in self._data.columns:
            series = self._data[col_name]
            null_count = int(series.isna().sum())
            unique_count = int(series.nunique())
            sample_values = series.dropna().head(5).tolist()
            infos.append(
                ColumnInfo(
                    name=str(col_name),
                    dtype=str(series.dtype),
                    inferred_type=self._infer_semantic_type(series),
                    null_count=null_count,
                    null_ratio=null_count / max(len(series), 1),
                    unique_count=unique_count,
                    sample_values=sample_values,
                )
            )
        return infos

    @staticmethod
    def _infer_semantic_type(series: pd.Series) -> str:
        dtype = series.dtype
        if pd.api.types.is_bool_dtype(dtype):
            return "boolean"
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "datetime"
        non_null = series.dropna()
        if len(non_null) == 0:
            return "empty"
        if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype):
            if non_null.nunique() <= 2 and set(non_null.unique()).issubset({0, 1, 0.0, 1.0}):
                return "boolean"
            if non_null.nunique() <= min(20, len(non_null) // 10 + 1):
                return "categorical"
            return "numeric"
        if non_null.nunique() == len(non_null):
            return "id" if len(non_null) > 100 else "text"
        if non_null.nunique() <= 30:
            return "categorical"
        try:
            pd.to_datetime(non_null.head(100), infer_datetime_format=True)
            return "datetime"
        except (ValueError, TypeError):
            pass
        return "text"


def wrap_dataframe(
    df: pd.DataFrame,
    name: str = "",
    source: str = "",
    format_: str = "memory",
) -> LoadedData:
    """将现有 DataFrame 快速包装为 LoadedData。"""
    t0 = time.perf_counter()
    loaded = LoadedData(df, name=name or source or "unnamed")
    loaded.meta.source = source or "memory"
    loaded.meta.format = format_
    loaded.meta.load_time_ms = (time.perf_counter() - t0) * 1000
    return loaded
