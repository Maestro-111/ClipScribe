"""Shared spreadsheet/download machinery for the export builders.

``abcd_exports`` (parser verdicts) and ``metadata_exports`` (extractor output)
serialize different data but share the same safety and packaging rules: cell
text is made inert and Excel-legal, columns are autosized, workbooks become
response bytes, and downloads get a filesystem-safe name.
"""

from __future__ import annotations

import io
import re
from collections.abc import Mapping

from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

XLSX_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

_FORMULA_PREFIXES = ("=", "+", "-", "@")
# Excel's per-cell character limit; longer text makes Excel "repair" the file.
_MAX_CELL_TEXT = 32_767


def export_filename(stem: str, fmt: str) -> str:
    """A safe download filename: slugged stem + format extension."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "export"
    return f"{safe}.{fmt}"


def safe_sheet_title(name: str, index: int, used: set[str]) -> str:
    """A unique, Excel-legal (<=31 char, no ``[]:*?/\\``) sheet title.

    Prefixed with the 1-based run number so order is obvious and collisions
    between same-named videos are impossible.
    """
    cleaned = re.sub(r"[\[\]:*?/\\]", " ", name).strip() or "run"
    prefix = f"{index}. "
    title = (prefix + cleaned)[:31].strip()
    base = title
    n = 2
    while title.lower() in used:
        suffix = f" ({n})"
        title = base[: 31 - len(suffix)] + suffix
        n += 1
    used.add(title.lower())
    return title


def spreadsheet_text(value: str) -> str:
    """Make text safe for a CSV/XLSX cell.

    Strips control characters openpyxl rejects, neutralizes formula-like
    prefixes (CSV/formula injection), and caps length at Excel's cell limit.
    """
    text = ILLEGAL_CHARACTERS_RE.sub("", value)
    if text.startswith(_FORMULA_PREFIXES):
        text = f"'{text}"
    return text[:_MAX_CELL_TEXT]


def cell_value(value: object) -> object:
    """Pass numbers/bools/None through; render anything else as safe text."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return spreadsheet_text(str(value))


def autosize(
    ws: Worksheet,
    max_width: int = 80,
    column_max_widths: Mapping[int, int] | None = None,
) -> None:
    """Rough column autosize from cell text length, capped so prose stays sane.

    ``column_max_widths`` overrides ``max_width`` for specific 1-based columns.
    """
    caps = column_max_widths or {}
    widths: dict[int, int] = {}
    for row in ws.iter_rows():
        for cell in row:
            if cell.value is None:
                continue
            longest = max(len(line) for line in str(cell.value).splitlines() or [""])
            widths[cell.column] = max(widths.get(cell.column, 0), longest)
    for col, width in widths.items():
        cap = caps.get(col, max_width)
        ws.column_dimensions[get_column_letter(col)].width = min(width + 2, cap)


def workbook_bytes(wb: Workbook) -> bytes:
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
