"""
Unit tests for ecosystem hardening patches across MinerU and PaddleOCR.
Verifies path traversal defenses, OMML namespace-agnostic translation,
and UNet table recovery grid coordinate bounds clamping.
"""

import os
from pathlib import Path
from unittest.mock import patch

import lxml.etree as ET
import numpy as np
import pytest
from mineru.model.docx.tools.math.omml import Tag2Method as MinerUTag2Method
from mineru.model.table.rec.unet_table.table_recover import TableRecover

from PaddleOCR.mcp_server.paddleocr_mcp.inference.shared.input_contract import (
    resolve_absolute_path,
)
from PaddleOCR.paddleocr._doc2md.math.omml import Tag2Method as PaddleTag2Method


def test_mcp_resolve_absolute_path_null_bytes() -> None:
    """Verify resolve_absolute_path rejects paths containing null bytes."""
    with pytest.raises(ValueError, match="null bytes"):
        resolve_absolute_path("C:\\safe\\path\x00\\secret.txt")


def test_mcp_resolve_absolute_path_sandbox_enforcement(tmp_path: Path) -> None:
    """Verify resolve_absolute_path blocks traversal outside MCP_ALLOWED_DIR."""
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    safe_file = sandbox_dir / "document.pdf"
    safe_file.write_bytes(b"%PDF-1.4 dummy")

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    secret_file = outside_dir / "secret.txt"
    secret_file.write_text("classified")

    # 1. Allowed path within sandbox succeeds
    with patch.dict(os.environ, {"MCP_ALLOWED_DIR": str(sandbox_dir)}):
        res = resolve_absolute_path(str(safe_file))
        assert res == safe_file.resolve()

    # 2. Path outside sandbox raises PermissionError
    with patch.dict(os.environ, {"MCP_ALLOWED_DIR": str(sandbox_dir)}):
        with pytest.raises(PermissionError, match="outside the permitted sandbox"):
            resolve_absolute_path(str(secret_file))


def test_omml_namespace_agnostic_extraction() -> None:
    """Verify OMML tag extraction works across different namespace URIs and prefixes."""
    # 1. Standard OMML namespace
    xml_std = (
        '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
        "<m:f><m:num><m:r><m:t>1</m:t></m:r></m:num><m:den><m:r><m:t>2</m:t></m:r></m:den></m:f>"
        "</m:oMath>"
    )
    # 2. Alternate namespace URI
    xml_alt = (
        '<math xmlns="http://schemas.microsoft.com/office/word/2010/math">'
        "<f><num><r><t>1</t></r></num><den><r><t>2</t></r></den></f>"
        "</math>"
    )

    elm_std = ET.fromstring(xml_std)
    elm_alt = ET.fromstring(xml_alt)

    paddle_parser = PaddleTag2Method()
    mineru_parser = MinerUTag2Method()

    # Verify PaddleOCR Tag2Method extracts local tag correctly
    for child in list(elm_std):
        assert paddle_parser._extract_stag(child.tag) == "f"
    for child in list(elm_alt):
        assert paddle_parser._extract_stag(child.tag) == "f"

    # Verify MinerU Tag2Method extracts local tag correctly
    for child in list(elm_std):
        assert mineru_parser._extract_stag(child.tag) == "f"
    for child in list(elm_alt):
        assert mineru_parser._extract_stag(child.tag) == "f"


def test_table_recover_bounds_clamping_prevents_index_error() -> None:
    """Verify TableRecover clamps cell bounds to avoid out-of-bounds index errors."""
    recover = TableRecover()

    # Create synthetic table polygons representing 2 rows and 2 columns
    polygons = np.array(
        [
            [[10, 10], [10, 30], [50, 30], [50, 10]],
            [[60, 10], [60, 30], [100, 30], [100, 10]],
            [[10, 40], [10, 60], [50, 60], [50, 40]],
            [[60, 40], [60, 60], [100, 60], [100, 40]],
        ],
        dtype=np.float32,
    )

    table_res, logic_points = recover(polygons)

    assert isinstance(table_res, dict)
    assert logic_points.shape == (4, 4)
    # Ensure all row and column indices are non-negative and bounded
    assert np.all(logic_points >= 0)
    for point in logic_points:
        row_start, row_end, col_start, col_end = point
        assert row_start <= row_end
        assert col_start <= col_end
