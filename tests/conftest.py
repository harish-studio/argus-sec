"""
Shared pytest configuration for argus-sec.

Adds the project root to sys.path so `services.*` and `app.*` import
the same way they do when uvicorn runs from the repo root.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


API_BASE = os.environ.get("ARGUS_API_URL", "http://localhost:8001")


@pytest.fixture(scope="session")
def api_base() -> str:
    return API_BASE


@pytest.fixture
def sample_payload() -> dict:
    """A Qdrant payload shaped like a real cve_chunks record."""
    return {
        "cve_id":         "CVE-2026-0545",
        "narrative_text": "MLflow allows arbitrary file write via a crafted archive.",
        "severity":       "CRITICAL",
        "packages":       ["MLflow"],
        "cisa_kev":       False,
        "fix_available":  True,
        "attack_vector":  "NETWORK",
        "published":      "2026-01-14",
        "nvd_url":        "https://nvd.nist.gov/vuln/detail/CVE-2026-0545",
    }
