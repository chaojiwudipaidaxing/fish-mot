#!/usr/bin/env python
"""Shared naming helpers for the cumulative main-chain tracker family."""

from __future__ import annotations

from typing import Dict, Iterable, List


MAIN_CHAIN_LABEL_MAP: Dict[str, str] = {
    "Base": "Base",
    "+gating": "Base+gating",
    "+traj": "Base+gating+traj",
    "+adaptive": "Base+gating+traj+adaptive",
    "Base+gating": "Base+gating",
    "Base+gating+traj": "Base+gating+traj",
    "Base+gating+traj+adaptive": "Base+gating+traj+adaptive",
}

MAIN_CHAIN_METHOD_ORDER: List[str] = [
    "Base",
    "Base+gating",
    "Base+gating+traj",
    "Base+gating+traj+adaptive",
]


def normalize_main_chain_label(label: str) -> str:
    """Return the canonical cumulative-chain display label when applicable."""
    return MAIN_CHAIN_LABEL_MAP.get(str(label).strip(), str(label).strip())


def normalize_main_chain_rows(rows: Iterable[dict], key: str = "method") -> List[dict]:
    """Copy rows and normalize their method labels when they belong to the main chain."""
    normalized: List[dict] = []
    for row in rows:
        copied = dict(row)
        if key in copied:
            copied[key] = normalize_main_chain_label(str(copied[key]))
        normalized.append(copied)
    return normalized
