"""Utilities for normalizing and expanding provider/model identifiers."""

from __future__ import annotations

from typing import Iterable


# Provider prefix expansion: infer the "provider/model" form from bare model strings.
def expand_model_variants(model_name: str, alias_map: dict[str, str] | None = None) -> list[str]:
    """Return a deduplicated list of candidate model strings to try for a given name.

    Handles:
    - Exact names (openai/gpt-4.1, gemini/gemini-2.5-pro, ...)
    - Bare model names with provider prefix auto-detection
    - Alias resolution via caller-supplied alias_map
    """
    token = (model_name or "").strip()
    if not token:
        return []

    resolved = (alias_map or {}).get(token.lower(), token)

    out: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        value = (name or "").strip()
        key = value.lower()
        if not value or key in seen:
            return
        seen.add(key)
        out.append(value)

    add(token)
    add(resolved)

    for base in list(out):
        if "/" in base:
            _, rhs = base.split("/", 1)
            add(rhs)
        else:
            lower = base.lower()
            if lower.startswith(("gpt-", "o1", "o3", "o4")):
                add(f"openai/{base}")
            if "claude" in lower:
                add(f"anthropic/{base}")
            if "gemini" in lower:
                add(f"gemini/{base}")
                add(f"google/{base}")

    return out


def normalize_model_list(models: Iterable[str], alias_map: dict[str, str] | None = None) -> list[str]:
    """Deduplicate and resolve a list of model name strings."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in models:
        token = str(raw or "").strip()
        if not token:
            continue
        resolved = (alias_map or {}).get(token.lower(), token)
        key = resolved.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(resolved)
    return out
