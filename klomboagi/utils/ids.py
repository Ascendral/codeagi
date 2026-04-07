from __future__ import annotations

from uuid import uuid4


def new_id(prefix: str = "id") -> str:
    """Generate a readable unique identifier."""
    return f"{prefix}_{uuid4().hex[:12]}"
