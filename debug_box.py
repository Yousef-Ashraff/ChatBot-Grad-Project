"""
debug_box.py — Shared Debug Box Printer
========================================

Single source of truth for the box-drawing style used throughout the
preprocessing and agent debug output.

Usage
─────
    from debug_box import box, set_verbose, is_verbose

    set_verbose(True)           # call once at CLI startup

    box("🤖  MY TITLE", ["line 1", "line 2"])   # prints only when verbose=True
    box("🤖  MY TITLE", ["line 1"], force=True)  # always prints regardless

Box style
─────────
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                     🤖  MY TITLE                                     ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║  line 1                                                              ║
    ║  line 2                                                              ║
    ╚══════════════════════════════════════════════════════════════════════╝

Long lines are automatically wrapped to fit inside the box.
"""

from __future__ import annotations

import textwrap
from typing import List

# ── Global verbose flag ───────────────────────────────────────────────────────
# Set once at CLI startup via set_verbose(True).
# Checked by box() to decide whether to print.

_VERBOSE: bool = False
W: int = 68          # usable width inside the box (excluding 2-char borders)


def set_verbose(value: bool) -> None:
    """Enable or disable debug output globally."""
    global _VERBOSE
    _VERBOSE = value


def is_verbose() -> bool:
    """Return the current verbose flag."""
    return _VERBOSE


def box(title: str, body_lines: List[str], force: bool = False) -> None:
    """
    Print a titled Unicode box with wrapped body text.

    Args:
        title:      Short title shown centred at the top of the box.
        body_lines: Lines of content. Each line is auto-wrapped at W chars.
        force:      If True, print even when verbose=False.
    """
    if not _VERBOSE and not force:
        return

    inner_w = W + 2     # width between the vertical border chars ║ ... ║

    border_top    = f"╔{'═' * inner_w}╗"
    border_sep    = f"╠{'═' * inner_w}╣"
    border_bottom = f"╚{'═' * inner_w}╝"

    # Centre the title
    pad   = (inner_w - len(title)) // 2
    extra = (inner_w - len(title)) % 2
    title_line = f"║{' ' * pad}{title}{' ' * (pad + extra)}║"

    print(f"\n{border_top}")
    print(title_line)

    if body_lines:
        print(border_sep)
        for raw in body_lines:
            # Auto-wrap each line so it never overflows the box
            for chunk in textwrap.wrap(str(raw), width=W) or [""]:
                print(f"║ {chunk:<{W}} ║")

    print(border_bottom)