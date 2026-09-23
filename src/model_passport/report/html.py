"""A tiny, dependency-free HTML builder that escapes everything by default.

``Html`` marks markup that is already safe. ``el`` escapes every child that is not ``Html``,
so user-controlled values (model names, declared text, findings) can never inject markup.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence

Child = object  # str, numbers, Html, None (skipped), or iterables of these


class Html(str):
    """A string of trusted markup. Only this module and ``render`` construct it."""

    __slots__ = ()


def esc(value: object) -> Html:
    return value if isinstance(value, Html) else Html(html.escape(str(value), quote=True))


def _flatten(children: Iterable[Child]) -> Iterable[Child]:
    for child in children:
        if child is None:
            continue
        if isinstance(child, Iterable) and not isinstance(child, str | bytes):
            yield from _flatten(child)
        else:
            yield child


def join(*children: Child) -> Html:
    return Html("".join(esc(c) for c in _flatten(children)))


def el(name: str, *children: Child, **attrs: object) -> Html:
    """``el("td", value, class_="mono")`` -> ``<td class="mono">value</td>`` (escaped)."""
    rendered = "".join(
        f' {key.rstrip("_").replace("_", "-")}="{html.escape(str(value), quote=True)}"'
        for key, value in attrs.items()
        if value is not None and value is not False
    )
    return Html(f"<{name}{rendered}>{join(*children)}</{name}>")


def table(headers: Sequence[str], rows: Iterable[Sequence[Child]]) -> Html:
    head = el("tr", [el("th", h) for h in headers])
    return el("table", head, [el("tr", [el("td", cell) for cell in row]) for row in rows])
