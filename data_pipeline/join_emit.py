"""Join aligned tokens into emit / transcript text.

Chinese tokens stay glued. Adjacent Latin/digit words get a single space,
including a leading space on a later chunk so ``"".join(emits) == text``
and SentencePiece sees ``▁`` on non-initial English words.
"""

from __future__ import annotations

from typing import Iterable, Optional


def _is_cjk_char(ch: str) -> bool:
    o = ord(ch)
    return (
        0x3400 <= o <= 0x4DBF
        or 0x4E00 <= o <= 0x9FFF
        or 0xF900 <= o <= 0xFAFF
        or 0x20000 <= o <= 0x2A6DF
        or 0x3040 <= o <= 0x30FF
        or 0xAC00 <= o <= 0xD7AF
    )


def is_alnum_word(tok: str) -> bool:
    """True if token is a space-delimited (Latin/digit) word, not CJK."""
    s = (tok or "").strip()
    if not s:
        return False
    if any(_is_cjk_char(c) for c in s):
        return False
    return any(c.isalnum() for c in s)


def needs_space_between(prev: str, cur: str) -> bool:
    return is_alnum_word(prev) and is_alnum_word(cur)


def _edge_word(text: str, *, last: bool) -> str:
    parts = (text or "").split()
    if not parts:
        return ""
    return parts[-1] if last else parts[0]


def join_emit_texts(chunks: Iterable[str]) -> str:
    """Join decoded emit strings; insert a space between Latin/digit chunks.

    Per-emit ``decode(...).strip()`` drops the SentencePiece ``▁`` that sat
    on the chunk boundary. Re-insert it here so display matches
    ``decode(all text_ids)``.
    """
    out: list[str] = []
    prev = ""
    for raw in chunks:
        cur = str(raw).strip() if raw is not None else ""
        if not cur:
            continue
        if prev and needs_space_between(
            _edge_word(prev, last=True),
            _edge_word(cur, last=False),
        ):
            out.append(" ")
        out.append(cur)
        prev = cur
    return "".join(out)


def join_emit_tokens(
    tokens: Iterable[str],
    *,
    prev_token: Optional[str] = None,
) -> str:
    """Join ``tokens``; insert a space between Latin/digit words.

    ``prev_token`` is the last token of the previous emit (if any). When it
    and the first token here both need a space, the result starts with `` ``
    so chunks concatenate back to the full transcript.
    """
    toks = [str(t).strip() for t in tokens if str(t).strip()]
    if not toks:
        return ""
    parts: list[str] = []
    prev = prev_token.strip() if prev_token else None
    for w in toks:
        if prev is not None and needs_space_between(prev, w):
            parts.append(" ")
        parts.append(w)
        prev = w
    return "".join(parts)
