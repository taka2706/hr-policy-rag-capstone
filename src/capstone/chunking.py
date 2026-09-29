"""Chunking: split docs into retrievable units.

Why: retrieval granularity = answer granularity. Too big -> noisy prompts, diluted
embeddings. Too small -> lost context. We split on markdown headings first (semantic
boundaries), then window inside long sections with overlap so a fact straddling a
cut still appears whole in at least one chunk.
"""
from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    source: str
    heading: str
    text: str


def _windows(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    out, i = [], 0
    while i < len(text):
        end = min(i + size, len(text))
        if end < len(text):                              # snap back to whitespace
            sp = text.rfind(" ", i + size // 2, end)
            end = sp if sp != -1 else end
        out.append(text[i:end].strip())
        if end >= len(text):
            break
        i = max(end - overlap, i + 1)
    return out


def chunk_document(source: str, text: str, size: int = 700, overlap: int = 100) -> list[Chunk]:
    sections, heading, buf = [], "", []
    for line in text.splitlines():
        if line.startswith("#"):
            if buf and "".join(buf).strip():
                sections.append((heading, "\n".join(buf).strip()))
            heading, buf = line.lstrip("# ").strip(), []
        else:
            buf.append(line)
    if buf and "".join(buf).strip():
        sections.append((heading, "\n".join(buf).strip()))
    chunks = []
    for h, body in sections:
        for w in _windows(body, size, overlap):
            chunks.append(Chunk(f"{source}#{len(chunks)}", source, h, w))
    return chunks
