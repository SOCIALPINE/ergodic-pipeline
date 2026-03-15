"""Base classes for information sources."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

try:
    import requests as _requests_module
except ImportError:
    _requests_module = None  # type: ignore[assignment]


@dataclass
class SearchResult:
    """Unified search result from any source."""

    title: str
    summary: str = ""
    source_type: str = ""
    source_name: str = ""
    url: str = ""
    date: str = ""
    year: int = 0
    authors: str = ""
    citations: int = 0
    venue: str = ""
    relevance_score: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_report_entry(self, index: int) -> str:
        lines = [f"**[{index}]** {self.title}"]
        if self.authors:
            lines.append(f"    Authors: {self.authors}")
        info_parts = []
        if self.year:
            info_parts.append(f"Year: {self.year}")
        if self.citations:
            info_parts.append(f"Citations: {self.citations}")
        info_parts.append(f"Type: {self.source_type}")
        info_parts.append(f"Source: {self.source_name}")
        lines.append(f"    {' | '.join(info_parts)}")
        if self.venue:
            lines.append(f"    Venue: {self.venue}")
        if self.url:
            lines.append(f"    URL: {self.url}")
        if self.summary:
            lines.append(f"    Summary: {self.summary[:350]}")
        return "\n".join(lines)


class BaseSource:
    """Base class for all information sources."""

    name: str = "base"
    source_type: str = "unknown"

    def __init__(self) -> None:
        self.available = _requests_module is not None
        self._requests = _requests_module

    def search(self, queries: list[str], max_results: int = 15) -> list[SearchResult]:
        raise NotImplementedError

    @staticmethod
    def _truncate_abstract(text: str, max_sentences: int = 2) -> str:
        if not text:
            return ""
        text = re.sub(r"<[^>]+>", "", text)
        sentences = text.split(". ")
        short = ". ".join(sentences[:max_sentences]).strip()
        if short and not short.endswith("."):
            short += "."
        return short