"""Wikipedia background knowledge search."""

from __future__ import annotations

import logging
import re
import time
import urllib.parse

from . import BaseSource, SearchResult


class WikipediaSource(BaseSource):
    name = "wikipedia"
    source_type = "encyclopedia"
    API_BASE = "https://en.wikipedia.org/w/api.php"

    def search(self, queries: list[str], max_results: int = 10) -> list[SearchResult]:
        if not self.available:
            return []
        results: list[SearchResult] = []
        seen_titles: set[str] = set()
        per_query = max(2, max_results // len(queries)) if queries else 3

        for query in queries[:4]:
            for entry in self._fetch(query, limit=per_query):
                if entry.title not in seen_titles:
                    seen_titles.add(entry.title)
                    results.append(entry)
            time.sleep(0.3)

        logging.info(f"[Wikipedia] {len(results)} results")
        return results

    def _fetch(self, query: str, limit: int = 3) -> list[SearchResult]:
        params = {
            "action": "query", "list": "search", "srsearch": query,
            "srlimit": limit, "format": "json", "srprop": "snippet|timestamp",
        }
        try:
            resp = self._requests.get(self.API_BASE, params=params, timeout=15)
            if resp.status_code != 200:
                return []
            data = resp.json()
            search_results = data.get("query", {}).get("search", [])
        except Exception:
            return []

        results: list[SearchResult] = []
        for item in search_results:
            title = item.get("title", "")
            snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
            timestamp = item.get("timestamp", "")
            year = int(timestamp[:4]) if timestamp and len(timestamp) >= 4 else 0
            extract = self._get_extract(title)
            summary = extract if extract else snippet
            url = (
                f"https://en.wikipedia.org/wiki/"
                f"{urllib.parse.quote(title.replace(' ', '_'))}"
            )
            results.append(SearchResult(
                title=f"[Wikipedia] {title}", summary=summary,
                source_type="encyclopedia", source_name="wikipedia",
                url=url, date=timestamp[:10] if timestamp else "",
                year=year, authors="Wikipedia contributors", venue="Wikipedia",
            ))
        return results

    def _get_extract(self, title: str) -> str:
        params = {
            "action": "query", "titles": title, "prop": "extracts",
            "exintro": True, "explaintext": True, "exsentences": 3, "format": "json",
        }
        try:
            resp = self._requests.get(self.API_BASE, params=params, timeout=10)
            if resp.status_code != 200:
                return ""
            pages = resp.json().get("query", {}).get("pages", {})
            for pid, page in pages.items():
                if pid == "-1":
                    continue
                return page.get("extract", "")
        except Exception:
            pass
        return ""