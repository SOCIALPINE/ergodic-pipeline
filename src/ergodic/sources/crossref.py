"""CrossRef paper metadata search."""

from __future__ import annotations

import logging
import time

from . import BaseSource, SearchResult


class CrossRefSource(BaseSource):
    name = "crossref"
    source_type = "paper"
    API_BASE = "https://api.crossref.org/works"

    def search(self, queries: list[str], max_results: int = 10) -> list[SearchResult]:
        if not self.available:
            return []
        results: list[SearchResult] = []
        seen_dois: set[str] = set()
        per_query = max(3, max_results // len(queries)) if queries else 5

        for query in queries[:5]:
            for entry in self._fetch(query, rows=per_query):
                doi = entry.url
                if doi and doi not in seen_dois:
                    seen_dois.add(doi)
                    results.append(entry)
            time.sleep(0.5)

        logging.info(f"[CrossRef] {len(results)} results")
        return results

    def _fetch(self, query: str, rows: int = 5) -> list[SearchResult]:
        params = {
            "query": query,
            "rows": rows,
            "sort": "relevance",
            "order": "desc",
            "select": (
                "DOI,title,author,published-print,published-online,"
                "container-title,is-referenced-by-count,abstract"
            ),
        }
        headers = {
            "User-Agent": "ERGODIC/0.9 (https://github.com/ergodic-pipeline/ergodic; "
                          "mailto:ergodic@example.com)"
        }
        try:
            resp = self._requests.get(
                self.API_BASE, params=params, headers=headers, timeout=20,
            )
            if resp.status_code != 200:
                return []
            items = resp.json().get("message", {}).get("items", [])
        except Exception:
            return []

        results: list[SearchResult] = []
        for item in items:
            titles = item.get("title", [])
            title = titles[0] if titles else "Unknown"

            authors_raw = item.get("author", [])
            names = []
            for a in authors_raw[:3]:
                given = a.get("given", "")
                family = a.get("family", "")
                if family:
                    names.append(f"{given} {family}".strip())
            author_str = ", ".join(names)
            if len(authors_raw) > 3:
                author_str += " et al."

            date_parts = item.get("published-print", {}) or item.get("published-online", {})
            parts = (date_parts or {}).get("date-parts", [[]])
            year = parts[0][0] if parts and parts[0] else 0

            doi = item.get("DOI", "")
            url = f"https://doi.org/{doi}" if doi else ""

            containers = item.get("container-title", [])
            venue = containers[0] if containers else ""
            cites = item.get("is-referenced-by-count", 0) or 0

            abstract = self._truncate_abstract(item.get("abstract", "") or "")

            results.append(SearchResult(
                title=title, summary=abstract, source_type="paper",
                source_name="crossref", url=url, date=str(year) if year else "",
                year=year, authors=author_str, citations=cites, venue=venue,
            ))
        return results