"""OpenAlex academic paper search."""

from __future__ import annotations

import logging
import time

from . import BaseSource, SearchResult


class OpenAlexSource(BaseSource):
    name = "openalex"
    source_type = "paper"
    API_BASE = "https://api.openalex.org/works"

    def search(self, queries: list[str], max_results: int = 15) -> list[SearchResult]:
        if not self.available:
            return []
        results: list[SearchResult] = []
        seen_ids: set[str] = set()

        for query in queries:
            for paper in self._fetch(query, per_page=10, sort="cited_by_count:desc"):
                pid = paper.get("id", "")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    results.append(self._to_result(paper))
            time.sleep(0.3)

        for query in queries[:4]:
            for paper in self._fetch(query, per_page=6, sort="publication_date:desc", year_from=2019):
                pid = paper.get("id", "")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    results.append(self._to_result(paper))
            time.sleep(0.3)

        logging.info(f"[OpenAlex] {len(results)} results from {len(queries)} queries")
        return results

    def _fetch(
        self, query: str, per_page: int = 10,
        sort: str = "cited_by_count:desc", year_from: int | None = None,
    ) -> list[dict]:
        params: dict = {"search": query, "per_page": per_page, "sort": sort}
        if year_from:
            params["filter"] = f"publication_year:>{year_from}"
        try:
            resp = self._requests.get(self.API_BASE, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json().get("results", [])
        except Exception as exc:
            logging.warning(f"[OpenAlex] Request failed: {exc}")
        return []

    def _to_result(self, paper: dict) -> SearchResult:
        title = paper.get("title", "Unknown") or "Unknown"
        year = paper.get("publication_year", 0) or 0
        cites = paper.get("cited_by_count", 0) or 0

        doi = paper.get("doi", "") or ""
        if doi and not doi.startswith("http"):
            doi = f"https://doi.org/{doi}"

        authorships = paper.get("authorships", []) or []
        names = []
        for a in authorships[:3]:
            n = (a.get("author", {}) or {}).get("display_name", "")
            if n:
                names.append(n)
        author_str = ", ".join(names)
        if len(authorships) > 3:
            author_str += " et al."

        loc = paper.get("primary_location", {}) or {}
        venue = (loc.get("source", {}) or {}).get("display_name", "") or ""

        abstract = self._reconstruct_abstract(paper)

        return SearchResult(
            title=title, summary=abstract, source_type="paper",
            source_name="openalex", url=doi, date=str(year),
            year=year, authors=author_str, citations=cites,
            venue=venue, metadata={"openalex_id": paper.get("id", "")},
        )

    @staticmethod
    def _reconstruct_abstract(paper: dict) -> str:
        inv = paper.get("abstract_inverted_index")
        if not inv:
            return ""
        try:
            wp = []
            for word, positions in inv.items():
                for pos in positions:
                    wp.append((pos, word))
            wp.sort()
            text = " ".join(w for _, w in wp)
            sentences = text.split(". ")
            short = ". ".join(sentences[:2]).strip()
            if short and not short.endswith("."):
                short += "."
            return short
        except Exception:
            return ""