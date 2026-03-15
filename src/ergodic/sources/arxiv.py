"""arXiv preprint search."""

from __future__ import annotations

import logging
import time
import urllib.parse
import xml.etree.ElementTree as ET

from . import BaseSource, SearchResult


class ArXivSource(BaseSource):
    name = "arxiv"
    source_type = "preprint"
    API_BASE = "http://export.arxiv.org/api/query"

    def search(self, queries: list[str], max_results: int = 15) -> list[SearchResult]:
        if not self.available:
            return []
        results: list[SearchResult] = []
        seen_ids: set[str] = set()
        per_query = max(3, max_results // len(queries)) if queries else 5

        for query in queries:
            for entry in self._fetch(query, max_results=per_query):
                eid = entry.url
                if eid and eid not in seen_ids:
                    seen_ids.add(eid)
                    results.append(entry)
            time.sleep(0.5)

        logging.info(f"[arXiv] {len(results)} results from {len(queries)} queries")
        return results

    def _fetch(self, query: str, max_results: int = 5) -> list[SearchResult]:
        search_query = urllib.parse.quote(f"all:{query}")
        url = (
            f"{self.API_BASE}?search_query={search_query}"
            f"&start=0&max_results={max_results}"
            f"&sortBy=relevance&sortOrder=descending"
        )
        try:
            resp = self._requests.get(url, timeout=30)
            if resp.status_code != 200:
                return []
            return self._parse_atom(resp.text)
        except Exception as exc:
            logging.warning(f"[arXiv] Request failed: {exc}")
            return []

    def _parse_atom(self, xml_text: str) -> list[SearchResult]:
        results: list[SearchResult] = []
        try:
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(xml_text)
            for entry in root.findall("atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                title = (
                    title_el.text.strip().replace("\n", " ")
                    if title_el is not None and title_el.text
                    else "Unknown"
                )

                summary_el = entry.find("atom:summary", ns)
                summary = ""
                if summary_el is not None and summary_el.text:
                    summary = self._truncate_abstract(summary_el.text.strip().replace("\n", " "))

                link = ""
                for el in entry.findall("atom:link", ns):
                    if el.get("type") == "text/html" or el.get("rel") == "alternate":
                        link = el.get("href", "")
                        break
                if not link:
                    id_el = entry.find("atom:id", ns)
                    link = id_el.text if id_el is not None and id_el.text else ""

                authors = []
                for author in entry.findall("atom:author", ns):
                    name_el = author.find("atom:name", ns)
                    if name_el is not None and name_el.text:
                        authors.append(name_el.text)
                author_str = ", ".join(authors[:3])
                if len(authors) > 3:
                    author_str += " et al."

                published = entry.find("atom:published", ns)
                date_str = published.text[:10] if published is not None and published.text else ""
                year = int(date_str[:4]) if date_str and len(date_str) >= 4 else 0

                results.append(SearchResult(
                    title=title, summary=summary, source_type="preprint",
                    source_name="arxiv", url=link, date=date_str,
                    year=year, authors=author_str, venue="arXiv",
                ))
        except Exception as exc:
            logging.warning(f"[arXiv] XML parsing error: {exc}")
        return results