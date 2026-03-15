"""
ERGODIC Pipeline — Multi-Agent Research Ideation Engine
========================================================
Emergent Recursive Generation Over Distributed Interpretation Cycles

Transforms random noise into emergent research ideas through
multi-source information gathering, multi-agent critique, and
recursive synthesis.

Pipeline topology:
  L0(Multi-API+LLM) → A0 → A1→A2,A3→A4,A5→S1→S0→F0→R1,R2→RS
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import string
import time
from dataclasses import asdict
from datetime import datetime
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from .prompts import (
    ARCHITECT_SYSTEM,
    FORMALIZE_F0,
    GOAL_ANALYST_SYSTEM,
    REVIEW,
    REVIEW_SUMMARY,
    SUMMARY_S1,
    SYNTHESIS_S0,
)
from .sources import BaseSource, SearchResult
from .sources.arxiv import ArXivSource
from .sources.crossref import CrossRefSource
from .sources.openalex import OpenAlexSource
from .sources.wikipedia import WikipediaSource

__all__ = ["ErgodicConfig", "ErgodicPipeline", "generate_noise"]

# ============================================================
# Source Registry & Minimum Guarantees
# ============================================================

SOURCE_REGISTRY: dict[str, type[BaseSource]] = {
    "openalex": OpenAlexSource,
    "arxiv": ArXivSource,
    "wikipedia": WikipediaSource,
    "crossref": CrossRefSource,
}

MIN_PER_SOURCE: dict[str, int] = {
    "openalex": 8,
    "arxiv": 3,
    "crossref": 4,
    "wikipedia": 2,
}


# ============================================================
# Noise Generator
# ============================================================

def generate_noise(length: int = 64, seed: Optional[int] = None) -> str:
    """Generate a random noise string to seed creative divergence."""
    if seed is not None:
        random.seed(seed)
    charset = string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;:',.<>?/~`"
    charset += "가나다라마바사아자차카타파하"
    return "".join(random.choice(charset) for _ in range(length))


# ============================================================
# Information Scout (L0)
# ============================================================

class InformationScout:
    """L0: Multi-source information gathering with adaptive search and LLM judging."""

    def __init__(self, max_results: int = 25, llm=None) -> None:
        self.max_results = max_results
        self.llm = llm
        self.sources: dict[str, BaseSource] = {}
        for name, cls in SOURCE_REGISTRY.items():
            src = cls()
            if src.available:
                self.sources[name] = src

    @staticmethod
    def _is_supplementary(result: SearchResult) -> bool:
        url = result.url.lower()
        if re.search(r"\.[s]\d{3,}$", url):
            return True
        if re.search(r"/s\d+$", url):
            return True
        if any(kw in url for kw in ["/suppl", "/supporting", "/si_"]):
            return True
        title_lower = result.title.lower()
        return any(
            kw in title_lower
            for kw in ["supporting information", "supplementary", "table of contents"]
        )

    def _route_sources(self, goal: str) -> dict[str, float]:
        if self.llm:
            routed = self._route_with_llm(goal)
            if routed:
                return routed
        return self._route_heuristic(goal)

    def _route_with_llm(self, goal: str) -> dict[str, float]:
        available = list(self.sources.keys())
        prompt = (
            f"Given a research goal, assign weights to information sources.\n"
            f"Available: {', '.join(available)}\n\n"
            f"Sources: openalex=academic papers, arxiv=preprints, "
            f"wikipedia=background, crossref=paper metadata\n\n"
            f'GOAL: "{goal}"\n\n'
            f"Weights must sum to 1.0, min 0.05 each. Output ONLY JSON."
        )
        try:
            resp = self.llm.invoke([
                SystemMessage(content="Output only a JSON object with source weights."),
                HumanMessage(content=prompt),
            ])
            raw = re.sub(r"```json\s*|```\s*", "", resp.content.strip())
            weights = json.loads(raw)
            valid = {k: float(v) for k, v in weights.items()
                     if k in self.sources and isinstance(v, (int, float)) and v > 0}
            if valid:
                total = sum(valid.values())
                return {k: v / total for k, v in valid.items()}
        except Exception as exc:
            logging.warning(f"L0: LLM routing failed: {exc}")
        return {}

    def _route_heuristic(self, goal: str) -> dict[str, float]:
        gl = goal.lower()
        cs_kw = {"ai", "machine learning", "deep learning", "neural", "algorithm",
                 "transformer", "llm"}
        prac_kw = {"business", "market", "startup", "product", "urban", "city", "policy"}
        cs = sum(1 for k in cs_kw if k in gl)
        prac = sum(1 for k in prac_kw if k in gl)
        available = set(self.sources.keys())
        if cs >= 2:
            w = {"openalex": 0.25, "arxiv": 0.45, "wikipedia": 0.1, "crossref": 0.2}
        elif prac >= 2:
            w = {"openalex": 0.25, "arxiv": 0.1, "wikipedia": 0.35, "crossref": 0.3}
        else:
            w = {"openalex": 0.4, "arxiv": 0.2, "wikipedia": 0.15, "crossref": 0.25}
        valid = {k: v for k, v in w.items() if k in available}
        total = sum(valid.values())
        return {k: v / total for k, v in valid.items()} if valid else {}

    def _generate_queries(self, goal: str) -> list[str]:
        if self.llm:
            qs = self._generate_queries_with_llm(goal)
            if qs:
                return qs
        return self._generate_queries_heuristic(goal)

    def _generate_queries_with_llm(self, goal: str) -> list[str]:
        prompt = (
            f"Generate exactly 8 search queries for this research goal.\n"
            f'GOAL: "{goal}"\n'
            f"Rules: 2-5 words each, diverse angles. One per line."
        )
        try:
            resp = self.llm.invoke([
                SystemMessage(content="Output exactly 8 search queries, one per line."),
                HumanMessage(content=prompt),
            ])
            queries = []
            for line in resp.content.strip().split("\n"):
                q = re.sub(r"^[Q]?\d+[.\):]?\s*", "", line.strip()).strip("\"'-.").strip()
                if q and len(q) > 6:
                    queries.append(q)
            return queries[:8] if len(queries) >= 4 else []
        except Exception:
            return []

    def _generate_queries_heuristic(self, goal: str) -> list[str]:
        stops = {"a", "an", "the", "is", "are", "and", "but", "or", "of", "in", "on",
                 "at", "to", "for", "with", "design", "novel", "new", "propose", "develop",
                 "method", "approach", "based", "using", "material", "system"}
        kw = [w for w in re.findall(r"[a-zA-Z]+", goal.lower())
              if w not in stops and len(w) > 2]
        qs = []
        if len(kw) >= 3:
            qs.append(" ".join(kw[:4]))
        if len(kw) >= 5:
            qs.append(" ".join(kw[2:6]))
        if len(kw) >= 3:
            qs.append(" ".join(kw[:3]) + " review")
            qs.append(" ".join(kw[:2]) + " advances")
            qs.append(" ".join(kw[1:4]))
        return qs[:8] or [goal[:100]]

    def _generate_adaptive_queries(self, goal: str, existing: list[SearchResult]) -> list[str]:
        if not self.llm or len(existing) < 5:
            return []
        titles = "\n".join(f"- {r.title[:80]}" for r in existing[:15])
        prompt = (
            f'GOAL: "{goal}"\nEXISTING:\n{titles}\n\n'
            f"Generate 3 COMPLEMENTARY queries for approaches NOT covered. One per line."
        )
        try:
            resp = self.llm.invoke([
                SystemMessage(content="Output exactly 3 queries, one per line."),
                HumanMessage(content=prompt),
            ])
            return [re.sub(r"^[Q]?\d+[.\):]?\s*", "", l.strip()).strip("\"'-.").strip()
                    for l in resp.content.strip().split("\n")
                    if len(l.strip()) > 6][:3]
        except Exception:
            return []

    def _llm_judge_relevance(self, goal: str, borderline: list[SearchResult]) -> list[SearchResult]:
        if not self.llm or not borderline:
            return []
        items = "\n".join(f"{i+1}. {r.title}\n   {r.summary[:200]}"
                          for i, r in enumerate(borderline[:10]))
        prompt = (
            f'Goal: "{goal}"\nResults with uncertain relevance:\n{items}\n\n'
            f"Output ONLY numbers of relevant results, comma-separated. If none: NONE"
        )
        try:
            resp = self.llm.invoke([
                SystemMessage(content="Output only numbers, comma-separated."),
                HumanMessage(content=prompt),
            ])
            raw = resp.content.strip()
            if raw.upper() == "NONE":
                return []
            indices = {int(n) - 1 for n in re.findall(r"\d+", raw)}
            return [r for i, r in enumerate(borderline[:10]) if i in indices]
        except Exception:
            return []

    def _extract_filter_config(self, goal: str) -> dict:
        stops = {"a", "an", "the", "is", "are", "and", "but", "or", "of", "in", "on",
                 "at", "to", "for", "with", "that", "this", "design", "novel", "new",
                 "propose", "develop", "using", "based", "via", "method", "approach"}
        generic = {"material", "materials", "conditions", "high", "low", "efficient",
                   "effective", "improved", "better", "performance", "properties",
                   "structure", "application", "framework"}
        words = re.findall(r"[a-zA-Z]+", goal.lower())
        all_kw = [w for w in words if w not in stops and len(w) > 2]
        specific = [w for w in all_kw if w not in generic and len(w) > 3]
        groups = self._extract_topic_groups(goal)
        neg = self._generate_negative_keywords(goal)
        return {"topic_groups": groups, "bonus": set(specific), "negative": neg}

    def _extract_topic_groups(self, goal: str) -> list[set[str]]:
        if self.llm:
            g = self._extract_topic_groups_llm(goal)
            if g:
                return g
        return self._extract_topic_groups_heuristic(goal)

    def _extract_topic_groups_llm(self, goal: str) -> list[set[str]]:
        prompt = (
            f"Identify 2 topic groups for filtering.\n"
            f"GROUP1=SUBJECT GROUP2=ACTION/APPLICATION, 5+ terms each.\n"
            f'GOAL: "{goal}"\nOutput 2 lines of comma-separated keywords.'
        )
        try:
            resp = self.llm.invoke([
                SystemMessage(content="Output exactly 2 lines of comma-separated keywords."),
                HumanMessage(content=prompt),
            ])
            groups = []
            for line in resp.content.strip().split("\n"):
                line = re.sub(r"^(GROUP\s*\d+\s*[:.]?\s*)", "", line.strip(), flags=re.IGNORECASE)
                line = re.sub(r"^\d+[.\):]?\s*", "", line)
                terms = set()
                for t in line.split(","):
                    t = t.strip().lower()
                    if len(t) > 1:
                        terms.add(t)
                        for w in t.split():
                            if len(w) > 2:
                                terms.add(w)
                if terms:
                    groups.append(terms)
            return groups[:2] if len(groups) == 2 else []
        except Exception:
            return []

    def _extract_topic_groups_heuristic(self, goal: str) -> list[set[str]]:
        stops = {"a", "an", "the", "is", "of", "in", "on", "at", "to", "for", "with",
                 "design", "novel", "new", "propose", "develop", "that", "and", "or"}
        generic = {"material", "system", "framework", "conditions", "method", "structure"}
        kw = [w for w in re.findall(r"[a-zA-Z]+", goal.lower())
              if w not in stops and w not in generic and len(w) > 3]
        if len(kw) < 2:
            return [set(kw)] if kw else []
        mid = len(kw) // 2
        return [set(kw[:mid]), set(kw[mid:])]

    def _generate_negative_keywords(self, goal: str) -> set[str]:
        base: set[str] = set()
        if self.llm:
            try:
                prompt = f'List 15-25 DEFINITELY IRRELEVANT keywords for:\n"{goal}"\nOne per line.'
                resp = self.llm.invoke([
                    SystemMessage(content="Output only keywords, one per line."),
                    HumanMessage(content=prompt),
                ])
                for line in resp.content.strip().split("\n"):
                    w = line.strip().lower().strip("-•").strip()
                    if w and len(w) > 2 and " " not in w:
                        base.add(w)
            except Exception:
                pass
        if len(base) < 5:
            base = {"recipe", "cooking", "fashion", "celebrity", "gossip",
                    "horoscope", "astrology", "lottery", "gambling"}
        return base

    def _score_relevance(self, result: SearchResult, config: dict) -> float:
        text = (result.title + " " + result.summary).lower()
        if not text.strip():
            return 0.0
        groups = config["topic_groups"]
        bonus_kw = config["bonus"]
        neg_kw = config["negative"]
        title_lower = result.title.lower()

        if sum(1 for kw in neg_kw if kw in title_lower) >= 2:
            return 0.0
        neg_pen = sum(0.15 for kw in neg_kw if kw in text)

        gs = [sum(1 for t in g if t in text) for g in groups]
        if groups and any(s == 0 for s in gs):
            if result.source_name == "wikipedia":
                if all(s == 0 for s in gs):
                    return 0.0
            else:
                return 0.0

        score = sum(min(s, 3) * 0.15 for s in gs)
        bonus_only = bonus_kw - {t for g in groups for t in g}
        score += sum(1 for kw in bonus_only if kw in text) * 0.05
        all_imp = {t for g in groups for t in g}
        score += sum(1 for kw in all_imp if kw in title_lower) * 0.1
        if result.source_name in ("openalex", "crossref") and result.citations > 50:
            score += 0.1
        if result.source_name == "arxiv" and result.year >= 2023:
            score += 0.05
        return max(score - neg_pen, 0.0)

    def search(self, goal: str) -> str:
        if not self.sources:
            return f"## INFORMATION SURVEY (L0)\nStatus: All sources unavailable.\nGoal: {goal}"

        logging.info("L0: Multi-source search starting...")
        queries = self._generate_queries(goal)
        weights = self._route_sources(goal)
        logging.info(f"L0: Source weights: {weights}")

        all_results: list[SearchResult] = []
        source_stats: dict[str, int] = {}
        for src_name, weight in weights.items():
            if src_name not in self.sources:
                continue
            src = self.sources[src_name]
            src_max = max(8, int(self.max_results * weight * 2.0))
            try:
                results = src.search(queries, max_results=src_max)
                all_results.extend(results)
                source_stats[src_name] = len(results)
            except Exception as exc:
                logging.warning(f"L0: {src_name} failed: {exc}")
                source_stats[src_name] = 0

        before = len(all_results)
        all_results = [r for r in all_results if not self._is_supplementary(r)]
        removed = before - len(all_results)
        if removed:
            logging.info(f"L0: Removed {removed} supplementary materials")

        config = self._extract_filter_config(goal)
        for r in all_results:
            r.relevance_score = self._score_relevance(r, config)

        scored = sorted(
            [r for r in all_results if r.relevance_score > 0],
            key=lambda r: r.relevance_score, reverse=True,
        )

        borderline = [r for r in all_results if 0.10 <= r.relevance_score <= 0.25]
        if borderline:
            approved = self._llm_judge_relevance(goal, borderline)
            for r in approved:
                r.relevance_score = max(r.relevance_score, 0.30)
            logging.info(f"L0: LLM approved {len(approved)}/{len(borderline)} borderline")

        high = [r for r in scored if r.relevance_score >= 0.3]
        if len(high) < 10:
            adaptive = self._generate_adaptive_queries(goal, scored[:15])
            if adaptive:
                logging.info(f"L0: Adaptive 2nd-round with {len(adaptive)} queries")
                for sn in ["openalex", "crossref"]:
                    if sn in self.sources:
                        try:
                            extra = self.sources[sn].search(adaptive, max_results=8)
                            extra = [r for r in extra if not self._is_supplementary(r)]
                            for r in extra:
                                r.relevance_score = self._score_relevance(r, config)
                            all_results.extend(extra)
                            scored.extend(r for r in extra if r.relevance_score > 0)
                        except Exception:
                            pass
                scored.sort(key=lambda r: r.relevance_score, reverse=True)

        filtered = [r for r in scored if r.relevance_score >= 0.2]
        if len(filtered) < 8:
            filtered = [r for r in scored if r.relevance_score > 0]
        if len(filtered) < 5:
            filtered = all_results

        final_ids = set(id(r) for r in filtered)
        for src_name, min_count in MIN_PER_SOURCE.items():
            src_results = [r for r in scored
                           if r.source_name == src_name and id(r) not in final_ids]
            in_filtered = sum(1 for r in filtered if r.source_name == src_name)
            needed = max(0, min_count - in_filtered)
            for r in src_results[:needed]:
                filtered.append(r)
                final_ids.add(id(r))

        def sort_key(r: SearchResult) -> float:
            recency = max(0, (r.year - 2010) * 3) if r.year > 2010 else 0
            return r.relevance_score * 100 + r.citations * 0.01 + recency * 0.1

        filtered.sort(key=sort_key, reverse=True)
        top = filtered[: self.max_results]
        logging.info(f"L0: {len(all_results)} raw → {len(top)} final")
        return self._format_report(top, goal, queries, weights, source_stats)

    def _format_report(self, results, goal, queries, weights, source_stats) -> str:
        lines = [
            "## INFORMATION SURVEY (L0 — Multi-Source)",
            f"Goal: {goal[:100]}...",
            f"Results found: {len(results)} (after relevance filtering)",
            "", "### SOURCES USED",
        ]
        for src, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  - {src}: weight={w:.2f}, raw={source_stats.get(src, 0)}")
        if queries:
            lines.append(f"\n### QUERIES ({len(queries)})")
            for i, q in enumerate(queries, 1):
                lines.append(f"  Q{i}: {q}")
        lines.extend(["", "### EXISTING WORK", ""])

        by_source: dict[str, list[SearchResult]] = {}
        for r in results:
            by_source.setdefault(r.source_name, []).append(r)
        idx = 1
        labels = {"openalex": "Academic Papers (OpenAlex)",
                  "crossref": "Academic Papers (CrossRef)",
                  "arxiv": "Preprints (arXiv)",
                  "wikipedia": "Background Knowledge (Wikipedia)"}
        for sn in ["openalex", "crossref", "arxiv", "wikipedia"]:
            items = by_source.get(sn, [])
            if not items:
                continue
            lines.append(f"#### {labels.get(sn, sn)}")
            lines.append("")
            for r in items:
                lines.append(r.to_report_entry(idx))
                lines.append("")
                idx += 1

        years = [r.year for r in results if r.year > 0]
        recent = [y for y in years if y >= 2020]
        high_cite = [r for r in results if r.citations > 100]
        lines.extend([
            "### SURVEY STATISTICS",
            f"- Total results: {len(results)}",
            f"- From 2020+: {len(recent)}/{len(results)}",
            f"- Highly cited (>100): {len(high_cite)}",
            "",
            "### WHAT ALREADY EXISTS",
            "The above results represent the current landscape. "
            "A0 should identify gaps and novel directions NOT covered.",
        ])
        return "\n".join(lines)


# ============================================================
# Configuration
# ============================================================

class ErgodicConfig:
    """Pipeline configuration."""

    GOOGLE_API_KEY: str = ""
    MODEL_NAME: str = "gemini-2.5-flash-lite"

    INFORMATION_SEARCH: bool = True
    MAX_RESULTS: int = 25

    GOAL_ANALYST_TEMPERATURE: float = 0.3
    GOAL_ANALYST_MAX_TOKENS: int = 2500
    ARCHITECT_TEMPERATURE: float = 0.9
    ARCHITECT_MAX_TOKENS: int = 1200
    SUMMARY_TEMPERATURE: float = 0.2
    SUMMARY_MAX_TOKENS: int = 800
    SYNTHESIS_MAX_TOKENS: int = 4000
    FORMALIZE_TEMPERATURE: float = 0.2
    FORMALIZE_MAX_TOKENS: int = 3000
    REVIEW_TEMPERATURE: float = 0.3
    REVIEW_MAX_TOKENS: int = 1500
    REVIEW_SUMMARY_MAX_TOKENS: int = 2000

    NUM_CYCLES: int = 2
    DELAY_SECONDS: int = 20

    NOISE_LENGTH: int = 64
    NOISE_SEED: Optional[int] = None

    GOAL: str = ""

    OUTPUT_DIR: str = "./ergodic_output"
    LOG_FILE: str = "ergodic_run.log"
    CHECKPOINT_FILE: str = "checkpoint.json"


# ============================================================
# Semantic Memory
# ============================================================

class SemanticMemory:
    def __init__(self) -> None:
        self.core_ideas: list[str] = []
        self.decisions: list[str] = []
        self.unresolved: list[str] = []
        self.cycle_history: list[str] = []

    def to_context_string(self) -> str:
        if not any([self.core_ideas, self.decisions, self.cycle_history]):
            return "[No prior memory — first cycle.]"
        parts = []
        if self.core_ideas:
            parts.append("CORE IDEAS:\n" + "\n".join(f"  - {i}" for i in self.core_ideas[-5:]))
        if self.decisions:
            parts.append("DECISIONS:\n" + "\n".join(f"  - {d}" for d in self.decisions[-5:]))
        return "\n\n".join(parts)

    def update_from_output(self, output: str) -> None:
        lines = [l.strip() for l in output.split("\n") if len(l.strip()) > 20]
        key = [l for l in lines if any(k in l.lower() for k in
               ["=", "→", "novel", "propose", "gap", "model", "design", "structure"])]
        self.core_ideas.append((key[0] if key else lines[0] if lines else "")[:200])
        self.core_ideas = self.core_ideas[-8:]
        self.decisions = self.decisions[-8:]

    def add_cycle_summary(self, cycle_num: int, summary: str) -> None:
        self.cycle_history.append(f"[Cycle {cycle_num}] {summary[:200]}")
        self.cycle_history = self.cycle_history[-5:]

    def to_dict(self) -> dict:
        return {"core_ideas": self.core_ideas, "decisions": self.decisions,
                "unresolved": self.unresolved, "cycle_history": self.cycle_history}

    def from_dict(self, d: dict) -> None:
        self.core_ideas = d.get("core_ideas", [])
        self.decisions = d.get("decisions", [])
        self.unresolved = d.get("unresolved", [])
        self.cycle_history = d.get("cycle_history", [])


# ============================================================
# Progress Display
# ============================================================

def _show_progress(current: int, total: int, agent_id: str, phase: str) -> None:
    pct = current / total * 100
    filled = int(30 * current / total)
    bar = "█" * filled + "░" * (30 - filled)
    print(f"\r  [{bar}] {pct:5.1f}% | Step {current}/{total} | {phase} → {agent_id}",
          end="", flush=True)


# ============================================================
# Agent
# ============================================================

PROMPT_MAP = {
    "goal_analyst": GOAL_ANALYST_SYSTEM,
    "architect": ARCHITECT_SYSTEM,
    "summary": SUMMARY_S1,
    "synthesis": SYNTHESIS_S0,
    "formalize": FORMALIZE_F0,
    "review": REVIEW,
    "review_summary": REVIEW_SUMMARY,
}


class Agent:
    def __init__(self, agent_id: str, agent_type: str, lm) -> None:
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.lm = lm
        self.memory = SemanticMemory()
        self.last_output = ""
        self.call_count = 0

    def call(self, input_text: str, context: str = "", delay: int = 20) -> str:
        system = PROMPT_MAP.get(self.agent_type, "")
        mem_ctx = self.memory.to_context_string()

        if self.agent_type == "goal_analyst":
            human = input_text
        elif self.agent_type == "architect":
            human = f"YOUR MEMORY:\n{mem_ctx}\n\n---\n\nINPUT:\n{input_text}"
            if context:
                human += f"\n\nCONTEXT:\n{context}"
        elif self.agent_type == "synthesis":
            human = f"YOUR MEMORY:\n{mem_ctx}\n\n---\n\nSYNTHESIZE:\n\n{input_text}"
        else:
            human = input_text

        messages = [SystemMessage(content=system), HumanMessage(content=human)]
        logging.info(f"[{self.agent_id}] Calling LLM (#{self.call_count + 1})...")

        output = None
        for attempt in range(3):
            try:
                output = self.lm.invoke(messages).content
                break
            except Exception as exc:
                wait = 30 * (attempt + 1)
                logging.error(f"[{self.agent_id}] Attempt {attempt+1} failed: {exc}")
                if attempt < 2:
                    time.sleep(wait)

        if output is None:
            output = "[ERROR: All attempts failed]"
        self.last_output = output
        self.call_count += 1
        self.memory.update_from_output(output)
        logging.info(f"[{self.agent_id}] Response ({len(output)} chars)")
        if delay > 0:
            time.sleep(delay)
        return output

    def to_dict(self) -> dict:
        return {"agent_id": self.agent_id, "agent_type": self.agent_type,
                "last_output": self.last_output, "call_count": self.call_count,
                "memory": self.memory.to_dict()}

    def from_dict(self, d: dict) -> None:
        self.last_output = d.get("last_output", "")
        self.call_count = d.get("call_count", 0)
        if "memory" in d:
            self.memory.from_dict(d["memory"])


# ============================================================
# Checkpoint
# ============================================================

class CheckpointManager:
    def __init__(self, output_dir: str, filename: str) -> None:
        self.filepath = os.path.join(output_dir, filename)

    def save(self, state: dict) -> None:
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load(self) -> dict | None:
        if not os.path.exists(self.filepath):
            return None
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def clear(self) -> None:
        if os.path.exists(self.filepath):
            os.remove(self.filepath)


# ============================================================
# Pipeline
# ============================================================

class ErgodicPipeline:
    """
    ERGODIC Pipeline — Multi-Agent Research Ideation.

    Topology: L0(Multi-API+LLM) → A0 → A1→A2,A3→A4,A5→S1→S0→F0→R1,R2→RS
    """

    STEPS_PER_CYCLE = [
        ("FORWARD", "A1"), ("FORWARD", "A2"), ("FORWARD", "A3"),
        ("FORWARD", "A4"), ("FORWARD", "A5"),
        ("BACKWARD", "S1"), ("BACKWARD", "S0"),
        ("FORMALIZE", "F0"),
        ("REVIEW", "R1"), ("REVIEW", "R2"), ("REVIEW", "RS"),
    ]

    def __init__(self, config: ErgodicConfig) -> None:
        self.config = config
        self.cycle_results: list[dict] = []
        self.survey_report = ""
        self.goal_brief = ""

        if not config.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY is required. Set it via environment variable "
                "or pass it in ErgodicConfig."
            )

        kw = {"google_api_key": config.GOOGLE_API_KEY, "model": config.MODEL_NAME}

        self.goal_analyst_lm = ChatGoogleGenerativeAI(
            temperature=config.GOAL_ANALYST_TEMPERATURE,
            max_output_tokens=config.GOAL_ANALYST_MAX_TOKENS, **kw)
        self.scout = InformationScout(max_results=config.MAX_RESULTS, llm=self.goal_analyst_lm)

        self.architect_lm = ChatGoogleGenerativeAI(
            temperature=config.ARCHITECT_TEMPERATURE,
            max_output_tokens=config.ARCHITECT_MAX_TOKENS, **kw)
        self.summary_lm = ChatGoogleGenerativeAI(
            temperature=config.SUMMARY_TEMPERATURE,
            max_output_tokens=config.SUMMARY_MAX_TOKENS, **kw)
        self.synthesis_lm = ChatGoogleGenerativeAI(
            temperature=0.3, max_output_tokens=config.SYNTHESIS_MAX_TOKENS, **kw)
        self.formalize_lm = ChatGoogleGenerativeAI(
            temperature=config.FORMALIZE_TEMPERATURE,
            max_output_tokens=config.FORMALIZE_MAX_TOKENS, **kw)
        self.review_lm = ChatGoogleGenerativeAI(
            temperature=config.REVIEW_TEMPERATURE,
            max_output_tokens=config.REVIEW_MAX_TOKENS, **kw)
        self.review_summary_lm = ChatGoogleGenerativeAI(
            temperature=0.1, max_output_tokens=config.REVIEW_SUMMARY_MAX_TOKENS, **kw)

        self.agents = {
            "A0": Agent("A0", "goal_analyst", self.goal_analyst_lm),
            "A1": Agent("A1", "architect", self.architect_lm),
            "A2": Agent("A2", "architect", self.architect_lm),
            "A3": Agent("A3", "architect", self.architect_lm),
            "A4": Agent("A4", "architect", self.architect_lm),
            "A5": Agent("A5", "architect", self.architect_lm),
            "S1": Agent("S1", "summary", self.summary_lm),
            "S0": Agent("S0", "synthesis", self.synthesis_lm),
            "F0": Agent("F0", "formalize", self.formalize_lm),
            "R1": Agent("R1", "review", self.review_lm),
            "R2": Agent("R2", "review", self.review_lm),
            "RS": Agent("RS", "review_summary", self.review_summary_lm),
        }

        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        self.checkpoint = CheckpointManager(config.OUTPUT_DIR, config.CHECKPOINT_FILE)
        log_path = os.path.join(config.OUTPUT_DIR, config.LOG_FILE)
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s | %(message)s",
            handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                      logging.StreamHandler()],
            force=True,
        )

    # ---- L0 & A0 ----

    def _run_information_search(self) -> str:
        if self.survey_report:
            return self.survey_report
        if not self.config.INFORMATION_SEARCH:
            self.survey_report = "## INFORMATION SURVEY (L0)\nStatus: Disabled."
            return self.survey_report
        print("  ▶ L0: Multi-source search...")
        self.survey_report = self.scout.search(self.config.GOAL)
        logging.info(f"L0: Survey report ({len(self.survey_report)} chars)")
        return self.survey_report

    def _run_goal_analysis(self, delay: int) -> str:
        if self.goal_brief:
            return self.goal_brief
        print("  ▶ A0: Analyzing goal...")
        a0_input = (
            f"GOAL:\n{self.config.GOAL}\n\n"
            f"=== INFORMATION SURVEY FROM L0 ===\n{self.survey_report}\n"
            f"=== END SURVEY ===\n\n"
            f"Produce a Goal Brief. Cite sources by [number]. "
            f"If L0 results are sparse (<15), also use domain knowledge "
            f"(marked as [A0 domain knowledge])."
        )
        self.goal_brief = self.agents["A0"].call(a0_input, delay=delay)
        return self.goal_brief

    # ---- Noise Prompts ----

    def _build_a1_noise_prompt(self, noise: str) -> str:
        return (
            f"=== GOAL BRIEF ===\n{self.goal_brief}\n=== END ===\n\n"
            f"NOISE CATALYST:\n{noise}\n\n"
            f"GOAL:\n{self.config.GOAL}\n\n"
            f"Use noise as creative fuel, DIRECTION from GAPS.\n"
            f"You MUST: 1) State which GAP  2) Why no overlap with cited sources  "
            f"3) Domain terminology  4) Meet success criterion"
        )

    def _build_a2_critique(self, a1_out: str, noise: str, ctx: str) -> str:
        return (
            f"YOUR UNIQUE NOISE:\n{noise}\n\n"
            f"=== A1's PROPOSAL ===\n{a1_out}\n\n{ctx}\n\n"
            f"1. CRITIQUE A1 against Goal Brief.\n"
            f"2. Propose your OWN DIFFERENT improvement from YOUR noise.\n"
            f"3. Must DIFFER from A1."
        )

    def _build_a3_critique(self, a1_out: str, noise: str, ctx: str) -> str:
        return (
            f"YOUR UNIQUE NOISE (different from A1 and A2):\n{noise}\n\n"
            f"=== A1's PROPOSAL ===\n{a1_out}\n\n{ctx}\n\n"
            f"1. CRITIQUE A1 against Goal Brief.\n"
            f"2. Propose your OWN DIFFERENT improvement from YOUR noise.\n"
            f"3. Must DIFFER from BOTH A1 and obvious alternatives."
        )

    def _build_revision_prompt(self, cycle_num: int) -> str:
        prev = self.cycle_results[-1] if self.cycle_results else {}
        prev_s0 = prev.get("S0", "[none]")
        prev_f0 = prev.get("F0", "[none]")
        prev_rs = prev.get("RS", "[none]")

        name = "[unnamed]"
        for line in prev_s0.split("\n"):
            if "##" in line and "NAME" in line.upper():
                candidate = line.replace("#", "").strip()
                for pfx in ["PROPOSAL NAME:", "PROPOSAL NAME",
                             "ARCHITECTURE NAME:", "DESIGN NAME:"]:
                    candidate = candidate.replace(pfx, "").strip()
                if candidate and len(candidate) > 3:
                    name = candidate[:80]
                    break

        return (
            f"=== GOAL BRIEF ===\n{self.goal_brief}\n=== END ===\n\n"
            f"Cycle {cycle_num}. Proposal '{name}' was reviewed.\n\n"
            f"STRICT RULES:\n"
            f"1. KEEP '{name}' as name.\n"
            f"2. RETAIN 80%+ of core structure.\n"
            f"3. Do NOT propose entirely new idea.\n"
            f"4. ONLY address PRIORITIZED FIXES.\n"
            f"5. If wanting to discard — STOP, that violates Rule 2.\n\n"
            f"=== FULL PROPOSAL (S0) — KEEP 80%+ ===\n{prev_s0}\n\n"
            f"=== SPEC (F0) ===\n{prev_f0[:2000]}\n\n"
            f"=== REVIEW (RS) — FIX ONLY THESE ===\n{prev_rs}\n\n"
            f"GOAL: {self.config.GOAL}"
        )

    # ---- Restore Agents ----

    def _restore_agents(self, agents_dict: dict) -> None:
        for aid, state in agents_dict.items():
            if aid in self.agents:
                self.agents[aid].from_dict(state)

    # ---- Cycle ----

    def run_cycle(
        self, noise: str, cycle_num: int, start_step: int = 0,
        existing_results: dict | None = None, all_results: dict | None = None,
    ) -> dict:
        delay = self.config.DELAY_SECONDS
        results = existing_results or {}
        total = len(self.STEPS_PER_CYCLE) * self.config.NUM_CYCLES
        base = (cycle_num - 1) * len(self.STEPS_PER_CYCLE)

        brief_pfx = f"=== GOAL BRIEF ===\n{self.goal_brief}\n=== END ===\n\n"
        seed = self.config.NOISE_SEED or 42
        noise_a2 = generate_noise(self.config.NOISE_LENGTH, seed=seed + 1000 + cycle_num)
        noise_a3 = generate_noise(self.config.NOISE_LENGTH, seed=seed + 2000 + cycle_num)

        if cycle_num == 1:
            crit_ctx = (
                f"{brief_pfx}CRITIQUE vs Goal Brief and cited sources:\n"
                f"- Overlap? Which gaps addressed/missed?\n"
                f"- Meets success criteria?\n"
                f"After critique, propose BETTER.\n\n"
                f"GOAL: {self.config.GOAL}"
            )
        else:
            crit_ctx = (
                f"{brief_pfx}REVISION. Check:\n"
                f"1. Fixed reviewer problems?  2. No overlap?  3. All gaps?  "
                f"4. All criteria?\n"
                f"FORBIDDEN: entirely new designs. Fix, don't replace.\n\n"
                f"GOAL: {self.config.GOAL}"
            )

        for si in range(start_step, len(self.STEPS_PER_CYCLE)):
            phase, aid = self.STEPS_PER_CYCLE[si]
            _show_progress(base + si + 1, total, aid, phase)
            print()

            if phase == "FORWARD":
                if aid == "A1":
                    inp = (self._build_a1_noise_prompt(noise) if cycle_num == 1
                           else self._build_revision_prompt(cycle_num))
                    results["A1"] = self.agents["A1"].call(inp, delay=delay)

                elif aid == "A2":
                    if cycle_num == 1:
                        inp = self._build_a2_critique(results["A1"], noise_a2, crit_ctx)
                        results["A2"] = self.agents["A2"].call(inp, delay=delay)
                    else:
                        results["A2"] = self.agents["A2"].call(
                            results["A1"], context=crit_ctx, delay=delay)

                elif aid == "A3":
                    if cycle_num == 1:
                        inp = self._build_a3_critique(results["A1"], noise_a3, crit_ctx)
                        results["A3"] = self.agents["A3"].call(inp, delay=delay)
                    else:
                        results["A3"] = self.agents["A3"].call(
                            results["A1"], context=crit_ctx, delay=delay)

                elif aid == "A4":
                    layer = (f"=== AGENT 2 ===\n{results['A2']}\n\n"
                             f"=== AGENT 3 ===\n{results['A3']}")
                    results["A4"] = self.agents["A4"].call(
                        layer, context=crit_ctx, delay=delay)

                elif aid == "A5":
                    layer = (f"=== AGENT 2 ===\n{results['A2']}\n\n"
                             f"=== AGENT 3 ===\n{results['A3']}")
                    results["A5"] = self.agents["A5"].call(
                        layer, context=crit_ctx, delay=delay)

            elif phase == "BACKWARD":
                if aid == "S1":
                    inp = (f"{brief_pfx}=== AGENT 4 ===\n{results['A4']}\n\n"
                           f"=== AGENT 5 ===\n{results['A5']}")
                    results["S1"] = self.agents["S1"].call(inp, delay=delay)
                    fb = results["S1"][:300]
                    self.agents["A2"].memory.core_ideas.append(f"[Backward] {fb}")
                    self.agents["A3"].memory.core_ideas.append(f"[Backward] {fb}")

                elif aid == "S0":
                    inp = (f"{brief_pfx}=== AGENT 2 ===\n{results['A2']}\n\n"
                           f"=== AGENT 3 ===\n{results['A3']}\n\n"
                           f"=== SUMMARY ===\n{results['S1']}")
                    results["S0"] = self.agents["S0"].call(inp, delay=delay)
                    self.agents["A1"].memory.core_ideas.append(
                        f"[Synthesis] {results['S0'][:300]}")
                    self.agents["A1"].memory.add_cycle_summary(
                        cycle_num, results["S0"][:200])

            elif phase == "FORMALIZE":
                inp = f"{brief_pfx}=== PROPOSAL ===\n{results['S0']}"
                results["F0"] = self.agents["F0"].call(inp, delay=delay)

            elif phase == "REVIEW":
                review_inp = f"{brief_pfx}=== SPEC ===\n{results['F0']}"
                if aid == "R1":
                    results["R1"] = self.agents["R1"].call(review_inp, delay=delay)
                elif aid == "R2":
                    results["R2"] = self.agents["R2"].call(review_inp, delay=delay)
                elif aid == "RS":
                    combined = (f"=== REVIEW 1 ===\n{results['R1']}\n\n"
                                f"=== REVIEW 2 ===\n{results['R2']}")
                    results["RS"] = self.agents["RS"].call(combined, delay=delay)
                    rs_fb = results["RS"][:400]
                    for a in ["A1", "A2", "A3", "A4", "A5", "S0"]:
                        self.agents[a].memory.core_ideas.append(
                            f"[REVIEW cycle {cycle_num}] {rs_fb}")

            if all_results is not None:
                self._save_checkpoint(noise, cycle_num, si + 1, results, all_results)

        return results

    def _save_checkpoint(self, noise, cycle, step, results, all_results):
        state = {
            "noise": noise, "current_cycle": cycle, "current_step": step,
            "results": results, "all_results": all_results,
            "survey_report": self.survey_report, "goal_brief": self.goal_brief,
            "agents": {k: a.to_dict() for k, a in self.agents.items()},
        }
        self.checkpoint.save(state)

    # ---- Main Entry Point ----

    def run(self, resume: bool = True) -> dict:
        """Execute the full pipeline."""
        start_time = datetime.now()
        ckpt = self.checkpoint.load() if resume else None

        if ckpt:
            noise = ckpt["noise"]
            start_cycle = ckpt["current_cycle"]
            start_step = ckpt["current_step"]
            all_results = ckpt["all_results"]
            cycle_results_so_far = ckpt.get("results", {})
            self.survey_report = ckpt.get("survey_report", "")
            self.goal_brief = ckpt.get("goal_brief", "")
            self._restore_agents(ckpt.get("agents", {}))
        else:
            noise = generate_noise(self.config.NOISE_LENGTH, self.config.NOISE_SEED)
            start_cycle = 1
            start_step = 0
            cycle_results_so_far = {}
            all_results = {
                "config": {
                    "model": self.config.MODEL_NAME,
                    "cycles": self.config.NUM_CYCLES,
                    "noise": noise,
                    "goal": self.config.GOAL,
                    "start_time": str(start_time),
                    "version": "0.9.0",
                },
                "cycles": [],
            }

        delay = self.config.DELAY_SECONDS

        if not self.survey_report:
            self._run_information_search()
            all_results["survey_report"] = self.survey_report

        if not self.goal_brief:
            self._run_goal_analysis(delay)
            all_results["goal_brief"] = self.goal_brief

        print(f"\n{'='*60}")
        print(f"  ERGODIC Pipeline v0.9")
        print(f"  Goal: {self.config.GOAL[:70]}...")
        print(f"  {self.config.NUM_CYCLES} cycles × "
              f"{len(self.STEPS_PER_CYCLE)} steps + L0 + A0")
        print(f"  Noise: {noise[:40]}...")
        print(f"{'='*60}\n")

        for cycle in range(start_cycle, self.config.NUM_CYCLES + 1):
            print(f"\n  ── Cycle {cycle}/{self.config.NUM_CYCLES} ──\n")
            step = start_step if cycle == start_cycle else 0
            existing = cycle_results_so_far if cycle == start_cycle else {}
            cycle_data = self.run_cycle(
                noise, cycle, start_step=step,
                existing_results=existing, all_results=all_results,
            )
            all_results["cycles"].append({"cycle": cycle, "results": dict(cycle_data)})
            self.cycle_results.append(cycle_data)
            cycle_results_so_far = {}
            start_step = 0

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        total_calls = sum(a.call_count for a in self.agents.values())

        all_results["end_time"] = str(end_time)
        all_results["duration_seconds"] = duration
        all_results["total_llm_calls"] = total_calls

        # Save JSON results
        results_path = os.path.join(self.config.OUTPUT_DIR, "ergodic_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        # Save human-readable synthesis
        final_path = os.path.join(self.config.OUTPUT_DIR, "final_synthesis.txt")
        with open(final_path, "w", encoding="utf-8") as f:
            f.write(f"ERGODIC Pipeline — Final Output\n{'='*60}\n\n")
            f.write(f"Goal: {self.config.GOAL}\nNoise: {noise}\n")
            f.write(f"Cycles: {self.config.NUM_CYCLES} | Duration: {duration:.0f}s\n")
            f.write(f"Total LLM calls: {total_calls}\n\n")

            f.write(f"{'='*60}\nINFORMATION SURVEY (L0)\n{'='*60}\n\n")
            f.write(f"{self.survey_report}\n\n")

            f.write(f"{'='*60}\nGOAL BRIEF (A0)\n{'='*60}\n\n")
            f.write(f"{self.goal_brief}\n\n")

            for aid in ["A1", "A2", "A3", "A4", "A5", "S1", "S0", "F0",
                         "R1", "R2", "RS"]:
                agent = self.agents[aid]
                f.write(f"\n{'='*60}\n")
                f.write(f"AGENT {aid} ({agent.agent_type}) — {agent.call_count} calls\n")
                f.write(f"{'='*60}\n\n{agent.last_output}\n\n")

            if self.cycle_results:
                last = self.cycle_results[-1]
                f.write(f"\n{'='*60}\nFINAL OUTPUTS\n{'='*60}\n\n")
                for key in ["S0", "F0", "RS"]:
                    if key in last:
                        f.write(f"--- {key} ---\n{last[key]}\n\n")

        self.checkpoint.clear()

        print(f"\n{'='*60}")
        print(f"  ✓ ERGODIC COMPLETE")
        print(f"  Duration: {duration:.0f}s | LLM calls: {total_calls}")
        print(f"  Results: {results_path}")
        print(f"  Synthesis: {final_path}")
        print(f"{'='*60}\n")

        return all_results