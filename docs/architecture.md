# ERGODIC Architecture

## Pipeline Topology

```
                    ┌─────────────────────────────────────────┐
                    │              ERGODIC Pipeline            │
                    └─────────────────────────────────────────┘

Phase 0: Information Gathering
  ┌────┐    ┌────┐
  │ L0 │───→│ A0 │──→ Goal Brief
  └────┘    └────┘
  Multi-API   Goal
  Search      Analyst

Phase 1: Forward Pass (Ideation + Critique)
  ┌────┐    ┌────┐    ┌────┐
  │ A1 │───→│ A2 │───→│ A4 │
  └────┘  ╲ └────┘    └────┘
   Seed    ╲
            ╲┌────┐    ┌────┐
             │ A3 │───→│ A5 │
             └────┘    └────┘

  A1: Initial idea from noise + gaps
  A2: Critique A1 with own noise (seed+1000)
  A3: Critique A1 with own noise (seed+2000)
  A4: Meta-critique of A2+A3
  A5: Meta-critique of A2+A3

Phase 2: Backward Pass (Compression)
  ┌────┐    ┌────┐
  │ S1 │───→│ S0 │──→ Unified Proposal
  └────┘    └────┘

Phase 3: Formalization
  ┌────┐
  │ F0 │──→ Formal Specification
  └────┘

Phase 4: Review
  ┌────┐    ┌────┐
  │ R1 │───→│ RS │──→ Verdict + Fixes
  │ R2 │───→│    │
  └────┘    └────┘
```

## L0: Information Scout

L0 searches 4 sources simultaneously:

| Source | Type | What it finds |
|--------|------|---------------|
| OpenAlex | Academic | Papers from all fields, citation counts |
| arXiv | Preprints | Latest preprints (CS, physics, math, bio) |
| CrossRef | Metadata | DOIs, citation data, venue info |
| Wikipedia | Background | Concept definitions, context |

**Adaptive features:**
- LLM-guided source weight routing based on goal domain
- LLM-generated search queries (8 diverse queries)
- Supplementary material DOI filtering
- Borderline relevance LLM judging
- Adaptive 2nd-round search when results are sparse
- Minimum per-source guarantees (OpenAlex 8, CrossRef 4, arXiv 3, Wikipedia 2)

## Agent Design

Each agent has:
- **Semantic Memory**: Tracks core ideas, decisions, cycle history
- **Retry logic**: 3 attempts with exponential backoff
- **System prompt**: Role-specific instructions with output format

## Checkpoint System

After every step, the full pipeline state is saved to `checkpoint.json`:
- All agent memories and outputs
- Current cycle and step position
- Survey report and goal brief
- All intermediate results

On resume, the pipeline restores state and continues from the last completed step.

## Dual-Axis Review

R1 and R2 score independently on:
- **Novelty (1-10)**: Does this exist in cited sources?
- **Feasibility (1-10)**: Can this be implemented with current technology?

RS merges reviews into prioritized fixes for the next cycle.