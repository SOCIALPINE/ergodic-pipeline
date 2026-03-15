# ERGODIC

**Emergent Recursive Generation Over Distributed Interpretation Cycles**

A multi-agent AI pipeline that transforms random noise into emergent research ideas through multi-source information gathering, recursive critique, and synthesis.

## How It Works

ERGODIC runs a structured pipeline of 12 specialized AI agents across multiple cycles:

```
L0 (Information Scout) → A0 (Goal Analyst)
  → A1 (Ideation) → A2, A3 (Parallel Critique with separate noise seeds)
  → A4, A5 (Meta-Critique) → S1 (Summary) → S0 (Synthesis)
  → F0 (Formalization) → R1, R2 (Dual Review) → RS (Review Summary)
  → [Cycle 2: Revision based on review feedback]
```

**What makes it different:**
- **Multi-source L0**: Searches OpenAlex, arXiv, CrossRef, and Wikipedia simultaneously with LLM-guided source routing
- **Noise-driven divergence**: Random noise seeds creative exploration while domain gaps guide direction
- **Separate noise per agent**: A1, A2, A3 each get unique noise seeds for maximum idea diversity
- **Dual-axis review**: Novelty (1-10) and Feasibility (1-10) scored independently
- **Checkpoint & resume**: Survives interruptions — picks up from the last completed step

## Installation

```bash
pip install git+https://github.com/SOCIALPINE/ergodic-pipeline.git
```

## Quick Start

### 1. Set your API key

```bash
# Mac/Linux
export GOOGLE_API_KEY="your-google-api-key"

# Windows
set GOOGLE_API_KEY=your-google-api-key
```

### 2. Run

```bash
# Simple
ergodic run --goal "Design a novel porous material for selective CO2 capture"

# With options
ergodic run \
  --goal "Propose a tabletop experiment to detect dark matter" \
  --cycles 3 \
  --seed 42 \
  --output ./my_results

# From config file
ergodic init-config          # generates ergodic.yaml
ergodic run --config ergodic.yaml
```

### 3. View results

```bash
ergodic show ./ergodic_output
```

Results are saved as:
- `ergodic_results.json` — full pipeline data (all agent outputs, config, timing)
- `final_synthesis.txt` — human-readable summary

## Python API

```python
from ergodic import ErgodicConfig, ErgodicPipeline

config = ErgodicConfig()
config.GOOGLE_API_KEY = "your-key"
config.GOAL = "Design a novel porous material for selective CO2 capture"
config.NUM_CYCLES = 2
config.NOISE_SEED = 42

pipeline = ErgodicPipeline(config)
results = pipeline.run()

# Access final outputs
last_cycle = results["cycles"][-1]["results"]
print(last_cycle["S0"])   # Final synthesis
print(last_cycle["RS"])   # Review summary
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `ergodic run --goal "..."` | Run the pipeline |
| `ergodic run --config file.yaml` | Run from YAML config |
| `ergodic show [dir]` | Display results from a completed run |
| `ergodic clear [dir]` | Clear checkpoint to start fresh |
| `ergodic init-config` | Generate a sample YAML config |

### Run Options

| Option | Default | Description |
|--------|---------|-------------|
| `--goal, -g` | (required) | Research goal statement |
| `--api-key, -k` | `$GOOGLE_API_KEY` | Google API key |
| `--model, -m` | `gemini-2.5-flash-lite` | LLM model name |
| `--cycles, -n` | `2` | Number of critique-revision cycles |
| `--seed, -s` | random | Noise seed for reproducibility |
| `--output, -o` | `./ergodic_output` | Output directory |
| `--delay, -d` | `20` | Seconds between API calls |
| `--no-search` | off | Disable L0 information search |
| `--no-resume` | off | Ignore checkpoint, start fresh |

## Configuration

All options can be set via YAML config:

```yaml
goal: "Design a novel approach to federated learning privacy"
model_name: gemini-2.5-flash-lite
num_cycles: 2
noise_seed: 42
delay_seconds: 20
max_results: 25
information_search: true
output_dir: ./ergodic_output
```

## Pipeline Architecture

```
Cycle 1 (Exploration):
  L0 → A0 → A1(noise) → A2(noise₂) → A3(noise₃) → A4 → A5
       → S1 → S0 → F0 → R1 → R2 → RS

Cycle 2+ (Revision):
  A1(revision) → A2(critique) → A3(critique) → A4 → A5
       → S1 → S0 → F0 → R1 → R2 → RS
```

**11 LLM calls per cycle + L0 queries + A0 = ~24 total calls for 2 cycles**

### Agent Roles

| Agent | Role | Temperature |
|-------|------|-------------|
| L0 | Multi-source information gathering | — |
| A0 | Goal analysis → Goal Brief | 0.3 |
| A1-A5 | Idea generation & critique | 0.9 |
| S1 | Summary of layer 2 | 0.2 |
| S0 | Full synthesis | 0.3 |
| F0 | Formalization | 0.2 |
| R1, R2 | Dual review (Novelty + Feasibility) | 0.3 |
| RS | Review summary & prioritized fixes | 0.1 |

## Cross-Domain Validation

Tested across 5 domains with the same noise seed:

| Domain | Novelty | Feasibility | Key Finding |
|--------|---------|-------------|-------------|
| CO₂ Capture Materials | 9/10 | 6/10 | Framework + amino domains |
| Federated Learning Privacy | 9/10 | 5/10 | PQC + causal anomaly detection |
| Macroeconomics (Stagflation) | 8.5/10 | 6.5/10 | Supply-chain network + heterogeneous expectations |
| Dark Matter Detection | 9/10 | 4/10 | Topological quantum metamaterials |
| Urban Planning (15-min City) | 9/10 | 8/10 | Adaptive commons + temporal-fluid spaces |

## Requirements

- Python 3.10+
- Google API key (Gemini)
- Internet access (for L0 source queries)

## References

This project draws on ideas from:

- Du et al., "Improving Factuality and Reasoning in Language Models through Multiagent Debate" (ICML 2024) — multi-agent debate improves reasoning over single LLM inference
- Liang et al., "Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate" (EMNLP 2024) — structured agent debate encourages divergent thinking
- Li et al., "A Survey on LLM Hallucination via a Creativity Perspective" (2024) — hallucinations as creative catalysts when paired with convergent evaluation
- Kumar et al., "Human Creativity in the Age of LLMs" (CHI 2025) — LLM usage leads to idea homogenization, motivating external noise injection for diversity

## License

MIT
