# Examples

## Quick Start

```bash
# 1. Set API key
export GOOGLE_API_KEY="your-key"

# 2. Run with CLI
ergodic run --goal "Design a novel porous material for selective CO2 capture" --seed 42

# 3. Or use config file
ergodic run --config examples/config.yaml
```

## Example Goals

These goals have been validated across diverse domains:

```bash
# Materials Science
ergodic run -g "Design a novel porous material for selective CO2 capture" -s 42

# Computer Science
ergodic run -g "Propose a privacy-preserving federated learning framework for healthcare" -s 42

# Economics
ergodic run -g "Develop a macroeconomic model for stagflation with supply chain disruptions" -s 42

# Physics
ergodic run -g "Design a tabletop experiment to detect dark matter candidates" -s 42

# Urban Planning
ergodic run -g "Design an adaptive 15-minute city framework for high-density Asian megacities" -s 42
```

## Python API

```python
from ergodic import ErgodicConfig, ErgodicPipeline

config = ErgodicConfig()
config.GOOGLE_API_KEY = "your-key"
config.GOAL = "Design a novel porous material for selective CO2 capture"
config.NOISE_SEED = 42

pipeline = ErgodicPipeline(config)
results = pipeline.run()

# Final synthesis
print(results["cycles"][-1]["results"]["S0"])

# Review scores
print(results["cycles"][-1]["results"]["RS"])
```