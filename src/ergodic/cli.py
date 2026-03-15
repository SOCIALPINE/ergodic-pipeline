"""
ERGODIC CLI — Command-line interface for the ERGODIC pipeline.

Usage:
    ergodic run --goal "Your research goal" [options]
    ergodic run --config config.yaml
"""

from __future__ import annotations

import os
import sys

import click
import yaml

from .pipeline import ErgodicConfig, ErgodicPipeline


@click.group()
@click.version_option(version="0.9.0", prog_name="ergodic")
def main():
    """ERGODIC — Multi-agent research ideation pipeline."""
    pass


@main.command()
@click.option("--goal", "-g", type=str, default=None, help="Research goal statement.")
@click.option("--config", "-c", type=click.Path(exists=True), default=None,
              help="YAML config file.")
@click.option("--api-key", "-k", type=str, default=None,
              help="Google API key (or set GOOGLE_API_KEY env var).")
@click.option("--model", "-m", type=str, default=None,
              help="LLM model name (default: gemini-2.5-flash-lite).")
@click.option("--cycles", "-n", type=int, default=None,
              help="Number of cycles (default: 2).")
@click.option("--seed", "-s", type=int, default=None,
              help="Random seed for reproducibility.")
@click.option("--output", "-o", type=str, default=None,
              help="Output directory (default: ./ergodic_output).")
@click.option("--delay", "-d", type=int, default=None,
              help="Delay between API calls in seconds (default: 20).")
@click.option("--no-search", is_flag=True, default=False,
              help="Disable L0 information search.")
@click.option("--no-resume", is_flag=True, default=False,
              help="Start fresh (ignore checkpoint).")
def run(goal, config, api_key, model, cycles, seed, output, delay, no_search, no_resume):
    """Run the ERGODIC pipeline to generate research ideas."""
    cfg = ErgodicConfig()

    # Load YAML config if provided
    if config:
        with open(config, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        for key, val in yaml_cfg.items():
            attr = key.upper()
            if hasattr(cfg, attr):
                setattr(cfg, attr, val)
            elif hasattr(cfg, key):
                setattr(cfg, key, val)

    # CLI overrides
    if goal:
        cfg.GOAL = goal
    if api_key:
        cfg.GOOGLE_API_KEY = api_key
    if model:
        cfg.MODEL_NAME = model
    if cycles is not None:
        cfg.NUM_CYCLES = cycles
    if seed is not None:
        cfg.NOISE_SEED = seed
    if output:
        cfg.OUTPUT_DIR = output
    if delay is not None:
        cfg.DELAY_SECONDS = delay
    if no_search:
        cfg.INFORMATION_SEARCH = False

    # Resolve API key from environment
    if not cfg.GOOGLE_API_KEY:
        cfg.GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

    # Validate
    if not cfg.GOAL:
        click.echo("Error: --goal is required (or set 'goal' in config YAML).", err=True)
        sys.exit(1)
    if not cfg.GOOGLE_API_KEY:
        click.echo(
            "Error: Google API key required.\n"
            "  Set GOOGLE_API_KEY environment variable, or use --api-key.",
            err=True,
        )
        sys.exit(1)

    # Run
    try:
        pipeline = ErgodicPipeline(cfg)
        pipeline.run(resume=not no_resume)
    except KeyboardInterrupt:
        click.echo("\n  ⚠ Interrupted. Progress saved to checkpoint.", err=True)
        sys.exit(130)
    except Exception as exc:
        click.echo(f"\nError: {exc}", err=True)
        sys.exit(1)


@main.command()
@click.argument("output_dir", type=click.Path(exists=True), default="./ergodic_output")
def show(output_dir):
    """Show the final synthesis from a completed run."""
    import json

    results_path = os.path.join(output_dir, "ergodic_results.json")
    if not os.path.exists(results_path):
        click.echo(f"No results found in {output_dir}", err=True)
        sys.exit(1)

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    click.echo(f"\n{'='*60}")
    click.echo(f"  Goal: {data['config']['goal'][:70]}...")
    click.echo(f"  Version: {data['config'].get('version', '?')}")
    click.echo(f"  Duration: {data.get('duration_seconds', 0):.0f}s")
    click.echo(f"  LLM calls: {data.get('total_llm_calls', '?')}")
    click.echo(f"{'='*60}\n")

    if data.get("cycles"):
        last = data["cycles"][-1]["results"]
        if "S0" in last:
            click.echo("── FINAL SYNTHESIS (S0) ──\n")
            click.echo(last["S0"])
        if "RS" in last:
            click.echo("\n── REVIEW SUMMARY (RS) ──\n")
            click.echo(last["RS"])


@main.command()
@click.argument("output_dir", type=click.Path(exists=True), default="./ergodic_output")
def clear(output_dir):
    """Clear checkpoint to start fresh on next run."""
    ckpt = os.path.join(output_dir, "checkpoint.json")
    if os.path.exists(ckpt):
        os.remove(ckpt)
        click.echo(f"Checkpoint cleared: {ckpt}")
    else:
        click.echo("No checkpoint found.")


@main.command(name="init-config")
@click.option("--output", "-o", type=str, default="ergodic.yaml",
              help="Output YAML filename.")
def init_config(output):
    """Generate a sample config YAML file."""
    sample = {
        "goal": "Design a novel porous material for selective CO2 capture",
        "model_name": "gemini-2.5-flash-lite",
        "num_cycles": 2,
        "noise_seed": 42,
        "delay_seconds": 20,
        "max_results": 25,
        "information_search": True,
        "output_dir": "./ergodic_output",
    }
    with open(output, "w", encoding="utf-8") as f:
        yaml.dump(sample, f, default_flow_style=False, allow_unicode=True)
    click.echo(f"Sample config written to {output}")
    click.echo("Edit the file, then run: ergodic run --config ergodic.yaml")


if __name__ == "__main__":
    main()