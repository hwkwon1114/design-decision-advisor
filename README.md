# Design Decision Advisor

A Claude Code skill for making empirical decisions in computational design research through automated code experiments.

## What it does

When you are at a decision point in your code — comparing design representations, optimization algorithms, or objectives — this skill collaboratively designs and runs experiments to tell you which option is better, with statistical rigor and visualizations.

**Stack**: JAX, PyTorch, BoTorch, NumPy, SciPy

## When it triggers

- "Which representation has better expressivity?"
- "Compare EI vs UCB for this problem"
- "Should I use bilinear or bicubic for my design space?"
- "Which optimizer converges faster?"
- Any question about design space coverage, gradient quality, or surrogate fit

## What you get

- Automatically written and executed experiment scripts
- Convergence curves with confidence intervals
- Coverage metrics (discrepancy, spread, effective dimensionality)
- Statistical comparisons (Mann-Whitney U, Sobol sensitivity, cross-validation)
- Visualizations saved as PNG + PDF
- A clear recommendation with evidence

## Installation

```bash
# From GitHub
/plugin install https://github.com/<your-username>/design-decision-advisor

# Or from a local directory (for development)
claude --plugin-dir ./design-decision-advisor
```

## Bundled utilities

| Script | Purpose |
|--------|---------|
| `scripts/coverage_metrics.py` | Discrepancy, DTW diversity, gradient norm stats |
| `scripts/convergence_analysis.py` | Convergence curves, hypervolume, GP cross-validation |
| `scripts/statistical_tests.py` | Mann-Whitney, Sobol sensitivity, effect sizes |
| `scripts/visualization.py` | All standard plot templates |
