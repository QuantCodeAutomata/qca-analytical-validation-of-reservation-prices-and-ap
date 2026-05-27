# Analytical Validation of Reservation Prices and Approximate Closed-Form Inventory-Based Quoting Strategy

Analytical validation and reproduction of the Avellaneda-Stoikov (2008) market-making model closed-form formulas.

## Overview

This project implements, derives, and validates the key analytical results from the Avellaneda-Stoikov model:

- **Frozen-inventory CARA utility value function** via Gaussian MGF
- **Reservation bid/ask prices** from utility indifference conditions
- **Reservation/indifference price** as inventory-adjusted mid-price
- **Optimal half-spread** combining inventory risk and liquidity costs
- **Two-step quoting rule**: center on reservation price, place quotes at ±h

## Repository Structure

```
├── src/
│   ├── formulas.py            # Pure NumPy formula implementations
│   └── symbolic_derivation.py # SymPy symbolic derivations & verification
├── exp/
│   ├── run_experiment.py      # Full experiment runner
│   └── plots.py               # Matplotlib visualizations
├── tests/
│   └── test_formulas.py       # 37 pytest tests
├── results/
│   ├── RESULTS.md             # Experiment findings and metrics
│   ├── numerical_validation.csv
│   └── *.png                  # Generated plots
├── data/                      # (No empirical data needed — analytical experiment)
└── AGENTS.md                  # Repository knowledge base
```

## Key Formulas

| Formula | Expression |
|---------|-----------|
| Frozen utility | `v = −exp(−γx)·exp(−γqs)·exp(½γ²q²σ²τ)` |
| Reservation ask | `r^a = s + ((1−2q)/2)·γσ²τ` |
| Reservation bid | `r^b = s + ((−1−2q)/2)·γσ²τ` |
| Reservation price | `r = s − q·γσ²τ` |
| Half-spread | `h = ½γσ²τ + (1/γ)·ln(1+γ/k)` |
| Optimal quotes | `p^a = r+h`, `p^b = r−h` |

where `τ = T − t` is time-to-maturity.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/test_formulas.py -v

# Run full experiment
python exp/run_experiment.py

# Generate plots
python exp/plots.py
```

## Parameters (paper defaults)

| Parameter | Value | Description |
|-----------|-------|-------------|
| S₀ | 100 | Initial mid-price |
| T | 1 | Time horizon |
| σ | 2 | Mid-price volatility |
| A | 140 | Order arrival intensity constant |
| k | 1.5 | Order book depth parameter |
| γ | {0.01, 0.1, 0.5} | Risk-aversion coefficients |
| q | −10,...,10 | Inventory grid |

## Results Summary

- All 37 tests pass
- Symbolic identity `(r^a+r^b)/2 − r = 0` verified by SymPy
- Reservation price strictly decreasing in q ✓
- Convergence to mid-price at maturity ✓
- Liquidity term converges to 1/k as γ→0 ✓

See [results/RESULTS.md](results/RESULTS.md) for full metrics.

## Reference

Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book*. Quantitative Finance, 8(3), 217–224.
