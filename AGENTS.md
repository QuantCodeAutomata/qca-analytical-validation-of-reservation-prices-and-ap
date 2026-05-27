# Project: Avellaneda-Stoikov Market-Making Model

## Structure
- `src/formulas.py` — Pure NumPy numerical implementations of all A-S closed-form formulas
- `src/symbolic_derivation.py` — SymPy symbolic derivations and algebraic verifications
- `exp/run_experiment.py` — Full experiment runner (numerical validation + symbolic checks)
- `exp/plots.py` — Matplotlib visualisation of model properties
- `tests/test_formulas.py` — 37-test pytest suite covering all formula classes
- `results/` — Output directory for CSV tables and PNG plots

## Key Formulas (paper: Avellaneda & Stoikov 2008)
- **Frozen value**: `v(x,s,q,t) = -exp(-γx)·exp(-γqs)·exp(½γ²q²σ²τ)`, τ = T−t
- **Reservation ask**: `r^a = s + ((1−2q)/2)·γσ²τ`
- **Reservation bid**: `r^b = s + ((−1−2q)/2)·γσ²τ`
- **Reservation price**: `r = s − qγσ²τ` (midpoint of r^a and r^b)
- **Half-spread**: `h = ½γσ²τ + (1/γ)·ln(1+γ/k)`
- **Optimal quotes**: `p^a = r + h`, `p^b = r − h`

## Symbolic derivation functions (src/symbolic_derivation.py)
- `derive_frozen_value_function()` → `(v_expr, (x,s,q,t,T,γ,σ))`
- `derive_reservation_bid_ask()` → `(sol_bid, sol_ask)` — SymPy solve results (non-empty lists)
- `derive_liquidity_term()` → `(liq_expr, ratio)` where `ratio = -1/k` (λ/λ')
- `verify_reservation_price_formula()` → `(r_mid, r_formula, diff_check)`, `diff_check == 0`
- `verify_half_spread_formula()` → `(spread_inv, h)` — inventory spread and half-spread formula

## Running Tests
```bash
python3 -m pytest tests/test_formulas.py -v
```
All 37 tests pass in ~1.8s (Python 3.12, pytest 9.0).

## Paper Parameters Used in Tests
`S0=100, T=1.0, σ=2.0, A=140.0, k=1.5, γ_default=0.1`
