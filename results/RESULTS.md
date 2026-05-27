# Experiment Results: Analytical Validation of Reservation Prices  
## Avellaneda-Stoikov Market-Making Model (2008)

---

## 1. Symbolic Verification (SymPy)

All closed-form formulas were derived and verified algebraically using SymPy.

### 1.1 Frozen-Inventory Value Function
```
v(x, s, q, t) = -exp(-γx) · exp(-γqs) · exp(½γ²q²σ²(T−t))
```
- **Derivation method**: Gaussian MGF with θ=−γq, μ=qs, var=σ²τ  
- **SymPy result**: `-exp(-gamma*x)*exp(-gamma*q*s)*exp(gamma**2*q**2*sigma**2*(T-t)/2)` ✓

### 1.2 Reservation Bid/Ask Prices
Solved from utility indifference conditions `v(x±r, s, q∓1, t) = v(x, s, q, t)`:

| Formula | SymPy Solution |
|---------|---------------|
| r^b(s,q,t) | `s + ((-1-2q)/2)·γσ²(T-t)` |
| r^a(s,q,t) | `s + ((1-2q)/2)·γσ²(T-t)` |

### 1.3 Reservation Price Identity
```
(r^a + r^b)/2 = s − q·γσ²(T−t)
```
- **Difference check** (SymPy simplify): **0** ✓  
- `reservation_price_verified = True`

### 1.4 Liquidity Term Derivation
- λ(δ) = A·exp(−kδ), λ'(δ) = −k·λ(δ)  
- **Ratio**: λ/λ' = **−1/k** ✓  
- **Liquidity term**: `(1/γ)·ln(1 + γ/k)` = `log(γ/k + 1)/γ` ✓

### 1.5 Half-Spread Components
- **Inventory spread** (r^a − r^b): `γσ²(T−t)` ✓  
- **Full half-spread**: `h = γσ²(T−t)/2 + log(γ/k + 1)/γ` ✓

---

## 2. Numerical Validation

**Parameters**: S₀=100, T=1, σ=2, A=140, k=1.5, γ ∈ {0.01, 0.1, 0.5}, q ∈ {−10,...,10}, t ∈ {0.0, 0.25, 0.5, 0.75, 0.99}

### 2.1 Monotonicity of Reservation Price in Inventory q

| gamma | t=0.0 | t=0.5 |
|-------|-------|-------|
| 0.01  | ✓ decreasing | ✓ decreasing |
| 0.1   | ✓ decreasing | ✓ decreasing |
| 0.5   | ✓ decreasing | ✓ decreasing |

**Result**: r(s,q,t) is strictly decreasing in q for all tested parameter combinations ✓

### 2.2 Maturity Convergence: max|r − S₀| at t=0.99

| gamma | max|r(s,q,0.99) − S₀| | Expected (q_max·γσ²·0.01) |
|-------|---------------------|--------------------------|
| 0.01  | 0.004000            | 0.004 (10·0.01·4·0.01)   |
| 0.1   | 0.040000            | 0.04 (10·0.1·4·0.01)     |
| 0.5   | 0.200000            | 0.2 (10·0.5·4·0.01)      |

**Result**: As t→T, reservation price converges to mid-price S₀=100 ✓

### 2.3 Half-Spread at Selected Parameters (t=0.5, γ=0.1, σ=2, k=1.5)

| Component | Value |
|-----------|-------|
| Inventory component: 0.5·γ·σ²·τ | 0.1 |
| Liquidity component: (1/γ)·ln(1+γ/k) | 0.6454 |
| **Total half-spread h** | **0.7454** |
| Total spread 2h | 1.4908 |

---

## 3. Gamma→0 Limit Analysis

| γ | Inventory adj q·γσ²τ | Half-spread h | h − 1/k |
|---|---------------------|---------------|---------|
| 0.1   | 1.000000 | 0.74539 | 0.07872  |
| 0.01  | 0.100000 | 0.67445 | 0.00779  |
| 0.001 | 0.010000 | 0.66744 | 0.00078  |
| 0.0001| 0.001000 | 0.66674 | 0.000078 |
| → 0   | → 0      | → 1/k=0.6667 | → 0    |

**Key finding**: As γ→0:  
- Inventory adjustment q·γσ²τ → 0 (inventory effect vanishes) ✓  
- Liquidity term (1/γ)·ln(1+γ/k) → 1/k = 0.6667 (L'Hôpital) ✓  
- Strategy converges to symmetric mid-price-centered quoting ✓

---

## 4. Test Results

**37/37 tests passed** (pytest, Python 3.12)

| Test Class | Count | Result |
|------------|-------|--------|
| TestFrozenValueFunction | 5 | ✓ All passed |
| TestReservationPrices | 11 | ✓ All passed |
| TestHalfSpread | 6 | ✓ All passed |
| TestOptimalQuotes | 7 | ✓ All passed |
| TestSymbolicDerivation | 4 | ✓ All passed |
| TestNumericalProperties | 4 | ✓ All passed |

Notable tests validated:
- **Utility indifference (bid)**: v(x−r^b, s, q+1, t) = v(x, s, q, t) ✓ (error < 1e-8)
- **Utility indifference (ask)**: v(x+r^a, s, q−1, t) = v(x, s, q, t) ✓ (error < 1e-8)
- **Reservation price identity**: (r^a+r^b)/2 = r ✓ (SymPy: diff = 0)
- **Gamma→0 convergence**: liq. term within 0.001 of 1/k for γ=0.001 ✓

---

## 5. Plots Generated

| Plot | Description |
|------|-------------|
| `reservation_price_vs_inventory.png` | r(s,q,t) vs inventory q for γ∈{0.01,0.1,0.5} at t∈{0,0.25,0.5,0.99} |
| `optimal_quotes_vs_inventory.png` | Ask p^a, bid p^b, and reservation r vs q at t=0.5 |
| `half_spread_vs_time.png` | Half-spread h_t vs time; decomposition into inventory + liquidity components |
| `gamma_zero_limit.png` | Log-log: inventory adj vs γ; semilog: liquidity term converging to 1/k |
| `maturity_convergence.png` | r(s,q,t) paths converging to S₀=100 as t→T |

---

## 6. Conclusions

All research objectives were met:

1. ✅ **Frozen-inventory value function** derived from Gaussian MGF and verified symbolically
2. ✅ **Reservation bid/ask prices** solved via SymPy utility indifference, match paper formulas exactly
3. ✅ **Reservation price converges to mid-price** at maturity (verified analytically and numerically)
4. ✅ **Closed-form spread formula** h = ½γσ²τ + (1/γ)ln(1+γ/k) derived from FOC with λ=Ae^{-kδ}
5. ✅ **Two-step quoting rule** confirmed: p^a = r+h, p^b = r−h
6. ✅ **γ→0 limit**: inventory effect vanishes, spread approaches 2/k (symmetric mid-price quoting)
