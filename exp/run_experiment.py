"""Full experiment runner for Avellaneda-Stoikov analytical validation."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from src.formulas import (
    frozen_inventory_value, reservation_ask, reservation_bid,
    reservation_price, half_spread, total_spread, optimal_quotes
)
from src.symbolic_derivation import (
    derive_frozen_value_function, derive_reservation_bid_ask,
    derive_liquidity_term, verify_reservation_price_formula,
    verify_half_spread_formula
)

# Paper parameters
S0 = 100.0
T = 1.0
SIGMA = 2.0
A = 140.0
K = 1.5
GAMMAS = [0.01, 0.1, 0.5]
Q_RANGE = list(range(-10, 11))
T_GRID = np.linspace(0, 0.99, 50)  # avoid t=T


def run_symbolic_verifications() -> dict:
    """Run all SymPy symbolic derivations and return verification results."""
    results = {}
    
    # 1. Frozen value function
    v, syms = derive_frozen_value_function()
    results['frozen_value_function'] = str(v)
    
    # 2. Reservation bid/ask
    sol_bid, sol_ask = derive_reservation_bid_ask()
    results['reservation_bid_solutions'] = [str(s) for s in sol_bid]
    results['reservation_ask_solutions'] = [str(s) for s in sol_ask]
    
    # 3. Liquidity term
    liq, ratio = derive_liquidity_term()
    results['liquidity_term'] = str(liq)
    results['lambda_ratio'] = str(ratio)
    
    # 4. Reservation price identity check
    r_mid, r_formula, diff_check = verify_reservation_price_formula()
    results['reservation_price_diff'] = str(diff_check)  # should be '0'
    results['reservation_price_verified'] = str(diff_check) == '0'
    
    # 5. Half-spread formula
    spread_inv, h = verify_half_spread_formula()
    results['inventory_spread'] = str(spread_inv)
    results['half_spread_formula'] = str(h)
    
    return results


def run_numerical_validation() -> pd.DataFrame:
    """Validate formulas numerically on grid of (q, t) for multiple gamma values."""
    records = []
    t_samples = [0.0, 0.25, 0.5, 0.75, 0.99]
    for gamma in GAMMAS:
        for q in Q_RANGE:
            for t in t_samples:
                tau = T - t
                r = reservation_price(S0, q, t, T, gamma, SIGMA)
                r_a = reservation_ask(S0, q, t, T, gamma, SIGMA)
                r_b = reservation_bid(S0, q, t, T, gamma, SIGMA)
                h = half_spread(t, T, gamma, SIGMA, K)
                spr = total_spread(t, T, gamma, SIGMA, K)
                p_a, p_b = optimal_quotes(S0, q, t, T, gamma, SIGMA, K)
                records.append({
                    'gamma': gamma, 'q': q, 't': round(t, 4), 'tau': round(tau, 4),
                    'r': r, 'r_ask': r_a, 'r_bid': r_b,
                    'half_spread': h, 'total_spread': spr,
                    'p_ask': p_a, 'p_bid': p_b,
                    'r_converges_to_S': abs(r - S0) < 0.01 if t > 0.98 else None
                })
    return pd.DataFrame(records)


def check_monotonicity(df: pd.DataFrame) -> dict:
    """Check that reservation price is decreasing in q (for fixed s,t,gamma)."""
    checks = {}
    for gamma in GAMMAS:
        for t in [0.0, 0.5]:
            subset = df[(df['gamma'] == gamma) & (df['t'] == round(t, 4))].sort_values('q')
            diffs = np.diff(subset['r'].values)
            checks[f'r_decreasing_in_q_gamma={gamma}_t={t}'] = bool(np.all(diffs <= 0))
    return checks


def check_maturity_convergence(df: pd.DataFrame) -> dict:
    """Check that reservation price -> S0 as t -> T."""
    checks = {}
    for gamma in GAMMAS:
        subset = df[(df['gamma'] == gamma) & (df['t'] == 0.99)]
        max_dev = (subset['r'] - S0).abs().max()
        checks[f'r_near_S0_at_maturity_gamma={gamma}'] = float(max_dev)
    return checks


def check_gamma_zero_limit() -> dict:
    """Check limiting behavior as gamma -> 0."""
    gammas_small = [0.1, 0.01, 0.001, 0.0001]
    results = {}
    t = 0.5
    q = 5
    for g in gammas_small:
        r = reservation_price(S0, q, t, T, g, SIGMA)
        h = half_spread(t, T, g, SIGMA, K)
        # As gamma->0: inventory adjustment q*gamma*sigma^2*tau -> 0
        # liquidity term (1/gamma)*ln(1+gamma/k) -> 1/k as gamma->0 (L'Hopital)
        results[f'gamma={g}'] = {
            'reservation_adj': float(S0 - r),  # q*gamma*sigma^2*tau -> 0
            'half_spread': float(h),
            'h_minus_1_over_k': float(h - 1/K)  # should -> 0
        }
    return results


def run_all_experiments() -> dict:
    """Run all experiments and return combined results."""
    print('Running symbolic verifications...')
    symbolic = run_symbolic_verifications()
    
    print('Running numerical validation...')
    df = run_numerical_validation()
    
    print('Checking monotonicity...')
    mono = check_monotonicity(df)
    
    print('Checking maturity convergence...')
    conv = check_maturity_convergence(df)
    
    print('Checking gamma->0 limit...')
    gamma_lim = check_gamma_zero_limit()
    
    return {
        'symbolic': symbolic,
        'numerical_df': df,
        'monotonicity': mono,
        'maturity_convergence': conv,
        'gamma_zero_limit': gamma_lim
    }


if __name__ == '__main__':
    results = run_all_experiments()
    
    # Save numerical table
    os.makedirs('/workspace/project/results', exist_ok=True)
    results['numerical_df'].to_csv('/workspace/project/results/numerical_validation.csv', index=False)
    print('Saved numerical_validation.csv')
    
    # Print summaries
    print('\n--- Symbolic Verification ---')
    for k, v in results['symbolic'].items():
        print(f'  {k}: {v}')
    
    print('\n--- Monotonicity Checks ---')
    for k, v in results['monotonicity'].items():
        print(f'  {k}: {v}')
    
    print('\n--- Maturity Convergence (max |r-S0| at t=0.99) ---')
    for k, v in results['maturity_convergence'].items():
        print(f'  {k}: {v:.6f}')
    
    print('\n--- Gamma->0 Limit ---')
    for k, v in results['gamma_zero_limit'].items():
        print(f'  {k}: {v}')
    
    print('\nDone.')
