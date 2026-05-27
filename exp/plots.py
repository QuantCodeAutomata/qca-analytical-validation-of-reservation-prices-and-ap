"""Visualization module for Avellaneda-Stoikov analytical validation."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from src.formulas import (
    reservation_price, reservation_ask, reservation_bid,
    half_spread, total_spread, optimal_quotes
)

# Paper parameters
S0, T, SIGMA, A, K = 100.0, 1.0, 2.0, 140.0, 1.5
GAMMAS = [0.01, 0.1, 0.5]
Q_RANGE = list(range(-10, 11))
RESULTS_DIR = '/workspace/project/results'

os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_reservation_price_vs_inventory():
    """Plot reservation price vs inventory for multiple gamma and time values."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Reservation Price vs Inventory\n(s=100, T=1, sigma=2)', fontsize=14)

    t_values = [0.0, 0.5, 0.75, 0.99]
    colors = plt.cm.viridis(np.linspace(0, 1, len(t_values)))

    for ax, gamma in zip(axes, GAMMAS):
        for t, color in zip(t_values, colors):
            prices = [reservation_price(S0, q, t, T, gamma, SIGMA) for q in Q_RANGE]
            ax.plot(Q_RANGE, prices, color=color, label=f't={t}', linewidth=2)
        ax.axhline(y=S0, color='black', linestyle='--', alpha=0.5, label='Mid S=100')
        ax.set_xlabel('Inventory q', fontsize=11)
        ax.set_ylabel('Reservation Price r(s,q,t)', fontsize=11)
        ax.set_title(f'gamma={gamma}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'reservation_price_vs_inventory.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_optimal_quotes_vs_inventory():
    """Plot optimal ask/bid quotes and reservation price vs inventory."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Optimal Quotes vs Inventory at t=0.5\n(s=100, T=1, sigma=2, k=1.5)', fontsize=14)

    t = 0.5
    for ax, gamma in zip(axes, GAMMAS):
        asks = [optimal_quotes(S0, q, t, T, gamma, SIGMA, K)[0] for q in Q_RANGE]
        bids = [optimal_quotes(S0, q, t, T, gamma, SIGMA, K)[1] for q in Q_RANGE]
        reservations = [reservation_price(S0, q, t, T, gamma, SIGMA) for q in Q_RANGE]

        ax.plot(Q_RANGE, asks, 'r-o', markersize=3, label='Ask p^a', linewidth=2)
        ax.plot(Q_RANGE, bids, 'b-o', markersize=3, label='Bid p^b', linewidth=2)
        ax.plot(Q_RANGE, reservations, 'g--', label='Reservation r', linewidth=2)
        ax.axhline(y=S0, color='black', linestyle=':', alpha=0.5, label='Mid S=100')
        ax.fill_between(Q_RANGE, bids, asks, alpha=0.15, color='gray', label='Spread')
        ax.set_xlabel('Inventory q', fontsize=11)
        ax.set_ylabel('Price', fontsize=11)
        ax.set_title(f'gamma={gamma}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'optimal_quotes_vs_inventory.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_half_spread_vs_time():
    """Plot half-spread as function of time for multiple gamma values."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Half-Spread Components vs Time\n(sigma=2, k=1.5, T=1)', fontsize=14)

    t_grid = np.linspace(0, 0.99, 200)

    # Left: total half-spread
    ax = axes[0]
    for gamma in GAMMAS:
        hs = [half_spread(t, T, gamma, SIGMA, K) for t in t_grid]
        ax.plot(t_grid, hs, label=f'gamma={gamma}', linewidth=2)
    ax.set_xlabel('Time t', fontsize=11)
    ax.set_ylabel('Half-Spread h_t', fontsize=11)
    ax.set_title('Total Half-Spread vs Time', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: decomposition for gamma=0.1
    ax = axes[1]
    gamma = 0.1
    hs = np.array([half_spread(t, T, gamma, SIGMA, K) for t in t_grid])
    inv_comp = np.array([0.5 * gamma * SIGMA**2 * (T - t) for t in t_grid])
    liq_comp = np.full_like(t_grid, (1.0/gamma) * np.log(1 + gamma/K))

    ax.plot(t_grid, hs, 'k-', label='Total h_t', linewidth=2)
    ax.plot(t_grid, inv_comp, 'r--', label='Inventory: 0.5*gamma*sigma^2*tau', linewidth=2)
    ax.plot(t_grid, liq_comp, 'b--', label='Liquidity: (1/gamma)*ln(1+gamma/k)', linewidth=2)
    ax.set_xlabel('Time t', fontsize=11)
    ax.set_ylabel('Half-Spread Component', fontsize=11)
    ax.set_title(f'Decomposition (gamma={gamma})', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'half_spread_vs_time.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_gamma_zero_limit():
    """Plot convergence behavior as gamma -> 0."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Gamma -> 0 Limit Behavior\n(s=100, T=1, sigma=2, k=1.5)', fontsize=14)

    gammas_log = np.logspace(-4, 0, 100)
    t = 0.5
    q = 5

    # Left: inventory adjustment |r - S0| vs gamma
    ax = axes[0]
    inv_adj = [abs(reservation_price(S0, q, t, T, g, SIGMA) - S0) for g in gammas_log]
    ax.loglog(gammas_log, inv_adj, 'b-', linewidth=2, label='|r - S0|')
    ax.loglog(gammas_log, q * SIGMA**2 * (T-t) * gammas_log, 'r--',
              linewidth=1.5, label='q*sigma^2*tau*gamma (linear)')
    ax.set_xlabel('gamma', fontsize=11)
    ax.set_ylabel('Inventory Adjustment |r - s|', fontsize=11)
    ax.set_title('Inventory Effect vs gamma (q=5, t=0.5)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: liquidity term vs gamma
    ax = axes[1]
    liq_terms = [(1.0/g)*np.log(1+g/K) for g in gammas_log]
    ax.semilogx(gammas_log, liq_terms, 'g-', linewidth=2, label='(1/gamma)*ln(1+gamma/k)')
    ax.axhline(y=1.0/K, color='r', linestyle='--', linewidth=1.5, label=f'1/k = {1/K:.4f} (limit)')
    ax.set_xlabel('gamma', fontsize=11)
    ax.set_ylabel('Liquidity Term', fontsize=11)
    ax.set_title('Liquidity Term Convergence to 1/k', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'gamma_zero_limit.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_maturity_convergence():
    """Plot reservation price convergence to mid-price as t approaches T."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Maturity Convergence: r(s,q,t) -> s as t -> T', fontsize=14)

    t_grid = np.linspace(0, 0.999, 200)
    q_values = [-5, -2, 0, 2, 5]
    gamma = 0.1

    # Left: reservation price for different q values
    ax = axes[0]
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(q_values)))
    for q, color in zip(q_values, colors):
        prices = [reservation_price(S0, q, t, T, gamma, SIGMA) for t in t_grid]
        ax.plot(t_grid, prices, color=color, label=f'q={q}', linewidth=2)
    ax.axhline(y=S0, color='black', linestyle='--', alpha=0.7, linewidth=1.5, label='Mid S=100')
    ax.set_xlabel('Time t', fontsize=11)
    ax.set_ylabel('Reservation Price r(s,q,t)', fontsize=11)
    ax.set_title(f'gamma={gamma}: Convergence to Mid-Price', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: deviation from mid-price
    ax = axes[1]
    for q, color in zip(q_values, colors):
        devs = [reservation_price(S0, q, t, T, gamma, SIGMA) - S0 for t in t_grid]
        ax.plot(t_grid, devs, color=color, label=f'q={q}', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Time t', fontsize=11)
    ax.set_ylabel('r(s,q,t) - s', fontsize=11)
    ax.set_title(f'gamma={gamma}: Deviation from Mid-Price', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'maturity_convergence.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def run_all_plots():
    """Run all visualization routines."""
    print('Generating plots...')
    plot_reservation_price_vs_inventory()
    plot_optimal_quotes_vs_inventory()
    plot_half_spread_vs_time()
    plot_gamma_zero_limit()
    plot_maturity_convergence()
    print('All plots saved to results/')


if __name__ == '__main__':
    run_all_plots()
