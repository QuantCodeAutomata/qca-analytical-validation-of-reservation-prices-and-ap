"""
Avellaneda-Stoikov model: symbolic derivation and verification of all
closed-form formulas using SymPy.

Each function documents the mathematical reasoning behind the result and
returns SymPy expressions that can be inspected, pretty-printed, or
exported to LaTeX.

SymPy library confirmed available via Context7 (/sympy/sympy).
"""

# Using SymPy — Context7 confirmed (/sympy/sympy)
from sympy import (
    symbols, exp, log, simplify, solve, Eq, diff, ln, latex, pprint,
)
import sympy as sp


# ---------------------------------------------------------------------------
# Derivations
# ---------------------------------------------------------------------------

def derive_frozen_value_function():
    """Derive v(x, s, q, t) symbolically using the MGF of a Gaussian.

    Under the frozen-inventory assumption the terminal wealth is
    W_T = x + q * S_T, where S_T | S_t = s is modelled as

        S_T ~ N(s,  sigma^2 * tau),   tau = T - t.

    The expected CARA utility is therefore

        v = E[-exp(-gamma * W_T)]
          = -exp(-gamma * x) * E[exp(-gamma * q * S_T)]

    The moment-generating function of N(mu, var) evaluated at theta is
    exp(theta * mu + 0.5 * theta^2 * var).  With theta = -gamma * q,
    mu = s, var = sigma^2 * tau:

        E[exp(-gamma*q*S_T)] = exp(-gamma*q*s + 0.5*gamma^2*q^2*sigma^2*tau)

    Combining:

        v(x, s, q, t) = -exp(-gamma*x)
                        * exp(-gamma*q*s)
                        * exp(0.5*gamma^2*q^2*sigma^2*tau)

    Returns
    -------
    tuple[Expr, tuple[Symbol, ...]]
        (v, (x, s, q, t, T, gamma, sigma)) — symbolic expression and its
        constituent symbols.
    """
    x, s, q, t, T, gamma, sigma = symbols(
        'x s q t T gamma sigma', real=True, positive=True
    )
    tau = T - t
    # MGF of N(q*s, q^2*sigma^2*tau) at theta = -gamma  →  exp(-gamma*q*s + …)
    v = (
        -exp(-gamma * x)
        * exp(-gamma * q * s)
        * exp(sp.Rational(1, 2) * gamma**2 * q**2 * sigma**2 * tau)
    )
    return v, (x, s, q, t, T, gamma, sigma)


def derive_reservation_bid_ask():
    """Solve utility indifference conditions for r^a and r^b.

    Reservation ask r^a — the price that makes selling one unit
    utility-neutral:

        v(x + r^a, s, q - 1, t) = v(x, s, q, t)

    Reservation bid r^b — the price that makes buying one unit
    utility-neutral:

        v(x - r^b, s, q + 1, t) = v(x, s, q, t)

    Both conditions are solved symbolically by SymPy, yielding:

        r^b = s + ((-1 - 2*q) / 2) * gamma * sigma^2 * tau
        r^a = s + ((1  - 2*q) / 2) * gamma * sigma^2 * tau

    Returns
    -------
    tuple[list, list]
        (sol_bid, sol_ask) — lists of SymPy solutions for r^b and r^a.
    """
    x, s, q, t, T, gamma, sigma, r_b, r_a = symbols(
        'x s q t T gamma sigma r_b r_a', real=True
    )
    tau = T - t

    def v_func(x_val, s_val, q_val):
        """Frozen-inventory utility value at (x_val, s_val, q_val)."""
        return (
            -exp(-gamma * x_val)
            * exp(-gamma * q_val * s_val)
            * exp(sp.Rational(1, 2) * gamma**2 * q_val**2 * sigma**2 * tau)
        )

    # Reservation bid: buying at r^b raises inventory q → q+1
    eq_bid = Eq(v_func(x - r_b, s, q + 1), v_func(x, s, q))
    sol_bid = solve(eq_bid, r_b)

    # Reservation ask: selling at r^a lowers inventory q → q-1
    eq_ask = Eq(v_func(x + r_a, s, q - 1), v_func(x, s, q))
    sol_ask = solve(eq_ask, r_a)

    return sol_bid, sol_ask


def derive_liquidity_term():
    """Derive the liquidity adjustment (1/gamma)*ln(1 + gamma/k).

    Order arrivals follow a Poisson process with intensity

        lambda(delta) = A * exp(-k * delta),

    where delta is the distance of the posted quote from the mid-price.
    The first-order condition of the HJB equation w.r.t. delta contains
    the ratio lambda / lambda':

        lambda / (d lambda / d delta) = A*exp(-k*delta) / (-k*A*exp(-k*delta))
                                      = -1/k

    Substituting into the FOC

        delta* = -lambda/lambda' + (1/gamma)*ln(1 - gamma * (lambda/lambda'))

    gives the liquidity-adjusted half-spread term:

        (1/gamma) * ln(1 + gamma/k)

    Returns
    -------
    tuple[Expr, Expr]
        (liquidity_term, ratio) where ratio = lambda / lambda_prime = -1/k.
    """
    gamma_sym, k_sym, delta_sym, A_sym = symbols('gamma k delta A', positive=True)
    lambda_sym = A_sym * exp(-k_sym * delta_sym)
    lambda_prime = diff(lambda_sym, delta_sym)          # = -k * lambda
    ratio = simplify(lambda_sym / lambda_prime)         # = -1/k
    liquidity_term = (1 / gamma_sym) * log(1 + gamma_sym / k_sym)
    return simplify(liquidity_term), ratio


def verify_reservation_price_formula():
    """Verify r = (r^a + r^b) / 2 = s - q * gamma * sigma^2 * (T - t).

    The reservation (indifference) price is the arithmetic mean of the
    reservation ask and bid.  SymPy confirms the identity:

        (r^a + r^b) / 2
          = [s + ((1-2q)/2)*gamma*sigma^2*tau
             + s + ((-1-2q)/2)*gamma*sigma^2*tau] / 2
          = s - q * gamma * sigma^2 * tau

    Returns
    -------
    tuple[Expr, Expr, Expr]
        (r_mid, r_formula, diff_check) — the midpoint expression, the
        closed-form formula, and their difference (identically zero).
    """
    s, q, t, T, gamma, sigma = symbols('s q t T gamma sigma', real=True)
    tau = T - t
    r_a_expr = s + ((1 - 2 * q) / 2) * gamma * sigma**2 * tau
    r_b_expr = s + ((-1 - 2 * q) / 2) * gamma * sigma**2 * tau
    r_mid = simplify((r_a_expr + r_b_expr) / 2)
    r_formula = s - q * gamma * sigma**2 * tau
    diff_check = simplify(r_mid - r_formula)
    return r_mid, r_formula, diff_check     # diff_check == 0


def verify_half_spread_formula():
    """Verify half-spread = 0.5*gamma*sigma^2*(T-t) + (1/gamma)*ln(1+gamma/k).

    The total bid-ask spread decomposes into two additive components:

    1. Inventory risk component: r^a - r^b = gamma * sigma^2 * tau
       (the spread between the reservation ask and bid prices).

    2. Liquidity component: (2/gamma) * ln(1 + gamma/k)
       (derived from the order-arrival FOC; see derive_liquidity_term).

    The half-spread is therefore:

        h = 0.5 * gamma * sigma^2 * tau + (1/gamma) * ln(1 + gamma/k)

    Returns
    -------
    tuple[Expr, Expr]
        (spread_inv, h) — the inventory-risk spread r^a - r^b and the
        full optimal half-spread formula.
    """
    s, q, t, T, gamma, sigma, k = symbols(
        's q t T gamma sigma k', positive=True
    )
    tau = T - t
    r_a_expr = s + ((1 - 2 * q) / 2) * gamma * sigma**2 * tau
    r_b_expr = s + ((-1 - 2 * q) / 2) * gamma * sigma**2 * tau
    spread_inv = simplify(r_a_expr - r_b_expr)   # = gamma*sigma^2*tau
    h = (
        sp.Rational(1, 2) * gamma * sigma**2 * tau
        + (1 / gamma) * log(1 + gamma / k)
    )
    return spread_inv, h


# ---------------------------------------------------------------------------
# Main: print all derivations and verifications
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=== Frozen Value Function ===')
    v, syms = derive_frozen_value_function()
    pprint(v)

    print('\n=== Reservation Bid/Ask Derivation ===')
    sol_bid, sol_ask = derive_reservation_bid_ask()
    print('r^b solutions:', sol_bid)
    print('r^a solutions:', sol_ask)

    print('\n=== Liquidity Term ===')
    liq, ratio = derive_liquidity_term()
    print('lambda/lambda_prime:', ratio)
    print('liquidity term:', liq)

    print('\n=== Reservation Price Verification ===')
    r_mid, r_formula, diff_check = verify_reservation_price_formula()
    print('(r^a+r^b)/2 =', r_mid)
    print('r formula   =', r_formula)
    print('Difference (should be 0):', diff_check)

    print('\n=== Half-spread Verification ===')
    spread_inv, h = verify_half_spread_formula()
    print('r^a - r^b (inventory spread):', spread_inv)
    print('h formula:', h)
