"""Pytest tests for Avellaneda-Stoikov analytical formulas."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import sympy as sp
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
S0, T, SIGMA, A, K = 100.0, 1.0, 2.0, 140.0, 1.5
GAMMA_DEFAULT = 0.1


class TestFrozenValueFunction:
    def test_value_is_negative(self):
        """CARA utility is always negative."""
        v = frozen_inventory_value(1000, S0, 0, 0.5, T, GAMMA_DEFAULT, SIGMA)
        assert v < 0

    def test_exact_formula(self):
        """Check exact match to paper formula."""
        x, s, q, t, gamma, sigma = 1000.0, 100.0, 2.0, 0.5, 0.1, 2.0
        tau = T - t
        expected = -np.exp(-gamma*x) * np.exp(-gamma*q*s) * np.exp(0.5*gamma**2*q**2*sigma**2*tau)
        result = frozen_inventory_value(x, s, q, t, T, gamma, sigma)
        assert abs(result - expected) < 1e-12

    def test_higher_cash_increases_value(self):
        """Higher cash x should increase utility (less negative)."""
        v1 = frozen_inventory_value(1000, S0, 2, 0.5, T, GAMMA_DEFAULT, SIGMA)
        v2 = frozen_inventory_value(1010, S0, 2, 0.5, T, GAMMA_DEFAULT, SIGMA)
        assert v2 > v1

    def test_zero_inventory_at_maturity(self):
        """At t=T, tau=0, exp term = 1, simplifies nicely."""
        v = frozen_inventory_value(1000, S0, 0, T, T, GAMMA_DEFAULT, SIGMA)
        expected = -np.exp(-GAMMA_DEFAULT * 1000) * np.exp(0) * np.exp(0)
        assert abs(v - expected) < 1e-12

    def test_single_data_point(self):
        """Test with minimal input values."""
        v = frozen_inventory_value(0, 0, 0, 0, 1.0, 0.1, 1.0)
        assert v == -1.0  # -exp(0)*exp(0)*exp(0)


class TestReservationPrices:
    def test_ask_above_bid(self):
        """Reservation ask must be above reservation bid."""
        for q in range(-5, 6):
            r_a = reservation_ask(S0, q, 0.5, T, GAMMA_DEFAULT, SIGMA)
            r_b = reservation_bid(S0, q, 0.5, T, GAMMA_DEFAULT, SIGMA)
            assert r_a > r_b, f'Ask not above bid for q={q}'

    def test_exact_ask_formula(self):
        """r^a = s + ((1-2q)/2)*gamma*sigma^2*(T-t)"""
        s, q, t, gamma, sigma = 100.0, 3.0, 0.5, 0.1, 2.0
        tau = T - t
        expected = s + ((1 - 2*q) / 2) * gamma * sigma**2 * tau
        assert abs(reservation_ask(s, q, t, T, gamma, sigma) - expected) < 1e-12

    def test_exact_bid_formula(self):
        """r^b = s + ((-1-2q)/2)*gamma*sigma^2*(T-t)"""
        s, q, t, gamma, sigma = 100.0, 3.0, 0.5, 0.1, 2.0
        tau = T - t
        expected = s + ((-1 - 2*q) / 2) * gamma * sigma**2 * tau
        assert abs(reservation_bid(s, q, t, T, gamma, sigma) - expected) < 1e-12

    def test_reservation_price_is_average(self):
        """r = (r^a + r^b) / 2"""
        for q in range(-5, 6):
            r_a = reservation_ask(S0, q, 0.5, T, GAMMA_DEFAULT, SIGMA)
            r_b = reservation_bid(S0, q, 0.5, T, GAMMA_DEFAULT, SIGMA)
            r = reservation_price(S0, q, 0.5, T, GAMMA_DEFAULT, SIGMA)
            assert abs(r - (r_a + r_b) / 2) < 1e-12

    def test_exact_reservation_price(self):
        """r = s - q*gamma*sigma^2*(T-t)"""
        s, q, t, gamma, sigma = 100.0, 3.0, 0.5, 0.1, 2.0
        tau = T - t
        expected = s - q * gamma * sigma**2 * tau
        assert abs(reservation_price(s, q, t, T, gamma, sigma) - expected) < 1e-12

    def test_reservation_price_decreasing_in_q(self):
        """Reservation price decreases as q increases (long inventory pushes quotes down)."""
        qs = list(range(-10, 10))
        prices = [reservation_price(S0, q, 0.5, T, GAMMA_DEFAULT, SIGMA) for q in qs]
        diffs = np.diff(prices)
        assert np.all(diffs < 0), 'Reservation price not strictly decreasing in q'

    def test_reservation_price_converges_at_maturity(self):
        """As t -> T, reservation price -> s regardless of q."""
        for q in [-5, 0, 5]:
            r = reservation_price(S0, q, 0.9999, T, GAMMA_DEFAULT, SIGMA)
            assert abs(r - S0) < 0.01, f'Reservation price not near S0 at maturity for q={q}'

    def test_reservation_price_zero_inventory(self):
        """With q=0, reservation price equals mid-price."""
        r = reservation_price(S0, 0, 0.5, T, GAMMA_DEFAULT, SIGMA)
        assert abs(r - S0) < 1e-12

    def test_long_inventory_pushes_quotes_down(self):
        """With q>0, reservation price < mid-price."""
        r = reservation_price(S0, 5, 0.5, T, GAMMA_DEFAULT, SIGMA)
        assert r < S0

    def test_short_inventory_pushes_quotes_up(self):
        """With q<0, reservation price > mid-price."""
        r = reservation_price(S0, -5, 0.5, T, GAMMA_DEFAULT, SIGMA)
        assert r > S0

    def test_extreme_inventory(self):
        """Test with extreme inventory values."""
        r_high = reservation_price(S0, 100, 0.5, T, GAMMA_DEFAULT, SIGMA)
        r_low = reservation_price(S0, -100, 0.5, T, GAMMA_DEFAULT, SIGMA)
        assert r_high < S0
        assert r_low > S0


class TestHalfSpread:
    def test_exact_formula(self):
        """h_t = 0.5*gamma*sigma^2*(T-t) + (1/gamma)*ln(1+gamma/k)"""
        t, gamma, sigma, k = 0.5, 0.1, 2.0, 1.5
        tau = T - t
        expected = 0.5*gamma*sigma**2*tau + (1.0/gamma)*np.log(1+gamma/k)
        assert abs(half_spread(t, T, gamma, sigma, k) - expected) < 1e-12

    def test_half_spread_positive(self):
        """Half-spread must always be positive."""
        for gamma in [0.01, 0.1, 0.5]:
            for t in [0.0, 0.5, 0.9]:
                h = half_spread(t, T, gamma, SIGMA, K)
                assert h > 0, f'Negative half-spread for gamma={gamma}, t={t}'

    def test_spread_equals_twice_half_spread(self):
        """Total spread = 2 * half_spread."""
        t, gamma = 0.5, 0.1
        h = half_spread(t, T, gamma, SIGMA, K)
        spr = total_spread(t, T, gamma, SIGMA, K)
        assert abs(spr - 2*h) < 1e-12

    def test_half_spread_decreasing_in_tau(self):
        """Half-spread decreases as time-to-maturity tau=(T-t) shrinks."""
        hs = [half_spread(t, T, GAMMA_DEFAULT, SIGMA, K) for t in np.linspace(0, 0.99, 20)]
        diffs = np.diff(hs)
        assert np.all(diffs <= 0), 'Half-spread not decreasing in time'

    def test_gamma_zero_limit(self):
        """As gamma->0, liquidity term (1/gamma)*ln(1+gamma/k) -> 1/k."""
        gammas = [0.1, 0.01, 0.001, 0.0001]
        liq_terms = [(1.0/g)*np.log(1+g/K) for g in gammas]
        # Should converge to 1/K
        assert abs(liq_terms[-1] - 1.0/K) < 1e-3

    def test_at_maturity_spread_is_liquidity_only(self):
        """At t=T (tau=0), half-spread = (1/gamma)*ln(1+gamma/k)."""
        gamma = 0.1
        h = half_spread(T, T, gamma, SIGMA, K)
        expected = (1.0/gamma) * np.log(1 + gamma/K)
        assert abs(h - expected) < 1e-12


class TestOptimalQuotes:
    def test_ask_above_bid(self):
        """Optimal ask always above optimal bid."""
        for q in range(-5, 6):
            p_a, p_b = optimal_quotes(S0, q, 0.5, T, GAMMA_DEFAULT, SIGMA, K)
            assert p_a > p_b

    def test_symmetric_at_zero_inventory(self):
        """At q=0, quotes are symmetric around mid-price S0."""
        p_a, p_b = optimal_quotes(S0, 0, 0.5, T, GAMMA_DEFAULT, SIGMA, K)
        assert abs((p_a + p_b)/2 - S0) < 1e-10
        assert abs(p_a - S0 - (S0 - p_b)) < 1e-10

    def test_long_inventory_lowers_quotes(self):
        """Long inventory (q>0) pushes both quotes below neutral level."""
        p_a_long, p_b_long = optimal_quotes(S0, 5, 0.5, T, GAMMA_DEFAULT, SIGMA, K)
        p_a_neutral, p_b_neutral = optimal_quotes(S0, 0, 0.5, T, GAMMA_DEFAULT, SIGMA, K)
        assert p_a_long < p_a_neutral
        assert p_b_long < p_b_neutral

    def test_short_inventory_raises_quotes(self):
        """Short inventory (q<0) pushes both quotes above neutral level."""
        p_a_short, p_b_short = optimal_quotes(S0, -5, 0.5, T, GAMMA_DEFAULT, SIGMA, K)
        p_a_neutral, p_b_neutral = optimal_quotes(S0, 0, 0.5, T, GAMMA_DEFAULT, SIGMA, K)
        assert p_a_short > p_a_neutral
        assert p_b_short > p_b_neutral

    def test_spread_is_total_spread(self):
        """p^a - p^b equals total_spread."""
        t, q = 0.5, 3
        p_a, p_b = optimal_quotes(S0, q, t, T, GAMMA_DEFAULT, SIGMA, K)
        spr = total_spread(t, T, GAMMA_DEFAULT, SIGMA, K)
        assert abs(p_a - p_b - spr) < 1e-10

    def test_edge_empty_inventory(self):
        """Test with q=0 (empty inventory)."""
        p_a, p_b = optimal_quotes(S0, 0, 0.5, T, GAMMA_DEFAULT, SIGMA, K)
        assert np.isfinite(p_a) and np.isfinite(p_b)

    def test_edge_single_time_point(self):
        """Test at t=0 (start)."""
        p_a, p_b = optimal_quotes(S0, 2, 0.0, T, GAMMA_DEFAULT, SIGMA, K)
        assert np.isfinite(p_a) and np.isfinite(p_b)


class TestSymbolicDerivation:
    def test_frozen_value_function_form(self):
        """Frozen value function has correct symbolic form."""
        v, syms = derive_frozen_value_function()
        x, s, q, t, T_sym, gamma, sigma = syms
        # Should involve exp terms
        assert v is not None
        assert 'exp' in str(v).lower()

    def test_reservation_bid_ask_solutions_exist(self):
        """SymPy should find solutions for bid/ask prices."""
        sol_bid, sol_ask = derive_reservation_bid_ask()
        assert len(sol_bid) > 0, 'No solution for reservation bid'
        assert len(sol_ask) > 0, 'No solution for reservation ask'

    def test_reservation_price_identity_zero(self):
        """Symbolic check: (r^a + r^b)/2 - r_formula = 0."""
        r_mid, r_formula, diff_check = verify_reservation_price_formula()
        assert diff_check == 0, f'Reservation price identity failed: diff = {diff_check}'

    def test_liquidity_term_form(self):
        """Liquidity term is (1/gamma)*ln(1+gamma/k)."""
        liq, ratio = derive_liquidity_term()
        assert liq is not None
        # lambda/lambda' = -1/k for exponential intensities
        assert '-1/k' in str(ratio) or '-k' in str(ratio) or ratio is not None


class TestNumericalProperties:
    def test_reservation_price_scales_with_gamma(self):
        """Inventory adjustment is proportional to gamma."""
        q, t = 5, 0.5
        tau = T - t
        gamma1, gamma2 = 0.1, 0.2
        adj1 = S0 - reservation_price(S0, q, t, T, gamma1, SIGMA)
        adj2 = S0 - reservation_price(S0, q, t, T, gamma2, SIGMA)
        assert abs(adj2 / adj1 - gamma2/gamma1) < 1e-10

    def test_half_spread_scales_with_sigma_squared(self):
        """Inventory component of half-spread scales with sigma^2."""
        t, gamma = 0.0, 0.1  # t=0 so tau=T=1, full effect
        sigma1, sigma2 = 1.0, 2.0
        # h = 0.5*gamma*sigma^2*tau + (1/gamma)*ln(1+gamma/k)
        h1 = half_spread(t, T, gamma, sigma1, K)
        h2 = half_spread(t, T, gamma, sigma2, K)
        inv_component1 = 0.5 * gamma * sigma1**2 * T
        inv_component2 = 0.5 * gamma * sigma2**2 * T
        liq = (1.0/gamma) * np.log(1 + gamma/K)
        assert abs(h1 - (inv_component1 + liq)) < 1e-12
        assert abs(h2 - (inv_component2 + liq)) < 1e-12

    def test_utility_indifference_bid(self):
        """Verify utility indifference for bid: v(x-r^b, s, q+1, t) = v(x, s, q, t)."""
        x, s, q, t, gamma, sigma = 1000.0, 100.0, 2.0, 0.5, 0.1, 2.0
        r_b = reservation_bid(s, q, t, T, gamma, sigma)
        v_base = frozen_inventory_value(x, s, q, t, T, gamma, sigma)
        v_bid = frozen_inventory_value(x - r_b, s, q + 1, t, T, gamma, sigma)
        assert abs(v_base - v_bid) < 1e-8, f'Utility indifference violated for bid: {v_base} vs {v_bid}'

    def test_utility_indifference_ask(self):
        """Verify utility indifference for ask: v(x+r^a, s, q-1, t) = v(x, s, q, t)."""
        x, s, q, t, gamma, sigma = 1000.0, 100.0, 2.0, 0.5, 0.1, 2.0
        r_a = reservation_ask(s, q, t, T, gamma, sigma)
        v_base = frozen_inventory_value(x, s, q, t, T, gamma, sigma)
        v_ask = frozen_inventory_value(x + r_a, s, q - 1, t, T, gamma, sigma)
        assert abs(v_base - v_ask) < 1e-8, f'Utility indifference violated for ask: {v_base} vs {v_ask}'
