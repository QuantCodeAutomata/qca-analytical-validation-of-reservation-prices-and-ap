"""
Avellaneda-Stoikov market-making model: core analytical formulas.

All functions are pure Python / NumPy; no external market-making library
was found to provide these specific closed-form expressions, so they are
implemented directly from the paper.

Reference: Avellaneda, M. & Stoikov, S. (2008). "High-frequency trading in a
limit order book." Quantitative Finance, 8(3), 217-224.
"""

import numpy as np


def frozen_inventory_value(
    x: float,
    s: float,
    q: float,
    t: float,
    T: float,
    gamma: float,
    sigma: float,
) -> float:
    """Frozen-inventory CARA utility value function.

    Computes the expected CARA utility under the assumption that the inventory
    q is held fixed until T and S_T | S_t = s is Gaussian(s, sigma^2 * tau):

        v(x, s, q, t) = E[-exp(-gamma * (x + q * S_T))]
                      = -exp(-gamma*x)
                        * exp(-gamma*q*s)
                        * exp(0.5 * gamma^2 * q^2 * sigma^2 * tau)

    where tau = T - t and the last factor arises from the MGF of the Gaussian.

    Paper: Avellaneda-Stoikov (eq. 9). No library equivalent found (Context7).

    Parameters
    ----------
    x : float
        Current cash wealth.
    s : float
        Current mid-price S_t.
    q : float
        Inventory (number of shares held).
    t : float
        Current time.
    T : float
        Terminal time horizon.
    gamma : float
        Absolute risk-aversion coefficient (> 0).
    sigma : float
        Mid-price diffusion volatility (> 0).

    Returns
    -------
    float
        Value of the frozen-inventory utility function (always negative).
    """
    tau = T - t
    return (
        -np.exp(-gamma * x)
        * np.exp(-gamma * q * s)
        * np.exp(0.5 * gamma**2 * q**2 * sigma**2 * tau)
    )


def reservation_ask(
    s: float,
    q: float,
    t: float,
    T: float,
    gamma: float,
    sigma: float,
) -> float:
    """Reservation ask price r^a(s, q, t).

    The ask price at which a risk-averse agent is indifferent between the
    current position q and selling one unit, derived from the utility
    indifference condition:

        v(x + r^a, s, q-1, t) = v(x, s, q, t)

    Closed form:

        r^a(s, q, t) = s + ((1 - 2*q) / 2) * gamma * sigma^2 * (T - t)

    Parameters
    ----------
    s : float
        Current mid-price.
    q : float
        Current inventory.
    t : float
        Current time.
    T : float
        Terminal time horizon.
    gamma : float
        Absolute risk-aversion coefficient.
    sigma : float
        Mid-price diffusion volatility.

    Returns
    -------
    float
        Reservation ask price.
    """
    tau = T - t
    return s + ((1 - 2 * q) / 2) * gamma * sigma**2 * tau


def reservation_bid(
    s: float,
    q: float,
    t: float,
    T: float,
    gamma: float,
    sigma: float,
) -> float:
    """Reservation bid price r^b(s, q, t).

    The bid price at which a risk-averse agent is indifferent between the
    current position q and buying one unit, derived from:

        v(x - r^b, s, q+1, t) = v(x, s, q, t)

    Closed form:

        r^b(s, q, t) = s + ((-1 - 2*q) / 2) * gamma * sigma^2 * (T - t)

    Parameters
    ----------
    s : float
        Current mid-price.
    q : float
        Current inventory.
    t : float
        Current time.
    T : float
        Terminal time horizon.
    gamma : float
        Absolute risk-aversion coefficient.
    sigma : float
        Mid-price diffusion volatility.

    Returns
    -------
    float
        Reservation bid price.
    """
    tau = T - t
    return s + ((-1 - 2 * q) / 2) * gamma * sigma**2 * tau


def reservation_price(
    s: float,
    q: float,
    t: float,
    T: float,
    gamma: float,
    sigma: float,
) -> float:
    """Reservation (indifference) price r(s, q, t).

    The mid-point between the reservation ask and reservation bid, equal to
    the certainty-equivalent price adjusted for inventory risk:

        r(s, q, t) = (r^a + r^b) / 2 = s - q * gamma * sigma^2 * (T - t)

    A positive inventory (q > 0) lowers the reservation price below s,
    incentivising the agent to sell; negative inventory raises it.

    Parameters
    ----------
    s : float
        Current mid-price.
    q : float
        Current inventory.
    t : float
        Current time.
    T : float
        Terminal time horizon.
    gamma : float
        Absolute risk-aversion coefficient.
    sigma : float
        Mid-price diffusion volatility.

    Returns
    -------
    float
        Reservation price.
    """
    tau = T - t
    return s - q * gamma * sigma**2 * tau


def half_spread(
    t: float,
    T: float,
    gamma: float,
    sigma: float,
    k: float,
) -> float:
    """Optimal half-spread h_t.

    Derived from the first-order condition of the HJB equation with Poisson
    order arrivals lambda = A * exp(-k * delta):

        h_t = 0.5 * gamma * sigma^2 * (T - t) + (1/gamma) * ln(1 + gamma/k)

    First term: half the inventory-risk spread (r^a - r^b) / 2.
    Second term: liquidity adjustment from the intensity-weighted FOC.

    Parameters
    ----------
    t : float
        Current time.
    T : float
        Terminal time horizon.
    gamma : float
        Absolute risk-aversion coefficient.
    sigma : float
        Mid-price diffusion volatility.
    k : float
        Order-book depth parameter controlling how quickly arrival rates
        decay with distance from the mid-price (> 0).

    Returns
    -------
    float
        Optimal half-spread.
    """
    tau = T - t
    return 0.5 * gamma * sigma**2 * tau + (1.0 / gamma) * np.log(1 + gamma / k)


def total_spread(
    t: float,
    T: float,
    gamma: float,
    sigma: float,
    k: float,
) -> float:
    """Optimal total bid-ask spread 2 * h_t.

        spread_t = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)

    Parameters
    ----------
    t : float
        Current time.
    T : float
        Terminal time horizon.
    gamma : float
        Absolute risk-aversion coefficient.
    sigma : float
        Mid-price diffusion volatility.
    k : float
        Order-book depth parameter (> 0).

    Returns
    -------
    float
        Optimal total spread.
    """
    tau = T - t
    return gamma * sigma**2 * tau + (2.0 / gamma) * np.log(1 + gamma / k)


def optimal_quotes(
    s: float,
    q: float,
    t: float,
    T: float,
    gamma: float,
    sigma: float,
    k: float,
) -> tuple[float, float]:
    """Optimal ask and bid quotes (p^a, p^b).

    Places the quotes symmetrically around the reservation price:

        p^a = r(s, q, t) + h_t
        p^b = r(s, q, t) - h_t

    so that both market and inventory risk are jointly optimised.

    Parameters
    ----------
    s : float
        Current mid-price.
    q : float
        Current inventory.
    t : float
        Current time.
    T : float
        Terminal time horizon.
    gamma : float
        Absolute risk-aversion coefficient.
    sigma : float
        Mid-price diffusion volatility.
    k : float
        Order-book depth parameter (> 0).

    Returns
    -------
    tuple[float, float]
        (p_ask, p_bid) — optimal ask and bid quotes.
    """
    r = reservation_price(s, q, t, T, gamma, sigma)
    h = half_spread(t, T, gamma, sigma, k)
    return r + h, r - h
