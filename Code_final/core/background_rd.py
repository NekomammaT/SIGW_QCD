"""Exact radiation-domination (RD) background and analytic mode functions,
used as ground truth to validate the WKB pipeline (never fed into the
numerical pipeline itself). a(eta)=eta, H(eta)=1/eta, cs2=w=1/3."""
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)


def a_of_eta(eta): return eta
def H_of_eta(eta): return 1.0 / eta


CS2 = W = 1.0 / 3.0


def cs2_of_eta(eta): return CS2 * jnp.ones_like(eta)
def w_of_eta(eta): return W * jnp.ones_like(eta)


# Generic-form coefficients, X'' + f1 X' + (k^2 f3 + f2) X = 0
def f1_Phi(eta): return 3.0 * H_of_eta(eta) * (1.0 + cs2_of_eta(eta))
def f2_Phi(eta): return 3.0 * H_of_eta(eta)**2 * (cs2_of_eta(eta) - w_of_eta(eta))
def f3_Phi(eta): return cs2_of_eta(eta)
def f1_g(eta): return jnp.zeros_like(eta)
def f2_g(eta): return -0.5 * (1.0 - 3.0 * w_of_eta(eta)) * H_of_eta(eta)**2
def f3_g(eta): return jnp.ones_like(eta)


# Closed-form solutions (sympy-verified), for validation only
_X_SMALL = 0.1  # below this, use the Taylor series (avoids 0/0 cancellation)


def Phi_analytic(eta, k):
    """Phi(x) = 3(sin x - x cos x)/x^3, x = k eta / sqrt(3)."""
    x = k * eta / jnp.sqrt(3.0)
    x_safe = jnp.where(jnp.abs(x) < _X_SMALL, 1.0, x)
    exact = 3.0 * (jnp.sin(x_safe) - x_safe * jnp.cos(x_safe)) / x_safe**3
    x2 = x**2
    series = 1.0 - x2 / 10.0 + x2**2 / 280.0 - x2**3 / 15120.0 + x2**4 / 1330560.0
    return jnp.where(jnp.abs(x) < _X_SMALL, series, exact)


def g1_analytic(eta, k):
    """g1(0)=0, g1'(0)=1; exact since a''/a=0 in RD."""
    return jnp.sin(k * eta) / k


def g2_analytic(eta, k):
    """g2(0)=1, g2'(0)=0."""
    return jnp.cos(k * eta)
