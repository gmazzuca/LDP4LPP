import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# --- Parameters ---
gamma_sq = 0.4
eta = 1.0
theta = 2.0

# Selected xi values for Regimes (a), (b), (c)
xi_values_to_plot = [-0.25, 0.25, 0.9]

# --- Helper Functions (Same as before) ---

def get_params(xi, gamma_sq, eta, theta):
    if -gamma_sq < xi <= 0:
        kappa = gamma_sq - abs(xi)
        nu = theta / eta
        m1 = 1.0 / kappa
        m2 = abs(xi) / kappa
        n1 = gamma_sq - 1.0
        n2 = abs(xi)
        prefactor_pow = eta
        prefactor_const = eta
    elif 0 < xi <= 1 - gamma_sq:
        kappa = gamma_sq
        nu = theta / eta
        m1 = 1.0 / kappa
        m2 = (theta * xi) / (eta * kappa)
        n1 = 0.0
        n2 = 1.0 - gamma_sq - xi
        prefactor_pow = eta
        prefactor_const = eta
    elif 1 - gamma_sq < xi < 1:
        kappa = 1.0 - xi
        nu = eta / theta
        m1 = 1.0 / kappa
        m2 = xi / kappa
        n1 = 1.0 - gamma_sq - xi
        n2 = 0.0
        prefactor_pow = theta
        prefactor_const = theta
    else:
        raise ValueError(f"xi={xi} is out of valid range")
    return nu, m1, m2, n1, n2, prefactor_pow, prefactor_const

def compute_constants(nu, m1, m2, n1, n2):
    m1_tilde = m1 * (n2 - n1)
    denom = m1_tilde * nu + nu + m2 + 1
    term_pow = (m2 + 1) / denom
    c0 = ((m1_tilde * nu + nu + m2) * (term_pow**(1/nu + 1))) / (m2 + 1)
    c1 = (nu * (m1_tilde + 1) * (term_pow**(1/nu - 1))) / (denom**2)
    s0 = (1 + m2) / (nu * (1 + m1_tilde))
    D = np.sqrt(4 * c0 * c1 * nu + c1**2 * (nu - 1)**2)
    sa = - (nu - 1)/(2*nu) - D / (2 * nu * c1)
    sb = - (nu - 1)/(2*nu) + D / (2 * nu * c1)
    return c0, c1, s0, sa, sb

def J_func(s, c0, c1, nu):
    term = (1 + 1/s + 0j)**(1/nu)
    return (c1 * s + c0) * term

def solve_inverse_J(y, c0, c1, nu, sa, sb):
    s_guess = (sa + sb) / 2.0 - 0.1j 
    def equations(vars):
        s_real, s_imag = vars
        s = s_real + 1j * s_imag
        val = J_func(s, c0, c1, nu)
        return [val.real - y, val.imag]
    s_sol = fsolve(equations, [s_guess.real, s_guess.imag])
    return s_sol[0] + 1j*s_sol[1]

def omega_density(y, nu, m1, m2, n1, n2):
    c0, c1, s0, sa, sb = compute_constants(nu, m1, m2, n1, n2)
    a = J_func(sa, c0, c1, nu).real
    b = J_func(sb, c0, c1, nu).real
    if y <= a or y >= b:
        return 0.0
    s = solve_inverse_J(y, c0, c1, nu, sa, sb)
    m1_tilde = m1 * (n2 - n1)
    prefactor = (m2 + 1) * (m1_tilde * nu + nu + m2 + 1) / (nu * (m1_tilde + 1) * np.pi * y)
    val = 1.0 / (s - s0)
    return prefactor * abs(val.imag)

def mu_density(x, xi, gamma_sq, eta, theta):
    try:
        nu, m1, m2, n1, n2, p_pow, p_const = get_params(xi, gamma_sq, eta, theta)
    except ValueError:
        return 0.0
    y = x**p_pow
    c0, c1, s0, sa, sb = compute_constants(nu, m1, m2, n1, n2)
    a = J_func(sa, c0, c1, nu).real
    b = J_func(sb, c0, c1, nu).real
    if y <= a or y >= b:
        return 0.0
    om = omega_density(y, nu, m1, m2, n1, n2)
    return p_const * (x**(p_const - 1)) * om

# --- Plotting Code ---

plt.figure(figsize=(10, 6))

x_grid = np.linspace(0.001, 0.999, 400) # Grid for x-axis

for xi in xi_values_to_plot:
    print(f"Calculating density for xi = {xi}...")
    
    # Calculate mu(x)
    mu_vals = []
    for x in x_grid:
        mu_vals.append(mu_density(x, xi, gamma_sq, eta, theta))
    
    # Determine label
    regime = ""
    if -gamma_sq < xi <= 0: regime = "(a)"
    elif 0 < xi <= 1 - gamma_sq: regime = "(b)"
    elif 1 - gamma_sq < xi < 1: regime = "(c)"
    
    # Plot line
    line, = plt.plot(x_grid, mu_vals, linewidth=2, label=f'$\\xi={xi}$ {regime}')
    
    # Add fill under curve (using the same color as the line)
    plt.fill_between(x_grid, mu_vals, color=line.get_color(), alpha=0.15)

plt.title(f'Density $\\mu(x)$ for different $\\xi$ ($\\gamma^2={gamma_sq}, \\eta={eta}, \\theta={theta}$)')
plt.xlabel('$x$')
plt.ylabel('$\\mu(x)$')
plt.xlim(0, 1)
plt.ylim(bottom=0)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()