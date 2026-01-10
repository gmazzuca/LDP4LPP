import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import quad

# --- Parameters ---
gamma_sq = 0.4
eta = 1.0
theta = 2.0
xi_res = 500
x_res = 500

# --- Helper Functions (Standard) ---
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
        raise ValueError(f"xi={xi} out of valid range")
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
    a_val = J_func(sa, c0, c1, nu).real
    b_val = J_func(sb, c0, c1, nu).real
    if y <= a_val or y >= b_val:
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
    a_val = J_func(sa, c0, c1, nu).real
    b_val = J_func(sb, c0, c1, nu).real
    if y <= a_val or y >= b_val:
        return 0.0
    om = omega_density(y, nu, m1, m2, n1, n2)
    return p_const * (x**(p_const - 1)) * om

# --- Computation ---
xi_vals = np.linspace(-gamma_sq + 0.01, 0.99, xi_res)
x_vals = np.linspace(0.01, 0.99, x_res)
boundary = 1 - gamma_sq
xi_vals = xi_vals[np.abs(xi_vals - boundary) > 0.01]
xi_vals = xi_vals[np.abs(xi_vals) > 0.01]

X_grid, XI_grid = np.meshgrid(x_vals, xi_vals)
Z_grid = np.zeros_like(X_grid)

print("Computing grid values...")
for i in range(len(xi_vals)):
    xi = xi_vals[i]
    for j in range(len(x_vals)):
        x = x_vals[j]
        Z_grid[i, j] = mu_density(x, xi, gamma_sq, eta, theta)

# --- Plotting with Corrected Legend ---

# Apply the custom transformation for the colors
scale_factor = 3.0
Z_transformed = 1 - np.exp(-Z_grid / scale_factor)

plt.figure(figsize=(10, 6))

# Plot the transformed values
mesh = plt.pcolormesh(XI_grid, X_grid, Z_transformed, cmap='jet', shading='auto', vmin=0, vmax=1)

# --- LEGEND FIX ---
# We define the true values we want to see on the legend
true_ticks = [0, 0.5, 1, 1.5, 2, 3, 5, 10, 40]

# We calculate where these true values lie on the transformed [0,1] scale
transformed_locs = [1 - np.exp(-v / scale_factor) for v in true_ticks]
tick_labels = [str(v) for v in true_ticks]

# Create colorbar with custom ticks
cbar = plt.colorbar(mesh)
cbar.set_ticks(transformed_locs)
cbar.set_ticklabels(tick_labels)
cbar.set_label(r'Density $\mu(x)$', rotation=270, labelpad=15)

plt.xlabel(r'$\xi$ (Time Parameter)')
plt.ylabel(r'$x$ (Position)')
plt.title(r'Asymptotic for the point process, $\theta=%.1f$, $\eta=%.1f$, $\gamma^2=%.2f$' % (theta, eta, gamma_sq))

# Regime boundaries
plt.axvline(x=1-gamma_sq, color='white', linestyle='--', alpha=0.6, label=r'Transition $1-\gamma^2$')
plt.axvline(x=0, color='white', linestyle='--', alpha=0.6, label=r'Transition $0$')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()