import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import warnings

# Suppress runtime warnings from the optimizer
warnings.filterwarnings('ignore')

# --- CONFIGURATION FOR LATEX/PAPER PLOTTING (Based on Simple_density.py) ---
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'lines.linewidth': 2.5
})

def J_map(s, c0, c1, nu):
    """Conformal mapping J_{c0, c1}(s)."""
    return (c1 * s + c0) * ((s + 1) / s)**(1 / nu)

def get_regime_params(xi, gamma_sq, eta, theta, beta):
    """Maps physical constants to the model parameters (Corollary 3.4)."""
    if theta > eta:
        nu = theta / eta
        prefactor_pow, prefactor_const = eta, eta
        if -gamma_sq < xi <= 0:
            kappa = gamma_sq - abs(xi)
            m1, n1, n2 = abs(xi) / kappa, -abs(xi), 1.0 - gamma_sq
            alpha, rho = theta, eta
            is_omega_nu = True
        elif 0 < xi <= 1 - gamma_sq:
            kappa = gamma_sq
            m1, n1, n2 = (theta * xi) / (eta * kappa), 0.0, 1.0 - gamma_sq - xi
            alpha, rho = theta, eta
            is_omega_nu = True
        elif 1 - gamma_sq < xi < 1:
            kappa = 1.0 - xi
            m1, n1, n2 = (theta * xi) / (eta * kappa), 1.0 - gamma_sq - xi, 0.0
            alpha, rho = eta, eta
            is_omega_nu = False
        else:
            raise ValueError(f"xi={xi} out of bounds")
    else:
        nu = eta / theta
        prefactor_pow, prefactor_const = theta, theta
        if -gamma_sq < xi <= 0:
            kappa = gamma_sq - abs(xi)
            m1, n1, n2 = (abs(xi) * eta) / (kappa * theta), -abs(xi), 1.0 - gamma_sq
            alpha, rho = theta, theta
            is_omega_nu = False
        elif 0 < xi <= 1 - gamma_sq:
            kappa = gamma_sq
            m1, n1, n2 = xi / kappa, 0.0, 1.0 - gamma_sq - xi
            alpha, rho = theta, theta
            is_omega_nu = False
        elif 1 - gamma_sq < xi < 1:
            kappa = 1.0 - xi
            m1, n1, n2 = xi / kappa, 1.0 - gamma_sq - xi, 0.0
            alpha, rho = eta, theta
            is_omega_nu = True
        else:
            raise ValueError(f"xi={xi} out of bounds")
            
    return nu, kappa, m1, n1, n2, alpha, rho, is_omega_nu, prefactor_pow, prefactor_const

def solve_moduli(xi, gamma_sq, eta, theta, beta):
    """Computes parameters using the analytical RHP solution."""
    nu, kappa, m1, n1, n2, alpha, rho, is_omega_nu, pref_pow, pref_const = get_regime_params(xi, gamma_sq, eta, theta, beta)
    x_max = np.exp(-rho * beta * (gamma_sq - kappa))
    
    if is_omega_nu:
        A = np.exp((alpha * beta * kappa / nu) * (nu + 1 + m1 + (nu / kappa) * (n2 - n1)))
        B = np.exp((alpha * beta * kappa / nu) * (1 + m1))
        s1 = A * (B - 1) / (A - B) 
        s2 = (B - 1) / (A - B)
        K1 = np.exp(n1 * alpha * beta / nu) * (A * (B - 1) / (B * (A - 1)))**(1 / nu)
        K2 = np.exp(n2 * alpha * beta / nu) * ((B - 1) / (A - 1))**(1 / nu)
    else:
        A = np.exp(alpha * beta * kappa * (nu + 1 + m1 + (1.0 / kappa) * (n2 - n1)))
        B = np.exp(alpha * beta * kappa * (1 + m1 + (1.0 / kappa) * (n2 - n1)))
        s1 = A * (B - 1) / (A - B)
        s2 = (B - 1) / (A - B)
        K1 = np.exp(n2 * alpha * beta) * ((A * (B - 1)) / (B * (A - 1)))**(1 / nu)
        K2 = np.exp(n1 * alpha * beta) * ((B - 1) / (A - 1))**(1 / nu)
        
    c1 = (K1 - K2) / (s1 - s2)
    c0 = (K2 * s1 - K1 * s2) / (s1 - s2)
    
    disc = np.maximum(0, 4 * c0 * c1 * nu + (c1**2) * ((nu - 1)**2))
    sa = -(nu-1)/(2*nu) - np.sqrt(disc)/(2*nu*c1)
    sb = -(nu-1)/(2*nu) + np.sqrt(disc)/(2*nu*c1)
    
    J_sa = np.real(J_map(sa+0j, c0, c1, nu))
    J_sb = np.real(J_map(sb+0j, c0, c1, nu))
    a, b = min(J_sa, J_sb), max(J_sa, J_sb)
    is_supercritical = (s1 > sb) if is_omega_nu else (s2 < sb)
        
    return {
        'nu': nu, 'kappa': kappa, 'alpha': alpha, 'rho': rho, 'is_omega_nu': is_omega_nu,
        'is_supercritical': is_supercritical, 'x_max': x_max,
        'c0': c0, 'c1': c1, 's1': s1, 's2': s2, 'sa': sa, 'sb': sb, 'a': a, 'b': b, 
        'pref_pow': pref_pow, 'pref_const': pref_const
    }

def compute_density_row(x_grid, xi, gamma_sq, eta, theta, beta):
    """Computes a single xi-slice of the density mu(x)."""
    try:
        p = solve_moduli(xi, gamma_sq, eta, theta, beta)
    except Exception:
        return np.zeros_like(x_grid), None
        
    y_grid = x_grid ** p['pref_pow']
    mu_vals = np.zeros_like(x_grid)
    mid, R = (p['sa'] + p['sb']) / 2.0, abs(p['sb'] - p['sa']) / 2.0
    
    for i, (x_val, y_val) in enumerate(zip(x_grid, y_grid)):
        if y_val <= p['a']:
            mu_vals[i] = 0.0
        elif y_val >= p['b']:
            omega = 1.0 / (beta * p['rho'] * p['kappa'] * y_val) if p['is_supercritical'] and y_val <= p['x_max'] else 0.0
            mu_vals[i] = p['pref_const'] * (x_val**(p['pref_const'] - 1)) * omega
        else:
            def obj(vars_arr):
                val = J_map(vars_arr[0] + 1j*vars_arr[1], p['c0'], p['c1'], p['nu'])
                return [np.real(val) - y_val, np.imag(val)]
            t = np.clip((y_val - p['a']) / (p['b'] - p['a']), 0.01, 0.99) 
            s_guess = mid + R * np.exp(1j * np.pi * t)
            res = least_squares(obj, [np.real(s_guess), np.imag(s_guess)], bounds=([-np.inf, 1e-12], [np.inf, np.inf]))
            Ip = res.x[0] + 1j*res.x[1]
            arg_s = np.abs(np.angle((p['s1'] - Ip) / (p['s2'] - Ip))) if p['is_omega_nu'] else np.abs(np.angle((p['s2'] - Ip) / (p['s1'] - Ip)))
            omega = (1.0 / (np.pi * beta * p['rho'] * p['kappa'] * y_val)) * arg_s
            mu_vals[i] = p['pref_const'] * (x_val**(p['pref_const'] - 1)) * max(0.0, omega)
    return np.nan_to_num(mu_vals), p

# --- PARAMETERS ---
beta, gamma_sq, eta, theta = 1.0, 0.4, 2.0, 5.0
xi_values_1d = [-0.2, 0.2, 0.8]  # Representative values for the 3 regimes
xi_res, x_res = 300, 400

fig, (ax_heat, ax_1d) = plt.subplots(1, 2, figsize=(12, 6))

# --- LEFT: HEATMAP ---
print("Computing heatmap data...")
xi_vals = np.linspace(-gamma_sq + 0.005, 0.995, xi_res)
xi_vals = xi_vals[np.abs(xi_vals - (1 - gamma_sq)) > 0.01]
xi_vals = xi_vals[np.abs(xi_vals) > 0.01]
x_vals = np.linspace(0.005, 0.995, x_res)
X_grid, XI_grid = np.meshgrid(x_vals, xi_vals)
Z_grid = np.zeros_like(X_grid)
x_b_traj, x_max_traj = np.zeros(len(xi_vals)), np.zeros(len(xi_vals))

for i, xi in enumerate(xi_vals):
    mu_vals, p = compute_density_row(x_vals, xi, gamma_sq, eta, theta, beta)
    Z_grid[i, :] = mu_vals
    if p is not None:
        x_b_traj[i] = min(max(0.0, p['b']) ** (1.0 / p['pref_pow']), 1)
        x_max_traj[i] = min(max(0.0, p['x_max']) ** (1.0 / p['pref_pow']), 1)

scale_factor, true_ticks = 3.0, [0, 0.5, 1, 2, 5, 20]
Z_trans = 1 - np.exp(-Z_grid / scale_factor)
mesh = ax_heat.pcolormesh(XI_grid, X_grid, Z_trans, cmap='jet', shading='auto', vmin=0, vmax=1)
ax_heat.plot(xi_vals, x_b_traj, color="#3FF806", lw=3, ls='--', label='Arctic Curve')
ax_heat.plot(xi_vals, x_max_traj, color='#00FFFF', lw=2, ls='-.', label='$x_{max}$ Constraint')
ax_heat.axvline(x=0, color='white', ls=':', alpha=0.8)
ax_heat.axvline(x=1-gamma_sq, color='white', ls=':', alpha=0.8)

cbar = fig.colorbar(mesh, ax=ax_heat, fraction=0.046, pad=0.04)
cbar.set_ticks([1 - np.exp(-v / scale_factor) for v in true_ticks])
cbar.set_ticklabels([str(v) for v in true_ticks])
ax_heat.set_xlabel(r'$\xi$'); ax_heat.set_ylabel(r'$x$')
ax_heat.set_title("Equilibrium Measure Heatmap")
ax_heat.legend(loc='lower right', fontsize=12)
ax_heat.set_ylim([0,1])

# --- RIGHT: 1D DENSITY ---
print("Computing 1D profile data...")
x_1d = np.linspace(0.001, 0.999, 500)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i, xi in enumerate(xi_values_1d):
    mu_vals, p = compute_density_row(x_1d, xi, gamma_sq, eta, theta, beta)
    ax_1d.plot(x_1d, mu_vals, color=colors[i], label=f'$\\xi={xi}$')
    ax_1d.fill_between(x_1d, mu_vals, color=colors[i], alpha=0.1)
    # Upper bound 1/(beta * kappa * x)
    ub = 1.0 / (beta * p['kappa'] * x_1d)
    ax_1d.plot(x_1d, ub, color=colors[i], ls=':', lw=1.5, alpha=0.6)

ax_1d.set_xlabel('$x$'); ax_1d.set_ylabel(r'$\mu(x)$', rotation=0, labelpad=25)
ax_1d.set_title(f"Densities ($\gamma^2={gamma_sq}, \\beta={beta}, \\eta={eta},\\theta={theta}$)")
ax_1d.set_ylim(0, 12); ax_1d.grid(True, alpha=0.2)
ax_1d.legend()
ax_1d.set_xlim(0,1)

plt.tight_layout()
plt.savefig("combined_heatmap_density.pdf", bbox_inches='tight', dpi=300)
plt.show()