import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import warnings

# Suppress runtime warnings from the optimizer
warnings.filterwarnings('ignore')

# --- CONFIGURATION FOR PLOTTING ---
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

def J_map(s, c0, c1, nu):
    """Conformal mapping J_{c0, c1}(s)."""
    return (c1 * s + c0) * ((s + 1) / s)**(1 / nu)

def get_regime_params(xi, gamma_sq, eta, theta, beta):
    """Maps physical constants exactly to the cleaned Corollary parameters."""
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
    """Computes explicit parameters using the precise geometric analytical collapse."""
    nu, kappa, m1, n1, n2, alpha, rho, is_omega_nu, pref_pow, pref_const = get_regime_params(xi, gamma_sq, eta, theta, beta)
    x_max = np.exp(-rho * beta * (gamma_sq - kappa))
    
    if is_omega_nu: # Model Problem 1
        A = np.exp((alpha * beta * kappa / nu) * (nu + 1 + m1 + (nu / kappa) * (n2 - n1)))
        B = np.exp((alpha * beta * kappa / nu) * (1 + m1))
        s1 = A * (B - 1) / (A - B) 
        s2 = (B - 1) / (A - B)
        K1 = np.exp(n1 * alpha * beta / nu) * (A * (B - 1) / (B * (A - 1)))**(1 / nu)
        K2 = np.exp(n2 * alpha * beta / nu) * ((B - 1) / (A - 1))**(1 / nu)
    else: # Model Problem 2
        A = np.exp(alpha * beta * kappa * (nu + 1 + m1 + (1.0 / kappa) * (n2 - n1)))
        B = np.exp(alpha * beta * kappa * (1 + m1 + (1.0 / kappa) * (n2 - n1)))
        s1 = A * (B - 1) / (A - B)
        s2 = (B - 1) / (A - B)
        K1 = np.exp(n2 * alpha * beta) * ((A * (B - 1)) / (B * (A - 1)))**(1 / nu)
        K2 = np.exp(n1 * alpha * beta) * ((B - 1) / (A - 1))**(1 / nu)
        
    c1 = (K1 - K2) / (s1 - s2)
    c0 = (K2 * s1 - K1 * s2) / (s1 - s2)
    
    discriminant = np.maximum(0, 4 * c0 * c1 * nu + (c1**2) * ((nu - 1)**2))
    term1 = -(nu - 1) / (2 * nu)
    term2 = np.sqrt(discriminant) / (2 * nu * c1)
    
    sa = min(term1 + term2, term1 - term2)
    sb = max(term1 + term2, term1 - term2)
    
    J_sa = np.real(J_map(sa+0j, c0, c1, nu))
    J_sb = np.real(J_map(sb+0j, c0, c1, nu))
    a, b = min(J_sa, J_sb), max(J_sa, J_sb)
    
    is_supercritical = (s1 > sb) if is_omega_nu else (s2 < sb)
        
    return {
        'nu': nu, 'kappa': kappa, 'alpha': alpha, 'rho': rho, 'is_omega_nu': is_omega_nu,
        'is_supercritical': is_supercritical, 'x_max': x_max,
        'c0': c0, 'c1': c1, 's1': s1, 's2': s2,
        'sa': sa, 'sb': sb, 'a': a, 'b': b, 
        'pref_pow': pref_pow, 'pref_const': pref_const
    }

def compute_density_row(x_grid, xi, gamma_sq, eta, theta, beta):
    """Vectorized base evaluation. Returns the density row and the parameter dict."""
    try:
        p = solve_moduli(xi, gamma_sq, eta, theta, beta)
    except Exception:
        return np.zeros_like(x_grid), None
        
    y_grid = x_grid ** p['pref_pow']
    mu_vals = np.zeros_like(x_grid)
    
    mid = (p['sa'] + p['sb']) / 2.0
    R = abs(p['sb'] - p['sa']) / 2.0
    
    for i, (x_val, y_val) in enumerate(zip(x_grid, y_grid)):
        if y_val <= p['a']:
            mu_vals[i] = 0.0
        elif y_val >= p['b']:
            if p['is_supercritical'] and y_val <= p['x_max']:
                omega = 1.0 / (beta * p['rho'] * p['kappa'] * y_val)
            else:
                omega = 0.0
            mu_vals[i] = p['pref_const'] * (x_val**(p['pref_const'] - 1)) * omega
        else:
            def obj(vars_arr):
                val = J_map(vars_arr[0] + 1j*vars_arr[1], p['c0'], p['c1'], p['nu'])
                return [np.real(val) - y_val, np.imag(val)]
            
            t = np.clip((y_val - p['a']) / (p['b'] - p['a']), 0.01, 0.99) 
            angle = np.pi * t if p['a'] < p['b'] else np.pi * (1 - t)
            s_guess = mid + R * np.exp(1j * angle)
            
            res = least_squares(obj, [np.real(s_guess), np.imag(s_guess)], 
                                bounds=([-np.inf, 1e-12], [np.inf, np.inf]),
                                ftol=1e-10, xtol=1e-10)
            
            I_p = res.x[0] + 1j*res.x[1]
            
            if p['is_omega_nu']:
                arg_s = np.abs(np.angle((p['s1'] - I_p) / (p['s2'] - I_p)))
            else:
                arg_s = np.abs(np.angle((p['s2'] - I_p) / (p['s1'] - I_p)))
                
            omega = (1.0 / (np.pi * beta * p['rho'] * p['kappa'] * y_val)) * arg_s
            mu_vals[i] = p['pref_const'] * (x_val**(p['pref_const'] - 1)) * max(0.0, omega)
            
    return np.nan_to_num(mu_vals, nan=0.0, posinf=0.0, neginf=0.0), p

# --- 2D GRID EXECUTION ---
configs = [
    {'gamma_sq': 0.3,  'eta': 7.0, 'theta': 1.0, 'beta': 0.5},
    {'gamma_sq': 0.7,  'eta': 3.0, 'theta': 3.0, 'beta': 0.2},
    {'gamma_sq': 0.4,  'eta': 2.0, 'theta': 5.0, 'beta': 1.0},
    {'gamma_sq': 0.25, 'eta': 3.0, 'theta': 2.0, 'beta': 1.5}
]

# EXTREMELY HIGH RESOLUTION (Expect a few minutes per plot)
xi_res = 450
x_res = 450

fig, axs = plt.subplots(2, 2, figsize=(20, 14))
axs = axs.flatten()

scale_factor = 3.0
true_ticks = [0, 0.5, 1, 1.5, 2, 3, 5, 10, 40]
transformed_locs = [1 - np.exp(-v / scale_factor) for v in true_ticks]

print("==================================================")
print(f" Generating High-Res Heatmaps ({xi_res}x{x_res})   ")
print("==================================================")

for idx, cfg in enumerate(configs):
    gamma_sq, eta, theta, beta = cfg['gamma_sq'], cfg['eta'], cfg['theta'], cfg['beta']
    
    print(f"\nComputing Grid {idx+1}/4: gamma^2={gamma_sq}, eta={eta}, theta={theta}, beta={beta}")
    
    boundary = 1 - gamma_sq
    xi_vals = np.linspace(-gamma_sq + 0.005, 0.995, xi_res)
    
    # Exclude points critically close to the transitions to avoid numerical noise
    xi_vals = xi_vals[np.abs(xi_vals - boundary) > 0.005]
    xi_vals = xi_vals[np.abs(xi_vals) > 0.005]
    
    x_vals = np.linspace(0.005, 0.995, x_res)
    X_grid, XI_grid = np.meshgrid(x_vals, xi_vals)
    Z_grid = np.zeros_like(X_grid)
    
    # Arrays to track the trajectories of the edges
    x_b_traj = np.zeros(len(xi_vals))
    x_max_traj = np.zeros(len(xi_vals))
    
    for i, xi in enumerate(xi_vals):
        if i % 50 == 0:
            print(f"  -> Progress: row {i}/{len(xi_vals)}")
            
        mu_vals, p = compute_density_row(x_vals, xi, gamma_sq, eta, theta, beta)
        Z_grid[i, :] = mu_vals
        
        # Track the physical coordinate of the theoretical edge J(s_b) and the constraint
        if p is not None:
            x_b_traj[i] = min(max(0.0, p['b']) ** (1.0 / p['pref_pow']),1)
            x_max_traj[i] = min(max(0.0, p['x_max']) ** (1.0 / p['pref_pow']),1)
        else:
            x_b_traj[i] = np.nan
            x_max_traj[i] = np.nan
            
    Z_transformed = 1 - np.exp(-Z_grid / scale_factor)
    
    ax = axs[idx]
    mesh = ax.pcolormesh(XI_grid, X_grid, Z_transformed, cmap='jet', shading='auto', vmin=0, vmax=1)
    
    # Plot the trajectories
    ax.plot(xi_vals,x_b_traj, color="#3FF806", linewidth=2,linestyle='--', label=r'Arctic Curve')
    ax.plot(xi_vals, x_max_traj, color='#00FFFF', linewidth=1.5, linestyle='-.', label=r'Constraint $x_{\max}$')
    
    # Overlay regime transitions
    ax.axvline(x=boundary, color='white', linestyle='--', alpha=0.8, label=r'Phase Transitions')
    ax.axvline(x=0, color='white', linestyle='--', alpha=0.8)
    
    # Add INDIVIDUAL colorbar to the specific axis
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks(transformed_locs)
    cbar.set_ticklabels([str(v) for v in true_ticks])
    cbar.set_label(r'$\mu(x)$', rotation=0, labelpad=15, fontsize=20)
    
    ax.set_title(r'$\beta=%.1f$, $\gamma^2=%.2f$, $\eta=%.1f$, $\theta=%.1f$' % (beta, gamma_sq, eta, theta), fontsize=24)
    if idx > 1:
      ax.set_xlabel(r'$\xi$', fontsize =20)
    ax.set_ylabel(r'$x$', fontsize=20, rotation=0, labelpad=15)
    ax.set_xlim(-gamma_sq, 1.0)
    ax.set_ylim(0, 1)
    
    # Position legend dynamically based on the shape of the curves to avoid covering the mass
    leg_loc = 'lower right'
    ax.legend(loc=leg_loc, fontsize=18, framealpha=0.9)

# Adjust layout to accommodate the 4 independent colorbars 
plt.tight_layout()
file_name = "constrained_heatmaps_overlay_grid.pdf"
plt.savefig(file_name, bbox_inches='tight', dpi=300)
print(f"\nSuccess! High-resolution heatmap with edge trajectories saved as: {file_name}")
plt.show()