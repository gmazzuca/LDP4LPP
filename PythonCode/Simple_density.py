import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.integrate import simpson
import warnings

# Suppress runtime warnings from the optimizer
warnings.filterwarnings('ignore')

# --- CONFIGURATION FOR LATEX/PAPER PLOTTING ---
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'lines.linewidth': 2.5
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
            regime_name = r"$\xi \in (-\gamma^2, 0]$"
        elif 0 < xi <= 1 - gamma_sq:
            kappa = gamma_sq
            m1, n1, n2 = (theta * xi) / (eta * kappa), 0.0, 1.0 - gamma_sq - xi
            alpha, rho = theta, eta
            is_omega_nu = True
            regime_name = r"$\xi \in (0, 1-\gamma^2]$"
        elif 1 - gamma_sq < xi < 1:
            kappa = 1.0 - xi
            m1, n1, n2 = (theta * xi) / (eta * kappa), 1.0 - gamma_sq - xi, 0.0
            alpha, rho = eta, eta
            is_omega_nu = False
            regime_name = r"$\xi \in (1-\gamma^2, 1]$"
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
            regime_name = r"$\xi \in (-\gamma^2, 0]$"
        elif 0 < xi <= 1 - gamma_sq:
            kappa = gamma_sq
            m1, n1, n2 = xi / kappa, 0.0, 1.0 - gamma_sq - xi
            alpha, rho = theta, theta
            is_omega_nu = False
            regime_name = r"$\xi \in (0, 1-\gamma^2]$"
        elif 1 - gamma_sq < xi < 1:
            kappa = 1.0 - xi
            m1, n1, n2 = xi / kappa, 1.0 - gamma_sq - xi, 0.0
            alpha, rho = eta, theta
            is_omega_nu = True
            regime_name = r"$\xi \in (1-\gamma^2, 1]$"
        else:
            raise ValueError(f"xi={xi} out of bounds")
            
    return nu, kappa, m1, n1, n2, alpha, rho, is_omega_nu, prefactor_pow, prefactor_const, regime_name

def solve_moduli(xi, gamma_sq, eta, theta, beta):
    """Computes explicit parameters using the precise geometric analytical collapse."""
    nu, kappa, m1, n1, n2, alpha, rho, is_omega_nu, pref_pow, pref_const, regime_name = get_regime_params(xi, gamma_sq, eta, theta, beta)
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
        'pref_pow': pref_pow, 'pref_const': pref_const,
        'm1': m1, 'n1': n1, 'n2': n2, 'regime_name': regime_name
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


# --- CONFIGURATION FOR THE 4 PLOTS ---
configs = [
    {'gamma_sq': 0.4, 'eta': 2.0, 'theta': 5.0, 'beta': 1.0, 'xi_vals': [-0.1, 0.2, 0.85]},
    {'gamma_sq': 0.3,  'eta': 1.0, 'theta': 5.0, 'beta': 0.5, 'xi_vals': [-0.1, 0.2, 0.85]},
    {'gamma_sq': 0.5,  'eta': 2.0, 'theta': 3.0, 'beta': 0.2, 'xi_vals': [-0.3, 0.0, 0.75]},
    {'gamma_sq': 0.3,  'eta': 2.0, 'theta': 5.0, 'beta': 0.5, 'xi_vals': [-0.2, 0.01, 0.8]}
]

x_grid = np.linspace(0.0001, 0.9999, 1000) 
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

fig, axs = plt.subplots(2, 2, figsize=(20, 14))
axs = axs.flatten()

print(f"==================================================")
print(f"      COMPREHENSIVE NUMERICAL EVALUATION          ")
print(f"==================================================\n")

for i, cfg in enumerate(configs):
    ax = axs[i]
    gamma_sq, eta, theta, beta = cfg['gamma_sq'], cfg['eta'], cfg['theta'], cfg['beta']
    xi_values_to_plot = cfg['xi_vals']
    
    print(f"==================================================")
    print(f" Configuration {i+1}: gamma^2={gamma_sq}, eta={eta}, theta={theta}, beta={beta}")
    print(f"==================================================")
    
    for idx, xi in enumerate(xi_values_to_plot):
        mu_vals, p = compute_density_row(x_grid, xi, gamma_sq, eta, theta, beta)
        
        if p is None: 
            continue
            
        line, = ax.plot(x_grid, mu_vals, color=colors[idx], 
                        label=f'$\\xi={xi}$ | {p["regime_name"]}')
        ax.fill_between(x_grid, mu_vals, color=line.get_color(), alpha=0.15)
        
        # Plot the Upper Constraint Limit 1/(beta * kappa * x)
        upper_bound = 1.0 / (beta * p['kappa'] * x_grid)
        ax.plot(x_grid, upper_bound, linewidth=1.5, linestyle='--', color=line.get_color(), alpha=0.7)
        
        # Calculate Integral
        integral_val = simpson(mu_vals, x=x_grid)
        
        # --- PRINTS THE PARAMETERS TO CONSOLE (FIXED) ---
        print(f"--- Results for xi = {xi} ---")
        print(f"  Regime:          {p['regime_name']}")
        print(f"  Status:          {'[SUPERCRITICAL]' if p['is_supercritical'] else '[SUBCRITICAL]'}")
        print(f"  Constants:       nu={p['nu']:.4f}, kappa={p['kappa']:.4f}, alpha={p['alpha']:.4f}, rho={p['rho']:.4f}")
        print(f"  Model Params:    m1={p['m1']:.4f}, n1={p['n1']:.4f}, n2={p['n2']:.4f}")
        print(f"  Moduli:          c0={p['c0']:.4f}, c1={p['c1']:.4f}")
        print(f"  Spectral Roots:  s1={p['s1']:.4f}, s2={p['s2']:.4f}")
        print(f"  Bounds in y:     a={p['a']:.4f}, b={p['b']:.4f}, x_max={p['x_max']:.4f}")
        print(f"  Integral of mu:  {integral_val:.4f}\n")
        
    ax.set_title(f'$\\beta={beta}$, $\\gamma^2={gamma_sq}$, $\\eta={eta}$, $\\theta={theta}$')
    ax.set_xlabel('$x$')
    if i % 2 == 0:
        ax.set_ylabel('$\\mu(x)$', rotation=0, labelpad=25)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 15)  
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

plt.tight_layout()

# Save the figure as a PDF for LaTeX inclusion
file_name = "density_plots_grid.pdf"
plt.savefig(file_name, bbox_inches='tight', dpi=100)
print(f"Success! The high-resolution plot has been saved in the local folder as: {file_name}")

plt.show()