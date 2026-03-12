import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, fsolve
from scipy.integrate import simpson
import warnings

# Suppress runtime warnings from the optimizer during bounded searches
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
    """Maps physical constants exactly to the cleaned Corollary 1.4 parameters."""
    if theta > eta:
        nu = theta / eta
        prefactor_pow, prefactor_const = eta, eta
        if -gamma_sq < xi <= 0:
            kappa = gamma_sq - abs(xi)
            m1 = abs(xi) / kappa
            n1, n2 = -abs(xi), 1.0 - gamma_sq
            alpha, rho = theta, eta
            is_omega_nu = True
            regime_name = r"$\xi \in (-\gamma^2, 0]$"
        elif 0 < xi <= 1 - gamma_sq:
            kappa = gamma_sq
            m1 = (theta * xi) / (eta * kappa)
            n1, n2 = 0.0, 1.0 - gamma_sq - xi
            alpha, rho = theta, eta
            is_omega_nu = True
            regime_name = r"$\xi \in (0, 1-\gamma^2]$"
        elif 1 - gamma_sq < xi < 1:
            kappa = 1.0 - xi
            m1 = (theta * xi) / (eta * kappa)
            n1, n2 = 1.0 - gamma_sq - xi, 0.0
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
            m1 = (abs(xi) * eta) / (kappa * theta)
            n1, n2 = -abs(xi), 1.0 - gamma_sq
            alpha, rho = theta, theta
            is_omega_nu = False
            regime_name = r"$\xi \in (-\gamma^2, 0]$"
        elif 0 < xi <= 1 - gamma_sq:
            kappa = gamma_sq
            m1 = xi / kappa
            n1, n2 = 0.0, 1.0 - gamma_sq - xi
            alpha, rho = theta, theta
            is_omega_nu = False
            regime_name = r"$\xi \in (0, 1-\gamma^2]$"
        elif 1 - gamma_sq < xi < 1:
            kappa = 1.0 - xi
            m1 = xi / kappa
            n1, n2 = 1.0 - gamma_sq - xi, 0.0
            alpha, rho = eta, theta
            is_omega_nu = True
            regime_name = r"$\xi \in (1-\gamma^2, 1]$"
        else:
            raise ValueError(f"xi={xi} out of bounds")
            
    return nu, kappa, m1, n1, n2, alpha, rho, is_omega_nu, prefactor_pow, prefactor_const, regime_name

def solve_moduli(xi, gamma_sq, eta, theta, beta):
    """Computes explicit parameters or precisely solves the stabilized nonlinear system."""
    nu, kappa, m1, n1, n2, alpha, rho, is_omega_nu, pref_pow, pref_const, regime_name = get_regime_params(xi, gamma_sq, eta, theta, beta)
    x_max = np.exp(-rho * beta * (gamma_sq - kappa))
    
    # Base subcritical explicit solutions
    if is_omega_nu:
        A = np.exp((alpha * beta * kappa / nu) * (nu + 1 + m1 + (nu / kappa) * (n2 - n1)))
        B = np.exp((alpha * beta * kappa / nu) * (1 + m1))
        s1 = A * (B - 1) / (A - B) 
        s2 = (B - 1) / (A - B)
        K1_target = np.exp(n1 * alpha * beta / nu)
        K2_target = np.exp(n2 * alpha * beta / nu)
        K1 = K1_target * ((A * (B - 1)) / (B * (A - 1)))**(1 / nu)
        K2 = K2_target * ((B - 1) / (A - 1))**(1 / nu)
    else:
        A = np.exp(alpha * beta * kappa * (nu + 1 + m1 + (nu / kappa) * (n2 - n1)))
        B = np.exp(alpha * beta * kappa * (1 + m1))
        s1 = A * (B - 1) / (A - B)
        s2 = (B - 1) / (A - B)
        K1_target = np.exp(n1 * alpha * beta)
        K2_target = np.exp(n2 * alpha * beta)
        # THE FIX: Denominator corrected to (A - 1)
        K1 = K1_target * ((A * (B - 1)) / (B * (A - 1)))**(1 / nu)
        K2 = K2_target * ((B - 1) / (A - 1))**(1 / nu)
        
    c1 = (K1 - K2) / (s1 - s2)
    c0 = (K2 * s1 - K1 * s2) / (s1 - s2)
    
    discriminant = np.maximum(0, 4*c0*c1*nu + (c1**2)*((nu-1)**2))
    sa = -(nu-1)/(2*nu) - np.sqrt(discriminant)/(2*nu*c1)
    sb = -(nu-1)/(2*nu) + np.sqrt(discriminant)/(2*nu*c1)
    
    J_sa = np.real(J_map(sa+0j, c0, c1, nu))
    J_sb = np.real(J_map(sb+0j, c0, c1, nu))
    a, b = min(J_sa, J_sb), max(J_sa, J_sb)
    
    is_supercritical = (b > x_max)
    q1, q2 = None, None
    
    if is_supercritical:
        # Unified Exp constants leveraging the generalized definitions
        E5 = np.exp(rho * kappa * beta * (nu + 1 + m1 + (nu / kappa) * (n2 - n1)))
        E6 = np.exp(rho * kappa * beta * (1 + m1))
        
        def sys_eqs(vars_array):
            c0_v, c1_v, d1_v, d2_v, d3_v, d4_v = vars_array
            
            disc = np.maximum(0, 4*c0_v*c1_v*nu + (c1_v**2)*((nu-1)**2))
            sb_v = -(nu-1)/(2*nu) + np.sqrt(disc)/(2*nu*c1_v)
            
            # Independent parameterization to smoothly allow s1 = q2
            q1_v = sb_v * (1.0 + d1_v)
            q2_v = sb_v * (1.0 - d2_v)
            s1_v = sb_v * (1.0 - d3_v) 
            s2_v = sb_v * (1.0 - d4_v) 
            
            eq1 = np.real(J_map(s1_v+0j, c0_v, c1_v, nu)) - K1_target
            eq2 = np.real(J_map(s2_v+0j, c0_v, c1_v, nu)) - K2_target
            eq3 = np.real(J_map(q1_v+0j, c0_v, c1_v, nu)) - x_max
            eq4 = np.real(J_map(q2_v+0j, c0_v, c1_v, nu)) - x_max
            eq5 = s1_v * q1_v - E5 * s2_v * q2_v
            eq6 = (s1_v + 1) * (q1_v + 1) - E6 * (s2_v + 1) * (q2_v + 1)
            
            return [eq1, eq2, eq3, eq4, eq5, eq6]

        # Use safe interior subcritical mappings as initial guesses
        d1_guess, d2_guess = 0.2, 0.2
        s1_safe = np.clip(s1, sb * 0.1, sb * 0.99)
        s2_safe = np.clip(s2, sb * 0.05, sb * 0.95)
        d3_guess = 1.0 - (s1_safe / sb)
        d4_guess = 1.0 - (s2_safe / sb)

        initial_guess = [c0, c1, d1_guess, d2_guess, d3_guess, d4_guess]
        
        # d1 > 0 guarantees q1 > sb. d2, d3, d4 in (0, 1) guarantees roots stay in interior (0, sb)
        bounds = (
            [-np.inf, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6], 
            [np.inf, np.inf, np.inf, 1.0 - 1e-6, 1.0 - 1e-6, 1.0 - 1e-6]
        )
        
        res = least_squares(sys_eqs, initial_guess, bounds=bounds, method='trf', 
                            ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=5000)
        
        c0, c1, d1, d2, d3, d4 = res.x
        
        discriminant = np.maximum(0, 4*c0*c1*nu + (c1**2)*((nu-1)**2))
        sa = -(nu-1)/(2*nu) - np.sqrt(discriminant)/(2*nu*c1)
        sb = -(nu-1)/(2*nu) + np.sqrt(discriminant)/(2*nu*c1)
        
        q1 = sb * (1.0 + d1)
        q2 = sb * (1.0 - d2)
        s1 = sb * (1.0 - d3)
        s2 = sb * (1.0 - d4)
        
        J_sa = np.real(J_map(sa+0j, c0, c1, nu))
        J_sb = np.real(J_map(sb+0j, c0, c1, nu))
        a, b = min(J_sa, J_sb), max(J_sa, J_sb)
        
    return {
        'nu': nu, 'kappa': kappa, 'alpha': alpha, 'rho': rho,
        'm1': m1, 'n1': n1, 'n2': n2,
        'is_supercritical': is_supercritical, 'x_max': x_max,
        'c0': c0, 'c1': c1, 's1': s1, 's2': s2, 'q1': q1, 'q2': q2,
        'sa': sa, 'sb': sb, 'J_sa': J_sa, 'J_sb': J_sb, 
        'a': a, 'b': b, 'pref_pow': pref_pow, 'pref_const': pref_const,
        'regime_name': regime_name
    }

def compute_density_row(x_grid, xi, gamma_sq, eta, theta, beta):
    try:
        p = solve_moduli(xi, gamma_sq, eta, theta, beta)
    except Exception as e:
        print(f"Error computing moduli: {e}")
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
            angle = np.pi * (1 - t) if (p['sa'] < p['sb']) ^ (p['J_sa'] > p['J_sb']) else np.pi * t
            s_guess = mid + R * np.exp(1j * angle)
            
            res = least_squares(obj, [np.real(s_guess), np.imag(s_guess)], 
                                bounds=([-np.inf, 1e-12], [np.inf, np.inf]),
                                ftol=1e-11, xtol=1e-11)
            
            I_p = res.x[0] + 1j*res.x[1]
            
            # The angles are safely separated via Absolute values exactly as established
            arg_s = np.abs(np.angle((p['s2'] - I_p) / (p['s1'] - I_p)))
            if p['is_supercritical']:
                arg_q = np.abs(np.angle((p['q1'] - I_p) / (p['q2'] - I_p)))
                omega = (1.0 / (np.pi * beta * p['rho'] * p['kappa'] * y_val)) * (arg_q + arg_s)
            else:
                omega = (1.0 / (np.pi * beta * p['rho'] * p['kappa'] * y_val)) * arg_s
                
            mu_vals[i] = p['pref_const'] * (x_val**(p['pref_const'] - 1)) * omega
            
    return np.nan_to_num(mu_vals, nan=0.0, posinf=0.0, neginf=0.0), p


# --- CONFIGURATION FOR THE 4 PLOTS ---
configs = [
    {'gamma_sq': 0.25, 'eta': 1.0, 'theta': 2.0, 'beta': 1.0, 'xi_vals': [-0.1, 0.2, 0.85]},
    {'gamma_sq': 0.3,  'eta': 1.0, 'theta': 5.0, 'beta': 0.5, 'xi_vals': [-0.1, 0.2, 0.85]},
    {'gamma_sq': 0.5,  'eta': 2.0, 'theta': 3.0, 'beta': 0.2, 'xi_vals': [-0.3, 0.0, 0.75]},
    {'gamma_sq': 0.3,  'eta': 2.0, 'theta': 5.0, 'beta': 0.5, 'xi_vals': [-0.2, 0.01, 0.8]}
]

# Heavy grid spacing ensures no numerical error when running Simpson's Rule
x_grid = np.linspace(0.0001, 0.9999, 1500) 
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

fig, axs = plt.subplots(2, 2, figsize=(16, 12))
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
        
        upper_bound = 1.0 / (beta * p['kappa'] * x_grid)
        ax.plot(x_grid, upper_bound, linewidth=1.5, linestyle='--', color=line.get_color(), alpha=0.7)
        
        # Calculate Accurate Integral
        integral_val = simpson(mu_vals, x=x_grid)
        
        # --- PRINTS THE PARAMETERS TO CONSOLE ---
        print(f"--- Results for xi = {xi} ---")
        print(f"  Regime:          {p['regime_name']}")
        print(f"  Status:          {'[SUPERCRITICAL]' if p['is_supercritical'] else '[SUBCRITICAL]'}")
        print(f"  Constants:       nu={p['nu']:.4f}, kappa={p['kappa']:.4f}, alpha={p['alpha']:.4f}, rho={p['rho']:.4f}")
        print(f"  Model Params:    m1={p['m1']:.4f}, n1={p['n1']:.4f}, n2={p['n2']:.4f}")
        print(f"  Moduli:          c0={p['c0']:.4f}, c1={p['c1']:.4f}")
        print(f"  Spectral Roots:  s1={p['s1']:.4f}, s2={p['s2']:.4f}")
        if p['is_supercritical']:
            print(f"  Saturated Roots: q1={p['q1']:.4f}, q2={p['q2']:.4f}")
        print(f"  Bounds in y:     a={p['a']:.4f}, b={p['b']:.4f}, x_max={p['x_max']:.4f}")
        print(f"  Integral of mu:  {integral_val:.4f}\n")
        
    ax.set_title(f'$\\beta={beta}$, $\\gamma^2={gamma_sq}$, $\\eta={eta}$, $\\theta={theta}$')
    ax.set_xlabel('$x$')
    if i % 2 == 0:
        ax.set_ylabel('$\\mu(x)$', rotation=0, labelpad=25)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 15)  
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

plt.tight_layout()

# Save the figure as a PDF for LaTeX inclusion
file_name = "density_plots_grid.pdf"
plt.savefig(file_name, bbox_inches='tight', dpi=300)
print(f"Success! The high-resolution plot has been saved in the local folder as: {file_name}")

plt.show()