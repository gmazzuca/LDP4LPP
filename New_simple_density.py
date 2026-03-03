import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.integrate import simpson
import warnings

warnings.filterwarnings('ignore')

# Default parameters
gamma_sq = 0.25
eta = 1.0
theta = 2.0
beta = 0.5  # Inverse temperature (beta > 0)

def J_map(s, c0, c1, nu):
    """Conformal mapping J_{c0, c1}(s)."""
    return (c1 * s + c0) * ((s + 1) / s)**(1 / nu)

def get_regime_params(xi, gamma_sq, eta, theta, beta):
    """Maps physical constants to the model problem constants based on Corollary 3.4."""
    if theta > eta:
        nu = theta / eta
        prefactor_pow, prefactor_const = eta, eta
        if -gamma_sq < xi <= 0:
            kappa = gamma_sq - abs(xi)
            m1, m2 = 1.0/kappa, abs(xi)/kappa
            n1, n2 = -abs(xi), 1.0 - gamma_sq
            alpha, delta, rho = theta, eta*kappa, eta
            is_omega_nu = True
            regime_name = r"$\xi \in (-\gamma^2, 0]$"
        elif 0 < xi <= 1 - gamma_sq:
            kappa = gamma_sq
            m1, m2 = 1.0/kappa, (theta*xi)/(eta*kappa)
            n1, n2 = 0.0, 1.0 - gamma_sq - xi
            alpha, delta, rho = theta, eta*kappa, eta
            is_omega_nu = True
            regime_name = r"$\xi \in (0, 1-\gamma^2]$"
        elif 1 - gamma_sq < xi < 1:
            kappa = 1.0 - xi
            m1, m2 = 1.0/kappa, (theta*xi)/(eta*kappa)
            n1, n2 = 1.0 - gamma_sq - xi, 0.0
            alpha, delta, rho = eta, eta*kappa, eta
            is_omega_nu = False
            regime_name = r"$\xi \in (1-\gamma^2, 1]$"
        else:
            raise ValueError("xi out of bounds")
    else:
        nu = eta / theta
        prefactor_pow, prefactor_const = theta, theta
        if -gamma_sq < xi <= 0:
            kappa = gamma_sq - abs(xi)
            m1, m2 = 1.0/kappa, (abs(xi)*eta)/(kappa*theta)
            n1, n2 = -abs(xi), 1.0 - gamma_sq
            alpha, delta, rho = theta, theta*kappa, theta
            is_omega_nu = False
            regime_name = r"$\xi \in (-\gamma^2, 0]$"
        elif 0 < xi <= 1 - gamma_sq:
            kappa = gamma_sq
            m1, m2 = 1.0/kappa, xi/kappa
            n1, n2 = 0.0, 1.0 - gamma_sq - xi
            alpha, delta, rho = theta, theta*kappa, theta
            is_omega_nu = False
            regime_name = r"$\xi \in (0, 1-\gamma^2]$"
        elif 1 - gamma_sq < xi < 1:
            kappa = 1.0 - xi
            m1, m2 = 1.0/kappa, xi/kappa
            n1, n2 = 1.0 - gamma_sq - xi, 0.0
            alpha, delta, rho = eta, theta*kappa, theta
            is_omega_nu = True
            regime_name = r"$\xi \in (1-\gamma^2, 1]$"
        else:
            raise ValueError("xi out of bounds")
            
    return nu, kappa, m1, m2, n1, n2, alpha, delta, rho, is_omega_nu, prefactor_pow, prefactor_const, regime_name

def solve_moduli(xi, gamma_sq, eta, theta, beta):
    """Computes explicit parameters or precisely solves the stabilized 6x6 nonlinear system."""
    nu, kappa, m1, m2, n1, n2, alpha, delta, rho, is_omega_nu, pref_pow, pref_const, regime_name = get_regime_params(xi, gamma_sq, eta, theta, beta)
    x_max = np.exp(-rho * beta * (gamma_sq - kappa))
    
    # Base subcritical explicit solutions
    if is_omega_nu:
        A = np.exp((alpha*beta/(m1*nu)) * (nu + 1 + m2 + m1*nu*(n2-n1)))
        B = np.exp((alpha*beta/(m1*nu)) * (1 + m2))
        s1 = A * (B - 1) / (A - 1)
        s2 = (B - 1) / (A - 1)
        K1_target = np.exp(n1*alpha*beta/nu)
        K2_target = np.exp(n2*alpha*beta/nu)
        K1 = (np.exp(n1*alpha*beta) * (A*(B - 1))/(B*(A - 1)))**(1/nu)
        K2 = (np.exp(n2*alpha*beta) * (B - 1)/(A - 1))**(1/nu)
    else:
        A = np.exp((alpha*beta/m1) * (nu + 1 + m2 + m1*nu*(n2-n1)))
        B = np.exp((alpha*beta/m1) * (1 + m2))
        s1 = A * (B - 1) / (A - B)
        s2 = (B - 1) / (A - B)
        K1_target = np.exp(n1*alpha*beta)
        K2_target = np.exp(n2*alpha*beta)
        K1 = np.exp(n1*alpha*beta) * ((A*(B - 1))/(B*(A - B)))**(1/nu)
        K2 = np.exp(n2*alpha*beta) * ((B - 1)/(A - B))**(1/nu)
        
    c1 = (K1 - K2) / (s1 - s2)
    c0 = (K2*s1 - K1*s2) / (s1 - s2)
    
    discriminant = 4*c0*c1*nu + (c1**2)*((nu-1)**2)
    sa = -(nu-1)/(2*nu) - np.sqrt(discriminant)/(2*nu*c1)
    sb = -(nu-1)/(2*nu) + np.sqrt(discriminant)/(2*nu*c1)
    
    J_sa = np.real(J_map(sa+0j, c0, c1, nu))
    J_sb = np.real(J_map(sb+0j, c0, c1, nu))
    a, b = min(J_sa, J_sb), max(J_sa, J_sb)
    
    is_supercritical = (b > x_max)
    q1, q2 = None, None
    
    if is_supercritical:
        E5 = np.exp(delta * beta * (nu + 1 + m2 + m1 * nu * (n2 - n1)))
        E6 = np.exp(delta * beta * (1 + m2))
        
        def sys_eqs(vars_array):
            c0_v, c1_v, s1_v, s2_v, q1_v, q2_v = vars_array
            eq1 = np.real(J_map(s1_v+0j, c0_v, c1_v, nu)) - K1_target
            eq2 = np.real(J_map(s2_v+0j, c0_v, c1_v, nu)) - K2_target
            eq3 = np.real(J_map(q1_v+0j, c0_v, c1_v, nu)) - x_max
            eq4 = np.real(J_map(q2_v+0j, c0_v, c1_v, nu)) - x_max
            eq5 = s1_v * q1_v - E5 * s2_v * q2_v
            eq6 = (s1_v + 1) * (q1_v + 1) - E6 * (s2_v + 1) * (q2_v + 1)
            return [eq1, eq2, eq3, eq4, eq5, eq6]

        initial_guess = [c0, c1, s1, s2, max(sa, sb) * 1.2, max(sa, sb) * 0.8]
        bounds = ([-np.inf, -np.inf, 1e-6, 1e-6, 1e-6, 1e-6], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        
        res = least_squares(sys_eqs, initial_guess, bounds=bounds, method='trf', 
                            ftol=1e-12, xtol=1e-12, gtol=1e-12)
        
        c0, c1, s1, s2, q1, q2 = res.x
        
        discriminant = 4*c0*c1*nu + (c1**2)*((nu-1)**2)
        sa = -(nu-1)/(2*nu) - np.sqrt(discriminant)/(2*nu*c1)
        sb = -(nu-1)/(2*nu) + np.sqrt(discriminant)/(2*nu*c1)
        
        J_sa = np.real(J_map(sa+0j, c0, c1, nu))
        J_sb = np.real(J_map(sb+0j, c0, c1, nu))
        a, b = min(J_sa, J_sb), max(J_sa, J_sb)
        
    return {
        'nu': nu, 'delta': delta, 'kappa': kappa, 'is_supercritical': is_supercritical, 'x_max': x_max,
        'c0': c0, 'c1': c1, 's1': s1, 's2': s2, 'q1': q1, 'q2': q2,
        'sa': sa, 'sb': sb, 'J_sa': J_sa, 'J_sb': J_sb, 
        'a': a, 'b': b, 'pref_pow': pref_pow, 'pref_const': pref_const,
        'regime_name': regime_name
    }

def compute_density_row(x_grid, xi, gamma_sq, eta, theta, beta):
    """Computes the density using constrained least_squares steering to eliminate scattering."""
    try:
        p = solve_moduli(xi, gamma_sq, eta, theta, beta)
    except Exception:
        return np.zeros_like(x_grid), "Error", None
        
    y_grid = x_grid ** p['pref_pow']
    mu_vals = np.zeros_like(x_grid)
    
    mid = (p['sa'] + p['sb']) / 2.0
    R = abs(p['sb'] - p['sa']) / 2.0
    
    for i, (x_val, y_val) in enumerate(zip(x_grid, y_grid)):
        if y_val <= p['a']:
            mu_vals[i] = 0.0
        elif y_val >= p['b']:
            if p['is_supercritical'] and y_val <= p['x_max']:
                omega = 1.0 / (beta * p['delta'] * y_val)
            else:
                omega = 0.0
            mu_vals[i] = p['pref_const'] * (x_val**(p['pref_const'] - 1)) * omega
        else:
            def obj(vars_arr):
                val = J_map(vars_arr[0] + 1j*vars_arr[1], p['c0'], p['c1'], p['nu'])
                return [np.real(val) - y_val, np.imag(val)]
            
            # Geometric tracker
            t = np.clip((y_val - p['a']) / (p['b'] - p['a']), 0.01, 0.99) 
            angle = np.pi * (1 - t) if (p['sa'] < p['sb']) ^ (p['J_sa'] > p['J_sb']) else np.pi * t
            s_guess = mid + R * np.exp(1j * angle)
            
            # --- THE BULLETPROOF BARRICADE ---
            # Bounds strictly block the solver from stepping into the lower half plane (v < 0).
            res = least_squares(obj, [np.real(s_guess), np.imag(s_guess)], 
                                bounds=([-np.inf, 1e-12], [np.inf, np.inf]),
                                ftol=1e-11, xtol=1e-11)
            
            I_p = res.x[0] + 1j*res.x[1]
            
            # Form the exact ratio defined in Theorem 3.9 (s2 is in numerator, s1 is in denominator)
            if p['is_supercritical']:
                ratio1 = ((p['q1'] - I_p) / (p['q2'] - I_p))
                ratio2 = ((p['s2'] - I_p) / (p['s1'] - I_p))
                arg_part = np.abs(np.angle(ratio1)) + np.abs(np.angle(ratio2))
            else:
                ratio = (p['s2'] - I_p) / (p['s1'] - I_p)
                arg_part = np.abs(np.angle(ratio))
            
            omega = (1.0 / (np.pi * beta * p['delta'] * y_val)) * arg_part
            mu_vals[i] = p['pref_const'] * (x_val**(p['pref_const'] - 1)) * omega
            
    return np.nan_to_num(mu_vals, nan=0.0, posinf=0.0, neginf=0.0), p['regime_name'], p['kappa']


# --- PLOTTING & INTEGRATION LOGIC ---

x_grid = np.linspace(0.0001, 0.9999, 1500) 
xi_values_to_plot = [-0.1, 0.3, 0.85] 

plt.figure(figsize=(12, 7))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

print(f"--- Numerical Integration Results (Beta={beta}) ---")
print("Expected mathematical integral mass: 1.000")

for idx, xi in enumerate(xi_values_to_plot):
    print(f"  -> Processing xi = {xi} ...")
    mu_vals, regime_name, kappa = compute_density_row(x_grid, xi, gamma_sq, eta, theta, beta)
    
    if kappa is None: continue
        
    line, = plt.plot(x_grid, mu_vals, linewidth=2.5, color=colors[idx], 
                     label=f'$\\xi={xi}$ | {regime_name}')
    plt.fill_between(x_grid, mu_vals, color=line.get_color(), alpha=0.15)
    
    upper_bound = 1.0 / (beta * kappa * x_grid)
    plt.plot(x_grid, upper_bound, linewidth=1.5, linestyle='--', color=line.get_color(), alpha=0.7)
    
    # Accurate Simpson integration over the dense grid
    integral_val = simpson(mu_vals, x=x_grid)
    print(f"     Integral of mu(x) dx: {integral_val:.4f}")

plt.title(f'Density $\\mu(x)$ and Upper Constraints (Dashed)\n$\\beta={beta}$, $\\gamma^2={gamma_sq}$, $\\eta={eta}$, $\\theta={theta}$', size=16)
plt.xlabel('$x$ (Position)', size=14)
plt.ylabel('$\\mu(x)$', rotation=0, labelpad=25, size=14)

plt.xlim(0, 1)
plt.ylim(0, 15)  
plt.legend(fontsize=14, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()