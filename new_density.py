import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, fsolve
import warnings

# Suppress runtime warnings from the optimizer
warnings.filterwarnings('ignore')

# Default parameters
gamma_sq = 0.1
eta = 1.0
theta = 2.0
beta = 0.05  # Inverse temperature (beta > 0)

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
        elif 0 < xi <= 1 - gamma_sq:
            kappa = gamma_sq
            m1, m2 = 1.0/kappa, (theta*xi)/(eta*kappa)
            n1, n2 = 0.0, 1.0 - gamma_sq - xi
            alpha, delta, rho = theta, eta*kappa, eta
            is_omega_nu = True
        elif 1 - gamma_sq < xi < 1:
            kappa = 1.0 - xi
            m1, m2 = 1.0/kappa, (theta*xi)/(eta*kappa)
            n1, n2 = 1.0 - gamma_sq - xi, 0.0
            alpha, delta, rho = eta, eta*kappa, eta
            is_omega_nu = False
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
        elif 0 < xi <= 1 - gamma_sq:
            kappa = gamma_sq
            m1, m2 = 1.0/kappa, xi/kappa
            n1, n2 = 0.0, 1.0 - gamma_sq - xi
            alpha, delta, rho = theta, theta*kappa, theta
            is_omega_nu = False
        elif 1 - gamma_sq < xi < 1:
            kappa = 1.0 - xi
            m1, m2 = 1.0/kappa, xi/kappa
            n1, n2 = 1.0 - gamma_sq - xi, 0.0
            alpha, delta, rho = eta, theta*kappa, theta
            is_omega_nu = True
        else:
            raise ValueError("xi out of bounds")
            
    return nu, kappa, m1, m2, n1, n2, alpha, delta, rho, is_omega_nu, prefactor_pow, prefactor_const

def solve_moduli(xi, gamma_sq, eta, theta, beta):
    """Computes the explicit parameters or solves the stabilized 6x6 nonlinear system."""
    nu, kappa, m1, m2, n1, n2, alpha, delta, rho, is_omega_nu, pref_pow, pref_const = get_regime_params(xi, gamma_sq, eta, theta, beta)
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
    
    a_raw, b_raw = np.real(J_map(sa+0j, c0, c1, nu)), np.real(J_map(sb+0j, c0, c1, nu))
    a, b = min(a_raw, b_raw), max(a_raw, b_raw)
    
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
        res = least_squares(sys_eqs, initial_guess, bounds=bounds, method='trf')
        c0, c1, s1, s2, q1, q2 = res.x
        
        discriminant = 4*c0*c1*nu + (c1**2)*((nu-1)**2)
        sa = -(nu-1)/(2*nu) - np.sqrt(discriminant)/(2*nu*c1)
        sb = -(nu-1)/(2*nu) + np.sqrt(discriminant)/(2*nu*c1)
        a_raw, b_raw = np.real(J_map(sa+0j, c0, c1, nu)), np.real(J_map(sb+0j, c0, c1, nu))
        a, b = min(a_raw, b_raw), max(a_raw, b_raw)
        
    return {
        'nu': nu, 'delta': delta, 'kappa': kappa, 'is_supercritical': is_supercritical, 'x_max': x_max,
        'c0': c0, 'c1': c1, 's1': s1, 's2': s2, 'q1': q1, 'q2': q2,
        'sa': sa, 'sb': sb, 'a': a, 'b': b, 'pref_pow': pref_pow, 'pref_const': pref_const
    }

def compute_density_row(x_grid, xi, gamma_sq, eta, theta, beta):
    """Computes an entire row of x values for a fixed xi efficiently."""
    try:
        p = solve_moduli(xi, gamma_sq, eta, theta, beta)
    except Exception:
        return np.zeros_like(x_grid)
        
    y_grid = x_grid ** p['pref_pow']
    mu_vals = np.zeros_like(x_grid)
    
    mid = (p['sa'] + p['sb']) / 2.0
    
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
            
            r, _, ier, _ = fsolve(obj, [mid, 0.1], full_output=True)
            if ier != 1:
                r = fsolve(obj, [mid, -0.1])
                
            I_p = r[0] + 1j*r[1]
            
            # Use the product trick to guarantee the angle never exceeds Pi, perfectly respecting the bounds
            if p['is_supercritical']:
                ratio = ((p['q1'] - I_p) / (p['q2'] - I_p)) * ((p['s1'] - I_p) / (p['s2'] - I_p))
            else:
                ratio = (p['s1'] - I_p) / (p['s2'] - I_p)
                
            arg_part = np.abs(np.angle(ratio))
            omega = (1.0 / (np.pi * beta * p['delta'] * y_val)) * arg_part
            mu_vals[i] = p['pref_const'] * (x_val**(p['pref_const'] - 1)) * omega
            
    return mu_vals


# --- HEATMAP COMPUTATION & PLOTTING LOGIC ---

# Grid parameters (Set to 300x300. Takes ~1 minute to evaluate 90k points)
xi_res = 500
x_res = 500

xi_vals = np.linspace(-gamma_sq + 0.01, 0.99, xi_res)
x_vals = np.linspace(0.001, 0.999, x_res)
boundary = 1 - gamma_sq

# Clean out phase boundaries to prevent divide-by-zero singularities in the plots
xi_vals = xi_vals[np.abs(xi_vals - boundary) > 0.02]
xi_vals = xi_vals[np.abs(xi_vals) > 0.02]

X_grid, XI_grid = np.meshgrid(x_vals, xi_vals)
Z_grid = np.zeros_like(X_grid)

print(f"Computing 2D Density Heatmap for beta = {beta} (this may take a couple of minutes)...")
for i in range(len(xi_vals)):
    if i % 25 == 0:
        print(f"  -> Processing time slice {i} / {len(xi_vals)}...")
    Z_grid[i, :] = compute_density_row(x_vals, xi_vals[i], gamma_sq, eta, theta, beta)

print("Rendering plot...")

# Transform the Z-axis for the colormap. 
# Because the density spikes near x=0 (singularity), a non-linear transformation 
# prevents the heatmap from being washed out by the extreme values.
scale_factor = 3.0
Z_transformed = 1 - np.exp(-Z_grid / scale_factor)

plt.figure(figsize=(10, 6))
mesh = plt.pcolormesh(XI_grid, X_grid, Z_transformed, cmap='jet', shading='auto', vmin=0, vmax=1)

# Format the colorbar to display the true numerical density values
true_ticks = [0, 0.5, 1, 1.5, 2, 3, 5, 10, 40]
transformed_locs = [1 - np.exp(-v / scale_factor) for v in true_ticks]

cbar = plt.colorbar(mesh)
cbar.set_ticks(transformed_locs)
cbar.set_ticklabels([str(v) for v in true_ticks])
cbar.set_label(r'$\mu(x)$', rotation=360, labelpad=15)

plt.xlabel(r'$\xi$ (Time Parameter)')
plt.ylabel(r'$x$ (Position)')
plt.title(r'Asymptotic Limit Shape of Particle System ($\beta=%.1f$, $\theta=%.1f$, $\eta=%.1f$, $\gamma^2=%.2f$)' % (beta, theta, eta, gamma_sq))

# Highlight the temporal phase boundaries
plt.axvline(x=boundary, color='white', linestyle='--', alpha=0.8, label=r'Transition $1-\gamma^2$')
plt.axvline(x=0, color='white', linestyle='--', alpha=0.8, label=r'Transition $0$')

plt.legend(loc='lower right')
plt.tight_layout()
plt.show()