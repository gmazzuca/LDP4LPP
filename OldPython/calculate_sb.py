import numpy as np
import matplotlib.pyplot as plt

def plot_sb_vs_beta(gamma, xi, nu):
    """
    Plots the critical point s_b as a function of beta.
    Automatically detects the regime (Case i, ii, iii) based on xi and gamma.
    """
    
    # 1. Detect Case and Set Parameters
    # We assume scaling parameter normalization:
    # For Cases i/ii (nu = theta/eta), we set eta=1 => theta=nu.
    # For Case iii (nu = eta/theta), we set theta=1 => eta=nu.
    
    case_label = ""
    
    # Case i: xi in (-gamma^2, 0]
    if -gamma**2 < xi <= 0:
        case_label = "Case i (xi <= 0)"
        kappa = gamma**2 - abs(xi)
        if kappa <= 0: return print("Error: Invalid xi (kappa <= 0)")
        
        # Scaling assumption: eta=1, theta=nu
        m1 = 1.0 / kappa
        m2 = abs(xi) / kappa
        n1 = gamma**2 - 1
        n2 = abs(xi)
        alpha = nu  # alpha = theta
        
    # Case ii: xi in (0, 1-gamma^2]
    elif 0 < xi <= (1 - gamma**2):
        case_label = "Case ii (0 < xi <= 1-gamma^2)"
        kappa = gamma**2
        
        # Scaling assumption: eta=1, theta=nu
        m1 = 1.0 / kappa
        m2 = (nu * xi) / kappa
        n1 = 0
        n2 = 1 - gamma**2 - xi
        alpha = nu # alpha = theta

    # Case iii: xi in (1-gamma^2, 1]
    elif (1 - gamma**2) < xi <= 1:
        case_label = "Case iii (xi > 1-gamma^2)"
        kappa = 1 - abs(xi)
        if kappa <= 0: return print("Error: Invalid xi (kappa <= 0)")
        
        # Scaling assumption: theta=1, eta=nu
        m1 = 1.0 / kappa
        m2 = xi / kappa
        n1 = 1 - gamma**2 - xi
        n2 = 0
        alpha = nu # alpha = eta

    else:
        print(f"Error: xi={xi} is outside the valid range based on gamma={gamma}")
        return

    print(f"[{case_label}] Detected.")
    print(f"Parameters: m1={m1:.2f}, m2={m2:.2f}, n1={n1:.2f}, n2={n2:.2f}, alpha={alpha:.2f}")

    # 2. Define Beta Range
    betas = np.linspace(0.01, 5.0, 500)
    sb_values = []
    valid_betas = []

    # 3. Calculate sb for each beta
    for beta in betas:
        try:
            # Calculate Exponents
            # Common denominator factor
            denom = m1 * nu
            
            # Exponent A: alpha*beta*(m2-1) / (m1*nu)
            exp_A = (alpha * beta * (m2 - 1)) / denom
            
            # Exponent B: alpha*beta*(m2 - 1 - nu + m1*nu*(n2-n1)) / (m1*nu)
            exp_B = (alpha * beta * (m2 - 1 - nu + denom * (n2 - n1))) / denom
            
            # Exponent Q = -alpha*beta / (m1*nu)
            exp_Q = -(alpha * beta) / denom

            A = np.exp(exp_A)
            B = np.exp(exp_B)
            Q = np.exp(exp_Q)
            
            # Avoid division by zero singularities
            if np.isclose(B, A) or np.isclose(Q, 1):
                continue

            # Calculate Ratio c0/c1 = s2 * (B-Q)/(Q-1)
            # where s2 = (A-1)/(B-A)
            s2 = (A - 1) / (B - A)
            ratio = s2 * (B - Q) / (Q - 1)
            
            # Calculate sb
            # sb = 1/(2nu) * [ -(nu-1) + sqrt((nu-1)^2 + 4*nu*ratio) ]
            discriminant = (nu - 1)**2 + 4 * nu * ratio
            
            if discriminant >= 0:
                sb = (1 / (2 * nu)) * (-(nu - 1) + np.sqrt(discriminant))
                sb_values.append(sb)
                valid_betas.append(beta)
                
        except Exception:
            continue

    # 4. Plotting
    if not sb_values:
        print("No valid solution points found in this range.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(valid_betas, sb_values, linewidth=2.5, color='darkblue', label=r"$s_b(\beta)$")
    
    plt.title(f"Critical Point $s_b$ vs Inverse Temperature $\\beta$\nCase: {case_label}", fontsize=14)
    plt.xlabel(r"$\beta$", fontsize=14)
    plt.ylabel(r"$s_b$", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    # Add parameter text box
    textstr = '\n'.join((
        r'$\gamma=%.2f$' % (gamma, ),
        r'$\xi=%.2f$' % (xi, ),
        r'$\nu=%.2f$' % (nu, )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    plt.show()

# ==========================================
# ENTER YOUR PARAMETERS HERE
# ==========================================

gamma_input = 0.7   # 0 < gamma < 1
xi_input    = 0.5   # Try varying this to switch cases (e.g., -0.1, 0.3, 0.9)
nu_input    = 1.5   # nu > 1

plot_sb_vs_beta(gamma_input, xi_input, nu_input)