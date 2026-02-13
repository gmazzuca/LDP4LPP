import sympy as sp
import time

def analyze_system():
    print("--- STARTING SYMBOLIC COMPUTATION ---")
    
    # 1. Define Variables
    # We use symbols for variables and parameters
    print("[1/6] Defining variables and equations...")
    s1, s2, q1, q2, c0, c1 = sp.symbols('s1 s2 q1 q2 c0 c1')
    K1, K2, E1, E2, nu = sp.symbols('K1 K2 E1 E2 nu')
    
    # Define the "Pre-power" term function for readability
    # P(x) corresponds to (c1*x + c0)^nu
    def P(x):
        return (c1*x + c0)**nu

    # 2. Define the System of 6 Equations
    # Ratio Constraints (Rows 1-2)
    eq1 = s1*q1 - K1*s2*q2
    eq2 = (s1 + 1)*(q1 + 1) - K2*(s2 + 1)*(q2 + 1)

    # Spectral Constraints (Rows 3-6)
    # Form: P(x)*(x+1) - E*x = 0
    spec_s1 = P(s1) * (s1 + 1) - E1*s1
    spec_s2 = P(s2) * (s2 + 1) - E2*s2
    spec_q1 = P(q1) * (q1 + 1) - q1  # E=1 for q
    spec_q2 = P(q2) * (q2 + 1) - q2  # E=1 for q

    # Order of equations and variables is crucial for block structure
    # Equations: [Ratio1, Ratio2, Spec_s1, Spec_s2, Spec_q1, Spec_q2]
    F = sp.Matrix([eq1, eq2, spec_s1, spec_s2, spec_q1, spec_q2])
    
    # Variables: [c0, c1, s1, s2, q1, q2] 
    # Putting c0, c1 first ensures the Top-Left block is independent of them (Zero Block)
    X = sp.Matrix([c0, c1, s1, s2, q1, q2])

    # 3. Compute Jacobian
    print("[2/6] Computing 6x6 Jacobian Matrix...")
    t0 = time.time()
    J = F.jacobian(X)
    dt = time.time() - t0
    print(f"      -> Done in {dt:.2f} seconds.")

    # 4. Extract Blocks
    print("[3/6] Extracting Blocks for Schur Complement...")
    # Structure:
    # [ A (2x2) | B (2x4) ]
    # [ C (4x2) | D (4x4) ]
    
    A = J[0:2, 0:2] # Should be Zero
    B = J[0:2, 2:6] # Ratio derivatives w.r.t s,q
    C = J[2:6, 0:2] # Spectral derivatives w.r.t c0,c1
    D = J[2:6, 2:6] # Spectral derivatives w.r.t s,q (Diagonal)

    # Check structure
    is_A_zero = A.is_zero_matrix
    # Check if D is diagonal (off-diagonal terms are 0)
    is_D_diag = D.is_diagonal()
    
    print(f"      -> Top-Left Block is Zero? {is_A_zero}")
    print(f"      -> Bottom-Right Block is Diagonal? {is_D_diag}")
    
    if not is_D_diag:
        print("      WARNING: D block is not diagonal. Check variable ordering.")

    # 5. Compute Schur Complement (The Effective 2x2 Matrix)
    print("[4/6] Computing Schur Complement (Phi = -B * D_inv * C)...")
    t0 = time.time()
    
    # Since D is diagonal, D_inv is just 1/diagonal_elements
    # We extract the diagonal terms manually to avoid full matrix inversion
    diag_elements = [D[i,i] for i in range(4)]
    
    # Construct B * D_inv
    # This scales each column of B by the inverse of the corresponding D diagonal
    B_scaled = sp.Matrix.zeros(2, 4)
    for col in range(4):
        scale_factor = 1 / diag_elements[col]
        B_scaled[:, col] = B[:, col] * scale_factor
        
    # Phi = A - B_scaled * C (A is zero, so just -B_scaled * C)
    Phi = -B_scaled * C
    
    # Simplify the 2x2 matrix Phi slightly to make determinant easier
    Phi = sp.simplify(Phi)
    dt = time.time() - t0
    print(f"      -> Done in {dt:.2f} seconds.")

    # 6. Compute Determinant
    print("[5/6] Computing Determinant (Det(D) * Det(Phi))...")
    t0 = time.time()
    
    det_D = sp.prod(diag_elements)
    det_Phi = Phi.det()
    full_det = det_D * det_Phi
    
    dt = time.time() - t0
    print(f"      -> Done in {dt:.2f} seconds.")

    # 7. Analyze Limit q1 -> q2
    print("[6/6] Analyzing limit q1 -> q2 and Factoring (This is the heavy step)...")
    t0 = time.time()
    
    # Substitute q2 -> q1
    det_limit = full_det.subs(q2, q1)
    
    # Factor the result to find the term (s1 - s2)
    # We collect terms w.r.t s1, s2 to speed up factoring
    factored_det = sp.factor(det_limit)
    
    dt = time.time() - t0
    print(f"      -> Done in {dt:.2f} seconds.")
    
    print("\n--- RESULTS ---")
    print("Does the determinant contain the factor (s1 - s2)?")
    
    # Check if (s1 - s2) is a factor
    has_s_diff = factored_det.has(s1 - s2)
    
    if has_s_diff:
        print("YES. The term (s1 - s2) is present.")
        print("The determinant is NON-ZERO when q1=q2, provided s1 != s2.")
    else:
        print("NO. Something went wrong or the factor is hidden.")

    return factored_det

# Run the function
if __name__ == "__main__":
    det = analyze_system()