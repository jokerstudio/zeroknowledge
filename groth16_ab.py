import galois
import numpy as np
from py_ecc.optimized_bn128 import multiply, G1, G2, add, pairing, neg, normalize, eq, curve_order, final_exponentiate
from py_ecc.fields import optimized_bn128_FQ12 as FQ12
from string import Template
poly1d = np.poly1d

def evaluate_poly(poly, trusted_points, verbose=False):
    coeff = poly.coefficients()[::-1]

    assert len(coeff) == len(trusted_points), "Polynomial degree mismatch!"

    if verbose:
        [print(normalize(point)) for point in trusted_points]

    terms = [multiply(point, int(coeff)) for point, coeff in zip(trusted_points, coeff)]
    evaluation = terms[0]
    for i in range(1, len(terms)):
        evaluation = add(evaluation, terms[i])

    if verbose:
        print("-" * 10)
        print(normalize(evaluation))
    return evaluation

def evaluate_poly_list(poly_list, x):
    results = []
    for poly in poly_list:
        results.append(poly(x))
    return results

def print_evaluation(name, results):
    print(f'\n{name} polynomial evaluations:')
    for i in range(0, len(results)):
        print(f'{name}_{i} = {results[i]}')

def to_poly(mtx):
    poly_list = []
    for i in range(0, mtx.shape[0]):
        poly_list.append(galois.Poly(mtx[i][::-1]) )
    return poly_list

def print_poly(name, poly_list):
    print(f'\n{name} polynomials:')
    for i in range(0, len(poly_list)):
        print(f'{name}_{i} = {poly_list[i]}')

def split_poly(poly):
    coef = [int(c) for c in poly.coefficients()]
    p1 = coef[-2:]
    p2 = coef[:-2] + [0] * 2

    return galois.Poly(p1, field=FP), galois.Poly(p2, field=FP)

p = curve_order
# p = 71
FP = galois.GF(p)

# input arguments
x = FP(2)
y = FP(3)

v1 = x * x
v2 = y * y
v3 = 5 * x * v1
v4 = 4 * v1 * v2
out = 5*x**3 - 4*x**2*y**2 + 13*x*y**2 + x**2 - 10*y

witness_vector = FP([1, out, x, y, v1, v2, v3, v4])

print("w =", witness_vector)

R = FP([[0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 5, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 4, 0, 0, 0],
         [0, 0, 13, 0, 0, 0, 0, 0]])

L = FP([[0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0]])

O = FP([[0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1],
         [0, 1, 0, 10, FP(p - 1), 0, FP(p - 1), 1]])


mtxs = [L, R, O]
poly_m = []

# Lagrange Interpolation
for m in mtxs:
    poly_list = []
    for i in range(0, m.shape[1]):
        points_x = FP(np.zeros(m.shape[0], dtype=int))
        points_y = FP(np.zeros(m.shape[0], dtype=int))
        for j in range(0, m.shape[0]):
            points_x[j] = FP(j+1)
            points_y[j] = m[j][i]

        poly = galois.lagrange_poly(points_x, points_y)
        coef = poly.coefficients()[::-1]
        if len(coef) < m.shape[0]:
            coef = np.append(coef, np.zeros(m.shape[0] - len(coef), dtype=int))
        poly_list.append(coef)
    
    poly_m.append(FP(poly_list))

Lp = poly_m[0]
Rp = poly_m[1]
Op = poly_m[2]

print("→ witness")
print("↓ statements")

print(f'''Lp
{Lp}
''')

print(f'''Rp
{Rp}
''')

print(f'''Op
{Op}
''')


################################
#                              #
#            SETUP             #
#                              #
################################
print("Setup phase")
print("-"*10)
print("Toxic waste:")

alpha = FP(2)
beta = FP(3)
tau = FP(20)

print(f"α = {alpha}")
print(f"β = {beta}")
print(f"τ = {tau}")

# Generate Target polynomial
T = galois.Poly([1, p-1], field=FP)
for i in range(2, L.shape[0] + 1):
    T *= galois.Poly([1, p-i], field=FP)

print("\nT = ", T)
T_tau = T(tau)
print(f"\nT(τ) = {T_tau}")


beta_U = beta * Lp # β ⊙ A
alpha_V = alpha * Rp # α ⊙ B
_W = Op # W
K = beta_U + alpha_V + _W # preimage of [βU + αV + W] without witness
Kp = to_poly(K) #[K0, K1, K2, ..., Kn] without witness
K_eval = evaluate_poly_list(Kp, tau) # [K0(tau), K1(tau), K2(tau), ..., Kn(tau)] without witness

print(f'''βU
{beta_U}
''')

print(f'''αV
{alpha_V}
''')

print(f'''W
{_W}
''')
print_poly("K", Kp)

print("K evaluations:")
print([int(k) for k in K_eval])

# G1[α]
alpha_G1 = multiply(G1, int(alpha))
# G2[β]
beta_G2 = multiply(G2, int(beta))
# G1[τ^0], G1[τ^1], ..., G1[τ^d]
tau_G1 = [multiply(G1, int(tau**i)) for i in range(0, T.degree)] # T.degree equal to number of constrains
# G2[τ^0], G2[τ^1], ..., G2[τ^d-1]
tau_G2 = [multiply(G2, int(tau**i)) for i in range(0, T.degree)]
# G1[τ^0 * T(τ)], G1[τ^1 * T(τ)], ..., G1[τ^d-1 * T(τ)]
target_G1 = [multiply(G1, int(tau**i * T_tau)) for i in range(0, T.degree - 1)]
# G1[βU0(τ) + αV0(τ) + W0(τ)], G1[βU1(τ) + αV1(τ) + W1(τ)], ..., G1[βUd(τ) + αVd(τ) + Wd(τ)]
k_G1 = [multiply(G1, int(k)) for k in K_eval] # [K0_G1, K1_G1, K2_G1, ..., Kn_G1] without witness

assert len(target_G1) == len(H.coefficients()), f"target_G1 length mismatch! {len(target_G1)} != {len(H.coefficients())}"


print("Trusted setup:")
print("-"*10)
print(f"[α]G1 = {normalize(alpha_G1)}")
print(f"[β]G2 = {normalize(beta_G2)}")
print(f"[τ]G1 = {[normalize(point) for point in tau_G1]}")
print(f"[τ]G2 = {[normalize(point) for point in tau_G2]}")
print(f"[k]G1 = {[normalize(point) for point in k_G1]}")
print(f"[τT(τ)]G1 = {[normalize(point) for point in target_G1]}")

################################
#                              #
#            TEST              #
#                              #
################################
# By multiplying (dot product)w with these matrices, the resulting vector will yield the coefficients for our polynomials
# Lp ⊙ witness_vector
# Rp ⊙ witness_vector
# Op ⊙ witness_vector
U = galois.Poly((witness_vector @ Lp)[::-1])
V = galois.Poly((witness_vector @ Rp)[::-1])
W = galois.Poly((witness_vector @ Op)[::-1])
print("U = ", U)
print("V = ", V)
print("W = ", W)
H = (U * V - W) // T # (U⊙w * V⊙w - W⊙w) / T
rem = (U * V - W) % T

# Test by given tau
u = U(tau)
v = V(tau)
w = W(tau) # sorry, w is taken by witness vector
ht = H(tau) * T_tau

a = alpha + u
b = beta + v
c = w + ht
k = (alpha * beta) + (beta * u) + (alpha * v) # αβ + (βA) + (αB)

# Test Aα * Bβ = αβ + βA + αB + C
assert a * b == k + c # should be equal



################################
#                              #
#            PROVER            #
#                              #
################################
# By multiplying (dot product)w with these matrices, the resulting vector will yield the coefficients for our polynomials
# Lp ⊙ witness_vector
# Rp ⊙ witness_vector
# Op ⊙ witness_vector
U = galois.Poly((witness_vector @ Lp)[::-1])
V = galois.Poly((witness_vector @ Rp)[::-1])
W = galois.Poly((witness_vector @ Op)[::-1])
print("U = ", U)
print("V = ", V)
print("W = ", W)
# Find H with no remainer
H = (U * V - W) // T # (U⊙w * V⊙w - W⊙w) / T
rem = (U * V - W) % T

print("H = ", H)
print("rem = ", rem)

assert rem == 0

print("\nProof generation:")
print("-"*10)

# G1[u0 * τ^0] + G1[u1 * τ^1] + ... + G1[ud-1 * τ^d-1]
A_G1 = evaluate_poly(U, tau_G1) # multiply with tau and sum
# G1[A] = G1[A] + G1[α]
A_G1 = add(A_G1, alpha_G1)
# G2[v0 * τ^0] + G2[v1 * τ^1] + ... + G2[vd-1 * τ^d-1]
B_G2 = evaluate_poly(V, tau_G2) # multiply with tau and sum
# G2[B] = G2[B] + G2[β]
B_G2 = add(B_G2, beta_G2)
# G1[h0 * τ^0 * T(τ)] + G1[h1 * τ^1 * T(τ)] + ... + G1[hd-2 * τ^d-2 * T(τ)]
HT_G1 = evaluate_poly(H, target_G1) # multiply with tau and sum
assert len(witness_vector) == len(k_G1), "Polynomial degree mismatch!"
# w0 * G1[k0] + w1 * G1[k1] + ... + wd-1 * G1[kd-1]
K_G1_terms = [multiply(point, int(scaler)) for point, scaler in zip(k_G1, witness_vector)]
K_G1 = K_G1_terms[0]
for i in range(1, len(K_G1_terms)):
    K_G1 = add(K_G1, K_G1_terms[i])

C_G1 = add(HT_G1, K_G1) # K+C


print(f"[A]G1 = {normalize(A_G1)}")
print(f"[B]G2 = {normalize(B_G2)}")
print(f"[C]G1 = {normalize(C_G1)}")
print("-" * 10)
print("Verifier uses:")
print(f"[α]G1 = {normalize(alpha_G1)}")
print(f"[β]G2 = {normalize(beta_G2)}")



################################
#                              #
#           VERIFIER           #
#                              #
################################
print("\nProof verification:")
print("-"*10)

first = pairing(B_G2, neg(A_G1))
second = pairing(beta_G2, alpha_G1)
third = pairing(G2, C_G1)
# AB == αβ + [βA + αB + C]
assert final_exponentiate(first * second * third) == FQ12.one(), "Pairing check failed!"

print("Pairing check passed!")


###############################
#                             #
# GENERATE VERIFIER CONTRACT  #
#                             #
###############################

with open("Verifier.sol.template", "r") as f:
    template = Template(f.read())
    variables = {
        "aG1_x": normalize(A_G1)[0],
        "aG1_y": normalize(A_G1)[1],
        "bG2_x1": normalize(B_G2)[0].coeffs[0],
        "bG2_x2": normalize(B_G2)[0].coeffs[1],
        "bG2_y1": normalize(B_G2)[1].coeffs[0],
        "bG2_y2": normalize(B_G2)[1].coeffs[1],
        "cG1_x": normalize(C_G1)[0],
        "cG1_y": normalize(C_G1)[1],
        "alphaG1_x": normalize(alpha_G1)[0],
        "alphaG1_y": normalize(alpha_G1)[1],
        "betaG2_x1": normalize(beta_G2)[0].coeffs[0],
        "betaG2_x2": normalize(beta_G2)[0].coeffs[1],
        "betaG2_y1": normalize(beta_G2)[1].coeffs[0],
        "betaG2_y2": normalize(beta_G2)[1].coeffs[1]
    }
    output = template.substitute(variables)

with open("Verifier.sol", "w") as f:
    f.write(output)
