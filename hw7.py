import numpy as np

def jacobi(A, b, tol=1e-6, max_iter=1000):
    x = np.zeros_like(b)
    D = np.diag(A)
    R = A - np.diagflat(D)
    for _ in range(max_iter):
        x_new = (b - np.dot(R, x)) / D
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

def gauss_seidel(A, b, tol=1e-6, max_iter=1000):
    x = np.zeros_like(b)
    for _ in range(max_iter):
        x_new = np.copy(x)
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

def sor(A, b, omega=1.25, tol=1e-6, max_iter=1000):
    x = np.zeros_like(b)
    for _ in range(max_iter):
        x_new = np.copy(x)
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (1 - omega) * x[i] + (omega * (b[i] - s1 - s2) / A[i, i])
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

def conjugate_gradient(A, b, tol=1e-6, max_iter=1000):
    x = np.zeros_like(b)
    r = b - np.dot(A, x)
    p = r.copy()
    rs_old = np.dot(r.T, r)
    for _ in range(max_iter):
        Ap = np.dot(A, p)
        alpha = rs_old / np.dot(p.T, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r.T, r)
        if np.sqrt(rs_new) < tol:
            return x
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x

# Matrix A and vector b
A = np.array([
    [4, -1, 0, -1, 0, 0],
    [-1, 4, -1, 0, -1, 0],
    [0, -1, 4, 0, 1, -1],
    [-1, 0, 0, 4, -1, -1],
    [0, -1, 0, -1, 4, -1],
    [0, 0, -1, 0, -1, 4]
], dtype=float)

b = np.array([0, -1, 9, 4, 8, 6], dtype=float)

# Compute solutions
x_jacobi = jacobi(A, b)
x_gs = gauss_seidel(A, b)
x_sor = sor(A, b, omega=1.25)
x_cg = conjugate_gradient(A, b)

# Print results
print("Jacobi solution:", x_jacobi)
print("Gauss-Seidel solution:", x_gs)
print("SOR solution:", x_sor)
print("Conjugate Gradient solution:", x_cg)
