import numpy as np
from numba import cuda
import math

@cuda.jit
def forward_elimination_kernel(A, b, n, pivot):
    row = cuda.grid(1)
    
    if row <= pivot or row >= n:
        return
    
    factor = A[row, pivot] / A[pivot, pivot]
    
    for col in range(pivot, n):
        A[row, col] -= factor * A[pivot, col]
    
    b[row] -= factor * b[pivot]

@cuda.jit
def back_substitution_kernel(A, b, x, n, current):
    idx = cuda.grid(1)
    
    if idx == 0:
        sum_val = 0.0
        for j in range(current + 1, n):
            sum_val += A[current, j] * x[j]
        x[current] = (b[current] - sum_val) / A[current, current]

def gaussian_elimination_cuda(A_host, b_host):
    n = A_host.shape[0]
    
    # Copy to GPU
    A_gpu = cuda.to_device(A_host.astype(np.float32))
    b_gpu = cuda.to_device(b_host.astype(np.float32))
    x_gpu = cuda.device_array(n, dtype=np.float32)
    
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    
    # Forward elimination
    for pivot in range(n - 1):
        forward_elimination_kernel[blocks_per_grid, threads_per_block](A_gpu, b_gpu, n, pivot)
        cuda.synchronize()
    
    # Back substitution
    for i in range(n - 1, -1, -1):
        back_substitution_kernel[1, 1](A_gpu, b_gpu, x_gpu, n, i)
        cuda.synchronize()
    
    return x_gpu.copy_to_host()

# Your original data
A = np.array([[3.0, 1.0, -1.0],
              [1.0, -2.0, 3.0],
              [-2.0, 1.0, 1.0]], dtype=np.float32)
b = np.array([1.0, 2.0, 0.0], dtype=np.float32)

print("Matrix A:")
print(A)
print("Vector b:")
print(b)

# Solve using CUDA
x = gaussian_elimination_cuda(A.copy(), b.copy())

print("\nSolution x:")
print(x)

print("\nVerification (A @ x):")
print(np.dot(A, x))

