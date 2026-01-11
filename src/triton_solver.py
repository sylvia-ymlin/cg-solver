import torch
import triton
import triton.language as tl
import argparse
import time

"""
Triton Implementation of the 2D Poisson CG Solver.
Requires: NVIDIA GPU (Triton does not currently support Mac MPS).
"""

@triton.jit
def laplacian_kernel(
    x_ptr,      # Pointer to input vector x
    y_ptr,      # Pointer to output vector y (y = Ax)
    n,          # Grid dimension (n x n)
    BLOCK_SIZE: tl.constexpr
):
    """
    Computes y = A * x for the 2D Laplacian on a flattened n*n grid.
    Stencil: 
          -1
       -1  4 -1
          -1
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n * n

    # Current index (flattened)
    idx = offsets
    
    # Row and Col logic
    row = idx // n
    col = idx % n

    # Center value: 4 * x[i, j]
    center_val = tl.load(x_ptr + idx, mask=mask, other=0.0)
    res = 4.0 * center_val

    # Neighbors (Boundary checks in mask)
    # Up: (row-1, col) -> idx - n
    mask_up = mask & (row > 0)
    val_up = tl.load(x_ptr + idx - n, mask=mask_up, other=0.0)
    res -= val_up

    # Down: (row+1, col) -> idx + n
    mask_down = mask & (row < n - 1)
    val_down = tl.load(x_ptr + idx + n, mask=mask_down, other=0.0)
    res -= val_down

    # Left: (row, col-1) -> idx - 1
    mask_left = mask & (col > 0)
    val_left = tl.load(x_ptr + idx - 1, mask=mask_left, other=0.0)
    res -= val_left

    # Right: (row, col+1) -> idx + 1
    mask_right = mask & (col < n - 1)
    val_right = tl.load(x_ptr + idx + 1, mask=mask_right, other=0.0)
    res -= val_right

    # Store result
    tl.store(y_ptr + idx, res, mask=mask)


def triton_cg_solver(n, max_iter=1000, tol=1e-6):
    device = torch.device('cuda')
    print(f"Running Triton Solver on {torch.cuda.get_device_name(0)}")
    
    # 1. Setup Data
    grid = torch.linspace(0, 1, n, device=device, dtype=torch.float32)
    X, Y = torch.meshgrid(grid, grid, indexing='ij')
    u_exact = X * (1 - X) * Y * (1 - Y)
    x = torch.zeros_like(u_exact)
    
    # Flatten for kernel
    x_flat = x.flatten()
    u_flat = u_exact.flatten()
    
    # Precompute RHS b = A * u_exact
    # We use our Triton kernel to compute b!
    b_flat = torch.empty_like(u_flat)
    
    # Kernel Launch Config
    grid_size = lambda meta: (triton.cdiv(n*n, meta['BLOCK_SIZE']), )
    
    laplacian_kernel[grid_size](u_flat, b_flat, n, BLOCK_SIZE=1024)
    
    # CG Loop
    r = b_flat.clone()
    p = r.clone()
    rsold = torch.dot(r, r)
    
    start_time = time.time()
    
    # Helper for MatMul A*p in loop
    Ap = torch.empty_like(p)
    
    for i in range(max_iter):
        # Ap = A * p (Triton Kernel)
        laplacian_kernel[grid_size](p, Ap, n, BLOCK_SIZE=1024)
        
        # Standard PyTorch (cuBLAS) for vector ops - highly optimized already
        alpha = rsold / torch.dot(p, Ap)
        x_flat = x_flat + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r, r)
        
        if torch.sqrt(rsnew) < tol:
            print(f"Converged at {i} iterations. Residual: {torch.sqrt(rsnew):.2e}")
            break
            
        p = r + (rsnew / rsold) * p
        rsold = rsnew
        
    duration = time.time() - start_time
    print(f"Triton Time: {duration:.4f}s")
    
    # Verify
    error = torch.norm(x_flat - u_flat) / torch.norm(u_flat)
    print(f"Error: {error:.2e}")

    
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: Triton requires NVIDIA GPU (CUDA). Running on Mac? Use src/gpu_solver.py instead.")
    else:
        triton_cg_solver(1000)
