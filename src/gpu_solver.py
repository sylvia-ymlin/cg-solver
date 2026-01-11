import torch
import time
import argparse

# Config
N = 1000  # Grid Size (N*N matrix)
TOL = 1e-6
MAX_ITER = 1000

def laplacian_2d(n, device='cpu', dtype=torch.float32):
    """
    Creates the 2D Laplacian matrix (A) as a sparse tensor.
    System Ax = b
    Size: n^2 x n^2
    """
    # For very large N, explicitly building sparse matrix might be slow/heavy.
    # We can use a LinearOperator logic or just stencil convolution.
    # For compactness here, we'll use a convolution-based MatMul (Matrix-Free).
    # This is much faster on GPUs for stencils.
    
    # Kernel: [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
    # Convolution approach enforces A*x without building A.
    
    stencil = torch.tensor([[0, -1, 0], 
                           [-1, 4, -1], 
                           [0, -1, 0]], device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
    return stencil

def matmul_A(x, stencil, n):
    """Compute A * x using convolution (Matrix-Free). x is (N, N)"""
    # x needs to be (Batch, Channel, H, W) -> (1, 1, n, n)
    x_in = x.unsqueeze(0).unsqueeze(0)
    # Conv2d padding=1 preserves size
    y = torch.nn.functional.conv2d(x_in, stencil, padding=1)
    return y.squeeze()

def cg_solver(n, device_name='mps', use_half=False):
    """
    Conjugate Gradient Solver on GPU
    """
    if device_name == 'mps' and not torch.backends.mps.is_available():
        print("Warning: MPS not available. Falling back to CPU.")
        device_name = 'cpu'
        
    device = torch.device(device_name)
    dtype = torch.float16 if use_half else torch.float32
    
    print(f"Running on {device_name.upper()} with {dtype}...")
    
    # 1. Setup Data
    # Real solution u_exact = x(1-x)y(1-y)
    # b = A * u_exact (We reverse engineer b to check convergence)
    
    grid = torch.linspace(0, 1, n, device=device, dtype=dtype)
    X, Y = torch.meshgrid(grid, grid, indexing='ij')
    u_exact = X * (1 - X) * Y * (1 - Y)
    
    # Operator A (Stencil)
    stencil = laplacian_2d(n, device, dtype)
    
    # RHS b = A * u_exact
    b = matmul_A(u_exact, stencil, n)
    
    # Initial Guess x0 = 0
    x = torch.zeros_like(b)
    
    # Initial Residual r0 = b - Ax0 = b
    r = b.clone()
    p = r.clone()
    rsold = torch.dot(r.flatten(), r.flatten())
    
    # Timing
    start_time = time.time()
    
    for i in range(MAX_ITER):
        Ap = matmul_A(p, stencil, n)
        alpha = rsold / torch.dot(p.flatten(), Ap.flatten())
        
        x = x + alpha * p
        r = r - alpha * Ap
        
        rsnew = torch.dot(r.flatten(), r.flatten())
        
        if torch.sqrt(rsnew) < TOL:
            print(f"Converged at iter {i} with residual {torch.sqrt(rsnew):.2e}")
            break
            
        p = r + (rsnew / rsold) * p
        rsold = rsnew
        
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Time: {duration:.4f}s | Per Iter: {duration/(i+1)*1000:.2f}ms")
    
    # Check Error
    error = torch.norm(x - u_exact) / torch.norm(u_exact)
    print(f"Relative Error: {error:.2e}")
    return duration, error

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=1000, help='Grid size')
    parser.add_argument('--device', type=str, default='mps', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--mixed', action='store_true', help='Use FP16 (Mixed Precision emulation)')
    args = parser.parse_args()
    
    cg_solver(args.N, args.device, args.mixed)
