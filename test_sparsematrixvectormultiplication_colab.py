import numpy as np
from scipy import sparse
import cupyx
import cupy as cp
from time import time
from numba import cuda


N = 1_600_000
density = 1e-5

x = np.random.random(N).astype(np.float32)
x_gpu = cp.array(x, dtype=np.float32)

data = np.random.random(int(density*N*N))
rows = np.random.randint(0, N, len(data))
cols = np.random.randint(0, N, len(data))

k_mod = sparse.coo_matrix((data, (rows, cols)), shape=(N,N), dtype=np.float32)
k_mod_gpu = cupyx.scipy.sparse.csr_matrix(k_mod, dtype=np.float32)

r_gpu = k_mod_gpu.dot(x_gpu)


num = 2000
_start = time()
for i in range(num):
    # k_mod @ u_curr 
    # k_mod_gpu.dot(u_curr_gpu)
    r_gpu = k_mod_gpu.dot(r_gpu)
cuda.synchronize()
print('total duration:', time()-_start)

mempool = cp.get_default_memory_pool()
mempool.free_all_blocks()

