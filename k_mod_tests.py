import numpy as np
import pickle
from matplotlib import pyplot as plt
from time import time


import cupyx
import cupy as cp
from numba import cuda
# from scipy.sparse import csr_matrix

fid = open('k_mod_structured_mesh_fine', 'rb')
# fid = open('k_mod_unstructured_mesh_fine','rb')
# fid = open('k_mod_very_unstructured_mesh_fine','rb')
# fid = open('k_mod_unstructured_Tet_mesh_fine','rb')
fid = open('k_mod_unstructured_Hex_mesh_fine','rb')

k_mod = pickle.load(fid).astype(np.float32)
print(k_mod.nnz)

a = k_mod.data
a[a<1e-9] = 0 
k_mod.data = a
k_mod.eliminate_zeros()
print(k_mod.nnz)


u_curr = np.random.random(k_mod.shape[0]).astype(np.float32)
# k_mod = k_mod.tocsr()

u_curr_gpu = cp.array(u_curr, dtype=np.float32)
k_mod_gpu = cupyx.scipy.sparse.csr_matrix(k_mod, dtype=np.float32)
# result_gpu = cp.zeros(u_curr.shape)
# result_cpu = k_mod @ u_curr
# result_gpu = k_mod_gpu @ u_curr_gpu
r = k_mod_gpu.dot(u_curr_gpu)



num = 2000
_start = time()
for i in range(num):
    # k_mod @ u_curr 
    # k_mod_gpu.dot(u_curr_gpu)
    r = k_mod_gpu.dot(r)
cuda.synchronize()
print('total duration:', time()-_start)

mempool = cp.get_default_memory_pool()
mempool.free_all_blocks()
    
    
    
    
    