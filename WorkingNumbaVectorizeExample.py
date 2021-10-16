import math
import numpy as np
import cupy as cp
from timeit import default_timer as timer

from numba import cuda, vectorize, types
import numba as nb

@vectorize([nb.float32(nb.float32, nb.float32, nb.float32)], target="cuda")
def nb_function_fast(xx, yy, xy):
    c1 = nb.float32(0)
    c2 = nb.float32(2)
    c3 = nb.float32(4)
    c4 = nb.float32(5)
    sqrt_term = math.sqrt(max(c1, xx * xx - c2 * xx * yy + c3 * xy * xy + yy * yy))
    return c4 * (xx + yy - sqrt_term)


@cp.fuse
def cp_function(xx, yy, xy):
    sqrt_term = cp.sqrt(cp.maximum(0., xx * xx - 2. * xx * yy + 4. * xy * xy + yy * yy))
    return .5 * (xx + yy - sqrt_term)


sz = 4096
a1, a2, a3 = (cp.empty((sz, sz), dtype=np.float32) for _ in range(3))
cuda.synchronize()

# Warmup
for _ in range(3):
    cp_function(a1, a2, a3)

# Timeit
_start = timer()
e1, e2 = cp.cuda.Event(), cp.cuda.Event()
cp_function(a1, a2, a3)

e1.record()
cp_function(a1, a2, a3)
e2.record()
cuda.synchronize()

print(f"Cupy: {cp.cuda.get_elapsed_time(e1, e2):.2f} ms")
print('cupy timer;', (timer()-_start)*1_000,' ms')




sz = 4096
a1, a2, a3, a_out = (cuda.device_array((sz, sz), dtype=np.float32) for _ in range(4))

# Warmup
for _ in range(3):
    nb_function_fast(a1, a2, a3, out=a_out)

# Timeit
_start = timer()
start_event = cuda.event()
end_event = cuda.event()
start_event.record()

nb_function_fast(a1, a2, a3, out=a_out)

end_event.record()
cuda.synchronize()

print('Vectorize timer;', (timer()-_start)*1_000,' ms') # ~19ms
print(f"Vectorize Event Timing: {cuda.event_elapsed_time(start_event, end_event):.3f} ms") #~6.65ms

# del a1, a2, a3, a_out
# cleanup
# cuda.get_current_device().reset()

