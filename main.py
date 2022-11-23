import time

import torch
from GA import GA

print("Torch version:{}".format(torch.__version__))
print(f"mps 사용 가능 여부: {torch.backends.mps.is_available()}")
print(f"mps 지원 환경 여부: {torch.backends.mps.is_built()}")
use_mps = torch.backends.mps.is_available()
device = torch.device('mps' if use_mps else 'cpu')
print(device)

start_time = time.perf_counter()
# Initialization
pop = GA.Pop()
population, conv_unit = pop.initialization_pop()
pop.evolve(device, population, conv_unit)

end_time = time.perf_counter()
elapsed_time_all = end_time - start_time
print("all CPU Time = ", elapsed_time_all)