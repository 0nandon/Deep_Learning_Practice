import numpy as np
gpu_enable = True

try:
    import cupy as cp
    cupy = cp
except:
    gpu_enable = False
from dezero import Variable


def get_array_module(x):
    if isinstance(x, Variable):
        x = x.data
        
    if not gpu_enable:
        return np
      
    xp = cp.get_array_module(x)
    return xp
  

  
