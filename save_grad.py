import torch
def pack_hook(x):
  print("Packing", x)
  return x
def unpack_hook(x):
  print("Unpacking", x)
  return x

a = torch.ones(5, requires_grad=True)
b = torch.ones(5, requires_grad=True) * 2

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
  y = a * b + a


y.sum().backward()

print('===')
import functools
import jax
import jax.numpy as jnp
import jax._src.test_util as jtu
from jax.ad_checkpoint import Offloadable
import numpy as np

def policy(prim, *avals, **params) -> Offloadable:
  return Offloadable(src='device', dst='pinned_host')

@functools.partial(jax.remat, policy=policy)
def f(x):
  x = jnp.sin(x)
  x = jnp.sin(x)
  return jnp.sum(x)

fwd_jaxpr, bwd_jaxpr = jtu.fwd_bwd_jaxprs(f, np.arange(16.))

gradient = jax.jit(jax.grad(f)).lower(np.arange(16.))

print(gradient.as_text())

