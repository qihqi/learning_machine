import functools
from typing import Tuple, Callable, List, Optional
import time
import dataclasses

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils, shard_map
from jax.sharding import PositionalSharding


from jax.sharding import Mesh
from jax.sharding import PartitionSpec
from jax.sharding import NamedSharding

devices = jax.devices()
P = PartitionSpec

devices = mesh_utils.create_device_mesh((len(devices),))
mesh = Mesh(devices, axis_names=("x",))
# y = jax.device_put(x, NamedSharding(mesh, P('a', 'b')))

L = 1 << 15


@dataclasses.dataclass
class BenchmarkCase:
  """BenchmarkCase."""

  name: str
  function: Callable
  args_shape: List[Tuple]
  args_sharding: List[PartitionSpec]
  profiler_output: Optional[str] = None


start_key = jax.random.key(0)


def _new_arg(shape, dtype):
  global start_key  #  pylint: disable=all
  start_key, _ = jax.random.split(start_key)
  with jax.default_device(jax.devices("cpu")[0]):
    if dtype == jnp.int8.dtype:
      return jax.random.randint(start_key, shape, 0, 100, dtype=dtype)
    else:
      return jax.random.normal(start_key, shape, dtype=dtype) + 1


def _new_args(case, dtype):
  args = []
  for shape, sharding in zip(case.args_shape, case.args_sharding):
    arg = _new_arg(shape, dtype)
    if sharding is not None:
      arg = jax.device_put(arg, NamedSharding(mesh, sharding))
    args.append(arg)
  return args


def _run_case(case, warmup=2, runtimes=5, dtype=jnp.bfloat16.dtype):
  for _ in range(warmup):
    args = _new_args(case, dtype)
    print(jax.jit(case.function).lower(*args).as_text())

  stamps = []
  for i in range(runtimes):
    args = _new_args(case, dtype)
    jax.block_until_ready(args)
    if case.profiler_output is not None and i == (runtimes - 1):
      jax.profiler.start_trace(case.profiler_output)
    start = time.perf_counter()
    jax.block_until_ready(case.function(*args))
    end = time.perf_counter()
    if case.profiler_output is not None and i == (runtimes - 1):
      jax.profiler.stop_trace()
    stamps.append(end - start)
  return sum(stamps) / runtimes


def _llama_ffn(x, w1, w2, w3):
  w1_res = jax.nn.silu((x @ w1).astype(jnp.bfloat16.dtype))
  w3_res = x @ w3
  res = (w1_res * w3_res) @ w2
  return res


@jax.jit
@functools.partial(
    shard_map.shard_map,
    mesh=mesh,
    in_specs=(P(), P(None, "x"), P("x"), P(None, "x")),
    out_specs=(P()),
)
def _llama_ffn_shmap(x, w1, w2, w3):
  for _ in range(3):
    x = _llama_ffn(x, w1, w2, w3)
    x = jax.lax.psum(x, "x")
  return x


@jax.jit
def _llama_ffn_spmd(x, w1, w2, w3):
  for _ in range(3):
    x = _llama_ffn(x, w1, w2, w3)
    x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P()))
  return x


dim = 4096
multiple_of = 256
# hidden_dim = 4 * dim
# hidden_dim = int(2 * hidden_dim / 3)
# hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
hidden_dim = 11008
BATCH = 1024


@jax.jit
@functools.partial(
    shard_map.shard_map,
    mesh=mesh,
    in_specs=(P("x"),),
    out_specs=(P()),
    check_rep=False,
)
def _all_gather(x):
  return jax.lax.all_gather(x, "x")


@jax.jit
@functools.partial(
    shard_map.shard_map, mesh=mesh, in_specs=(P("x"),), out_specs=(P())
)
def _all_reduce(x):
  return jax.lax.psum(x, "x")


allcases = [
    BenchmarkCase(
        name="Matmul replicated",
        function=jax.jit(jnp.matmul),
        args_shape=((L, L), (L, L)),
        args_sharding=(P(), P()),  # replicated
    ),
    BenchmarkCase(
        name="Matmul sharded colrow",
        function=jax.jit(jnp.matmul),
        args_shape=((L, L), (L, L)),
        args_sharding=(P(None, "x"), P("x")),  # replicated
    ),
    BenchmarkCase(
        name="matmul sharded rowcol",
        function=jax.jit(jnp.matmul),
        args_shape=((L, L), (L, L)),
        args_sharding=(P("x"), P("x", None)),  # replicated
    ),
    BenchmarkCase(
        name="all_gather",
        function=_all_gather,
        args_shape=((L, L),),
        args_sharding=(P("x"),),  # replicated
    ),
    BenchmarkCase(
        name="all_reduce",
        function=_all_reduce,
        args_shape=((L, L),),
        args_sharding=(P("x"),),  # replicated
    ),
    BenchmarkCase(
        name="Llama 3xffn shardmap",
        function=_llama_ffn_shmap,
        args_shape=(
            (BATCH, dim),
            (dim, hidden_dim),
            (hidden_dim, dim),
            (dim, hidden_dim),
        ),
        args_sharding=(P(), P(None, "x"), P("x"), P(None, "x")),
    ),
    BenchmarkCase(
        name="Llama 3xffn gspmd",
        function=_llama_ffn_spmd,
        args_shape=(
            (BATCH, dim),
            (dim, hidden_dim),
            (hidden_dim, dim),
            (dim, hidden_dim),
        ),
        args_sharding=(P(), P(None, "x"), P("x"), P(None, "x")),
    ),
]


def _run_call_cases(cases):
  for dtype in (jnp.bfloat16.dtype, jnp.int8.dtype):
    for case in cases:
      avg = _run_case(case, dtype=dtype)
      dtype_size = 2 if dtype == jnp.bfloat16.dtype else 1
      input_sizes = tuple(
          [
              f"{np.prod(size) * dtype_size / (1<<20) :.6} MiB"
              for size in case.args_shape
          ]
      )
      print(
          f"{dtype} \t {case.name}: \t{avg * 1000 :.6} ms \t sizes: {input_sizes}"
      )


def main():
  print("Number of devices: ", len(devices))
  _run_call_cases(allcases)


if __name__ == "__main__":
  main()
