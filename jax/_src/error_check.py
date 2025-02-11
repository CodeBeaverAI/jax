# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from contextlib import contextmanager
from itertools import count
import jax
import jax.numpy as jnp
from jax import Array
from jax._src import core
from jax._src import source_info_util, traceback_util
from jax._src.source_info_util import Traceback
from jax._src.core import MutableArray, mutable_array
from jax.sharding import NamedSharding, PartitionSpec as P


traceback_util.register_exclusion(__file__)


def get_traceback():
  return source_info_util.current().traceback


class JaxValueError(ValueError):
  pass


error_code_ref: MutableArray = None  # pytype: disable=annotation-type-mismatch
_unique_id = count(1).__next__
_no_error = (None, None)
# (error_message, traceback) pair.
_error_list: list[tuple[str | None, Traceback | None]] = [_no_error]


def _try_initialize_error_code_ref():
  global error_code_ref
  if error_code_ref is None:
    with core.eval_context():
      _error_code = jnp.uint32(0)
      error_code_ref = mutable_array(_error_code)


# TODO(ayx): for vmap and shard_map.
# def _reduce_fn(x):
#   return x.ravel()[jnp.flatnonzero(x, size=1)].reshape(())


def set_error_if(cond: Array, msg: str) -> None:
  """Set error if cond is true."""
  _try_initialize_error_code_ref()

  err_msg = "Condition must be a scalar or have the same shape as the mesh"
  if cond.shape != error_code_ref.shape:
    raise ValueError(err_msg)

  error_code = error_code_ref[...]
  should_update = jnp.logical_and(cond, error_code == 0)
  error_code = jnp.where(should_update, _unique_id(), error_code)
  # TODO(ayx): change this to mutable array accumulator for vmap and shard_map.
  error_code_ref[...] = error_code
  traceback_obj = get_traceback()
  _error_list.append((msg, traceback_obj))


def raise_if_error() -> None:
  """Raise error if there is any error set."""
  _try_initialize_error_code_ref()
  error_code = error_code_ref[...]
  # TODO(ayx): use the following line in vmap and shard_map.
  # error_code = reduce_fn(error_code_ref[...]).item()
  try:
    if error_code == 0:
      return
    msg, traceback_obj = _error_list[error_code]
    exc = JaxValueError(msg)
    filtered_tb = traceback_util.filter_traceback(traceback_obj.as_python_traceback())
    exc.with_traceback(filtered_tb)
    raise exc
  finally:
    error_code_ref[...] = jnp.zeros(shape=error_code_ref.shape, dtype=error_code_ref.dtype)


# TODO(ayx): for vmap and shard_map.
# @contextmanager
# def error_checking_context(mesh):
#   sharding = NamedSharding(mesh, P(*mesh.axis_names))
#   global error_code_ref
#   old_error_code_ref = error_code_ref
#   new_error_code = jnp.zeros_like(mesh.device_ids, dtype=jnp.int32)
#   new_error_code = jax.make_array_from_callback(
#       new_error_code.shape, sharding, lambda idx: new_error_code[idx]
#   )
#   new_error_code_ref = mutable_array(new_error_code)
#   # redefine a global mutable array that has the same size as the mesh.
#   error_code_ref = new_error_code_ref
#   try:
#     yield
#   finally:
#     error_code_ref = old_error_code_ref
