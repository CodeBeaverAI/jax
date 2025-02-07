# Copyright 2024 The JAX Authors.
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

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()


class PallasTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.is_cuda_compute_capability_at_least("9.0"):
      self.skipTest("Only works on a GPU with capability >= sm90")

    super().setUp()


class PallasCallTest(PallasTest):

  @parameterized.named_parameters(
      ("add_float", lambda x, y: x + y, np.float32),
      ("add_int", lambda x, y: x + y, np.int32)
  )
  def test_binary_op(self, bop, dtype):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], dtype=dtype),
    )
    def kernel(x_ref, y_ref, o_ref):
      o_ref[...] = bop(x_ref[...], y_ref[...])

    x = jnp.arange(256).astype(dtype)
    y = x + 1
    np.testing.assert_array_equal(kernel(x, y), bop(x, y))


if __name__ == "__main__":
  absltest.main()
