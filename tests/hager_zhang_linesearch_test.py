# Copyright 2021 Google LLC
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

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp

from jaxopt import HagerZhangLineSearch
from jaxopt import objective
from jaxopt._src import test_util
from jaxopt.tree_util import tree_scalar_mul
from jaxopt.tree_util import tree_vdot
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_negative

import numpy as onp

from sklearn import datasets


class HagerZhangLinesearchTest(test_util.JaxoptTestCase):

  def setUp(self):
    self.rng = lambda: onp.random.RandomState(0)

  def _check_step_in_state(self, x, p, s, fun, fun_der, state):
    step = tree_add_scalar_mul(x, s, p)
    self.assertAllClose(step, state.params, atol=1e-5, rtol=1e-5)
    self.assertAllClose(fun(step), state.value, atol=1e-5, rtol=1e-5)
    self.assertAllClose(fun_der(step), state.grad, atol=1e-5, rtol=1e-5)
    self.assertTrue(~(state.failed & state.done))

  def _assert_conds(self, s, value_fun, slope_fun, c1=1e-4, c2=0.9, err_msg=""):
    value_init = value_fun(0)
    value_step = value_fun(s)
    slope_init = slope_fun(0)
    slope_step = slope_fun(s)
    msg = (
        "s = {}; value(0) = {}; value(s) = {}; slope(0) = {}; slope(s) = {}; {}"
        .format(s, value_init, value_step, slope_init, slope_step, err_msg)
    )

    self.assertTrue(
        value_step <= value_init + c1 * s * slope_init,
        "Sufficient decrease (Armijo) failed: " + msg,
    )
    self.assertTrue(
        abs(slope_step) <= abs(c2 * slope_init),
        "Small curvature (strong Wolfe) failed: " + msg,
    )

  def _assert_line_conds(self, x, p, s, fun, fun_der, **kw):
    self._assert_conds(
        s,
        value_fun=lambda sp: fun(x + p * sp),
        slope_fun=lambda sp: jnp.dot(fun_der(x + p * sp), p),
        **kw,
    )

  # OLD TESTS
  def test_logreg(self):
    x, y = datasets.make_classification(
        n_samples=10, n_features=5, n_classes=2, n_informative=3, random_state=0
    )
    data = (x, y)
    fun = objective.binary_logreg

    def fun_(w):
      return fun(w, data)

    rng = onp.random.RandomState(0)
    w_init = rng.randn(x.shape[1])
    initial_grad = jax.grad(fun)(w_init, data=data)
    descent_dir = tree_negative(initial_grad)

    # Call to run.
    ls = HagerZhangLineSearch(fun=fun, maxiter=20)
    stepsize, state = ls.run(
        init_stepsize=1.0, params=w_init, fun_kwargs={"data": data}
    )

    self._assert_line_conds(
        w_init, descent_dir, stepsize, fun_, jax.grad(fun_), c1=ls.c1, c2=ls.c2
    )
    self._check_step_in_state(
        w_init, descent_dir, stepsize, fun_, jax.grad(fun_), state
    )

    # Call to run with value_and_grad=True
    ls = HagerZhangLineSearch(
        fun=jax.value_and_grad(fun), maxiter=20, value_and_grad=True
    )
    stepsize, state = ls.run(
        init_stepsize=1.0, params=w_init, fun_kwargs={"data": data}
    )

    self._assert_line_conds(
        w_init, descent_dir, stepsize, fun_, jax.grad(fun_), c1=ls.c1, c2=ls.c2
    )
    self._check_step_in_state(
        w_init, descent_dir, stepsize, fun_, jax.grad(fun_), state
    )

  @parameterized.product(val=[onp.inf, onp.nan])
  def test_hager_zhang_linesearch_non_finite(self, val):

    def fun(x):
      result = jnp.where(x > 4., val, (x - 2)**2)
      grad = jnp.where(x > 4., onp.nan, 2 * (x - 2.))
      return result, grad
    x_init = -0.001

    ls = HagerZhangLineSearch(fun=fun, value_and_grad=True, jit=False)
    stepsize = 1.25
    state = ls.init_state(init_stepsize=1.25, params=x_init)

    stepsize, state = ls.update(stepsize=stepsize, state=state, params=x_init)
    # Should work around the Nan/Inf regions and provide a reasonable step size.
    self.assertTrue(state.done)

  ###########
  # NEW TESTS
  # -- scalar functions

  def _scalar_fun_1(self, s):
    p = -s - s**3 + s**4
    dp = -1 - 3 * s**2 + 4 * s**3
    return p, dp

  def _scalar_fun_2(self, s):
    p = jnp.exp(-4 * s) + s**2
    dp = -4 * jnp.exp(-4 * s) + 2 * s
    return p, dp

  def _scalar_fun_3(self, s):
    p = -jnp.sin(10 * s)
    dp = -10 * jnp.cos(10 * s)
    return p, dp

  # -- n-d functions

  def _line_fun_1(self, x):
    f = jnp.dot(x, x)
    df = 2 * x
    return f, df

  def _line_fun_2(self, x):
    f = jnp.dot(x, jnp.dot(self.A, x)) + 1
    df = jnp.dot(self.A + self.A.T, x)
    return f, df

  def _rosenbrock_fun(self, x):
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

  def _line_fun_3(self, x):
    # Rosenbrock function
    f = self._rosenbrock_fun(x)
    df = jax.grad(self._rosenbrock_fun)(x)
    return f, df
  
    # -- Generic scalar searches

  @parameterized.product(
      name=["_scalar_fun_1", "_scalar_fun_2", "_scalar_fun_3"]
  )
  def test_scalar_search(self, name):
    def bind_index(fun, idx):
      # Remember Python's closure semantics!
      return lambda *a, **kw: fun(*a, **kw)[idx]

    value = getattr(self, name)
    fun = bind_index(value, 0)
    fun_der = bind_index(value, 1)
    for old_value in self.rng().randn(3):
      ls = HagerZhangLineSearch(fun)
      x, p = 0.0, 1.0
      s, state = ls.run(init_stepsize=1.0, params=x, descent_direction=p)
      self._check_step_in_state(x, p, s, fun, fun_der, state)
      self._assert_conds(s, fun, fun_der, err_msg=f"{name} {old_value:g}")

 # -- Generic line searches

  @parameterized.product(name=["_line_fun_1", "_line_fun_2", "_line_fun_3"])
  def test_line_search(self, name):
    def bind_index(fun, idx):
      # Remember Python's closure semantics!
      return lambda *a, **kw: fun(*a, **kw)[idx]

    value = getattr(self, name)
    fun = bind_index(value, 0)
    fun_der = bind_index(value, 1)

    k = 0
    N = 20
    rng = self.rng()
    # sets A in one of the line functions
    self.A = self.rng().randn(N, N)
    while k < 9:
      x = rng.randn(N)
      p = rng.randn(N)
      if jnp.dot(p, fun_der(x)) >= 0:
        # always pick a descent pk
        continue
      if fun(x + 1e6 * p) < fun(x):
        # If the function is unbounded below, the linesearch cannot finish
        continue
      k += 1

      f0 = fun(x)
      g0 = fun_der(x)
      ls = HagerZhangLineSearch(fun)
      s, state = ls.run(
          init_stepsize=1.0, params=x, descent_direction=p, value=f0, grad=g0
      )
      self._check_step_in_state(x, p, s, fun, fun_der, state)
      self._assert_line_conds(x, p, s, fun, fun_der, err_msg=f"{name}")

  def test_aux_value(self):
    def fun(x):
      return jnp.cos(jnp.sum(jnp.exp(-x)) ** 2), x

    x = jnp.ones(2)
    p = jnp.array([-0.5, -0.25])
    ls = HagerZhangLineSearch(fun=fun, maxiter=100, has_aux=True)
    new_stepsize, state = ls.run(
        init_stepsize=1.0, params=x, descent_direction=p
    )
    new_x = x + new_stepsize * p
    self.assertArraysEqual(state.aux, new_x)


  @parameterized.product(out_dtype=[jnp.float32, jnp.float64])
  def test_correct_dtypes(self, out_dtype):
    def fun(x):
      return jnp.cos(jnp.sum(jnp.exp(-x)) ** 2).astype(out_dtype)

    with jax.experimental.enable_x64():
      xk = jnp.ones(2, dtype=jnp.float32)
      pk = jnp.array([-0.5, -0.25], dtype=jnp.float32)
      ls = HagerZhangLineSearch(fun, maxiter=100)
      _, state = ls.run(init_stepsize=1.0, params=xk, descent_direction=pk)
      for name in ("done", "failed"):
        self.assertEqual(getattr(state, name).dtype, jnp.bool_)
      for name in ("iter_num",):
        self.assertEqual(getattr(state, name).dtype, jnp.int64)
      for name in ("params", "grad"):
        self.assertEqual(getattr(state, name).dtype, jnp.float32, name)
      for name in ("low", "high"):
        self.assertEqual(getattr(state, name).dtype, jnp.float32, name)
      for name in ("value", "error"):
        self.assertEqual(getattr(state, name).dtype, out_dtype)


if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  # jax.config.update("jax_enable_x64", True)
  absltest.main()
