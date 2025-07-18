import unittest

import numpy as np

from src.ops.math import MathOps
from src.tensor import Tensor


class TestMathOps(unittest.TestCase):

    def test_log_forward_backward(self):
        a = Tensor([[1.0, np.e]], requires_grad=True)
        props = MathOps.log(a)
        result = Tensor.from_TProps(props)

        expected = np.log(a.data)
        np.testing.assert_array_almost_equal(result.data, expected)

        result.backward(np.ones_like(result.data))
        np.testing.assert_array_almost_equal(a.grad, 1 / a.data)  # type: ignore

    def test_log_no_grad(self):
        a = Tensor([[1.0, 2.0]], requires_grad=False)
        props = MathOps.log(a)
        result = Tensor.from_TProps(props)
        expected = np.log(a.data)
        np.testing.assert_array_almost_equal(result.data, expected)
        self.assertIsNone(props.grad_fn)

    def test_exp_forward_backward(self):
        a = Tensor([[0.0, 1.0]], requires_grad=True)
        props = MathOps.exp(a)
        result = Tensor.from_TProps(props)

        expected = np.exp(a.data)
        np.testing.assert_array_almost_equal(result.data, expected)

        result.backward(np.ones_like(result.data))
        np.testing.assert_array_almost_equal(a.grad, expected)  # type: ignore

    def test_exp_no_grad(self):
        a = Tensor([[0.0, 1.0]], requires_grad=False)
        props = MathOps.exp(a)
        result = Tensor.from_TProps(props)
        expected = np.exp(a.data)
        np.testing.assert_array_almost_equal(result.data, expected)
        self.assertIsNone(props.grad_fn)

    def test_tanh_forward_backward(self):
        a = Tensor([[0.0, 1.0]], requires_grad=True)
        props = MathOps.tanh(a)
        result = Tensor.from_TProps(props)

        expected = np.tanh(a.data)
        np.testing.assert_array_almost_equal(result.data, expected)

        result.backward(np.ones_like(result.data))
        expected_grad = 1 - expected**2
        np.testing.assert_array_almost_equal(a.grad, expected_grad)  # type: ignore

    def test_tanh_no_grad(self):
        a = Tensor([[0.0, 1.0]])
        result = Tensor.from_TProps(MathOps.tanh(a))
        expected = np.tanh(a.data)
        np.testing.assert_array_equal(result.data, expected)
        self.assertIsNone(result.grad_fn)
