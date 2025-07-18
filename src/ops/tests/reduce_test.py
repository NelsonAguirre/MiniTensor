import unittest

import numpy as np

from ..reduce import Reduce
from ...tensor import Tensor


class TestReduceOps(unittest.TestCase):
    def test_sum_no_axis(self):
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=False)
        result = Tensor.from_TProps(Reduce.sum(x))
        self.assertEqual(result._data, 10.0)

    def test_sum_axis_keepdims(self):
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        result = Tensor.from_TProps(Reduce.sum(x, dim=1, keepdims=True))
        np.testing.assert_allclose(result._data, [[3.0], [7.0]])
        grad = np.array([[1.0], [1.0]])
        backward = result.grad_fn.backward(grad)[0]  # type: ignore
        np.testing.assert_allclose(backward, np.ones_like(x.data))

    def test_sum_axis_nokeepdims(self):
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        result = Tensor.from_TProps(Reduce.sum(x, dim=1, keepdims=False))
        grad = np.array([1.0, 1.0])
        backward = result.grad_fn.backward(grad)[0]  # type: ignore
        np.testing.assert_allclose(backward, np.ones_like(x.data))

    def test_mean_scalar(self):
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        result = Reduce.mean(x)
        self.assertAlmostEqual(result._data, 2.0)  # type: ignore
        backward = result.grad_fn.backward(np.array(1.0))[0]  # type: ignore
        np.testing.assert_allclose(backward, np.array([1.0, 1.0, 1.0]) / 3)

    def test_max_backward(self):
        x = Tensor(np.array([[1, 3], [5, 2]]), requires_grad=True)
        result = Tensor.from_TProps(Reduce.max(x, dim=0))
        self.assertTrue(x.requires_grad)
        grad = np.array([1.0, 1.0])
        backward = result.grad_fn.backward(grad)[0]  # type: ignore
        expected = np.array([[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_allclose(backward, expected)

    def test_min_backward(self):
        x = Tensor(np.array([[2, 5], [1, 3]]), requires_grad=True)
        result = Tensor.from_TProps(Reduce.min(x, dim=0))
        grad = np.array([1.0, 1.0])
        backward = result.grad_fn.backward(grad)[0]  # type: ignore
        expected = np.array([[0.0, 0.0], [1.0, 1.0]])
        np.testing.assert_allclose(backward, expected)

    def test_max_keepdims(self):
        x = Tensor(np.array([[1, 5], [4, 2]]), requires_grad=True)
        result = Tensor.from_TProps(Reduce.max(x, dim=1, keepdims=True))
        grad = np.array([[1.0], [1.0]])
        backward = result.grad_fn.backward(grad)[0]  # type: ignore
        expected = np.array([[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_allclose(backward, expected)

    def test_min_keepdims(self):
        x = Tensor(np.array([[3, 4], [1, 2]]), requires_grad=True)
        result = Tensor.from_TProps(Reduce.min(x, dim=1, keepdims=True))
        grad = np.array([[1.0], [1.0]])
        backward = result.grad_fn.backward(grad)[0]  # type: ignore
        expected = np.array([[1.0, 0.0], [1.0, 0.0]])
        np.testing.assert_allclose(backward, expected)
