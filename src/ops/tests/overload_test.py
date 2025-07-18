import unittest

import numpy as np

from ..overload import OverloadOps
from ...tensor import Tensor


class TestOverloadOps(unittest.TestCase):
    def test_get_item_forward_backward(self):
        t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        index = (0, slice(None))
        props = Tensor.from_TProps(OverloadOps.get_item(t, index))
        out = props._data
        self.assertTrue(np.allclose(out, np.array([1.0, 2.0])))

        grad_out = np.array([1.0, 1.0])
        back = props.grad_fn.backward(grad_out)[0]
        self.assertTrue(np.allclose(back, np.array([[1.0, 1.0], [0.0, 0.0]])))

    def test_neg_forward_backward(self):
        t = Tensor(np.array([1.0, -2.0, 3.0]), requires_grad=True)
        props = Tensor.from_TProps(OverloadOps.neg(t))
        self.assertTrue(np.allclose(props._data, np.array([-1.0, 2.0, -3.0])))

        grad = np.array([1.0, 1.0, 1.0])
        back = props.grad_fn.backward(grad)[0]
        self.assertTrue(np.allclose(back, -grad))

    def test_add_forward_backward(self):
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        b = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        props = Tensor.from_TProps(OverloadOps.add(a, b))
        self.assertTrue(np.allclose(props._data, np.array([4.0, 6.0])))

        grad = np.array([1.0, 1.0])
        ga, gb = props.grad_fn.backward(grad)
        self.assertTrue(np.allclose(ga, grad))
        self.assertTrue(np.allclose(gb, grad))

    def test_mul_forward_backward(self):
        a = Tensor(np.array([2.0, 3.0]), requires_grad=True)
        b = Tensor(np.array([4.0, 5.0]), requires_grad=True)
        props = Tensor.from_TProps(OverloadOps.mul(a, b))
        self.assertTrue(np.allclose(props._data, np.array([8.0, 15.0])))

        grad = np.array([1.0, 1.0])
        ga, gb = props.grad_fn.backward(grad)
        self.assertTrue(np.allclose(ga, b.data))
        self.assertTrue(np.allclose(gb, a.data))

    def test_matmul_forward_backward_matrix(self):
        a = Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
        b = Tensor(np.array([[3.0], [4.0]]), requires_grad=True)
        props = Tensor.from_TProps(OverloadOps.matmul(a, b))
        self.assertTrue(np.allclose(props._data, np.array([[11.0]])))

        grad = np.array([[1.0]])
        ga, gb = props.grad_fn.backward(grad)
        self.assertTrue(np.allclose(ga, b.data.T))
        self.assertTrue(np.allclose(gb, a.data.T))

    def test_matmul_forward_backward_scalar(self):
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        b = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        props = Tensor.from_TProps(OverloadOps.matmul(a, b))
        self.assertTrue(np.allclose(props._data, np.array(11.0)))

        grad = np.array(1.0)
        ga, gb = props.grad_fn.backward(grad)
        self.assertTrue(np.allclose(ga, b.data))
        self.assertTrue(np.allclose(gb, a.data))

    def test_pow_forward_backward(self):
        a = Tensor([[2.0, 3.0]], requires_grad=True)
        exponent = 3
        props = OverloadOps.pow(a, Tensor(exponent))
        result = Tensor.from_TProps(props)

        expected = a.data**exponent
        np.testing.assert_array_almost_equal(result.data, expected)

        result.backward(np.ones_like(result.data))
        expected_grad = exponent * (a.data ** (exponent - 1))
        np.testing.assert_array_almost_equal(a.grad, expected_grad)  # type: ignore

    def test_pow_no_grad(self):
        a = Tensor([[2.0, 3.0]])
        result = Tensor.from_TProps(OverloadOps.pow(a, Tensor(2)))
        expected = a.data**2
        np.testing.assert_array_equal(result.data, expected)

    def test_sqrt_forward_backward(self):
        a = Tensor([[4.0, 9.0]], requires_grad=True)
        result = a.sqrt()

        expected = np.sqrt(a.data)
        np.testing.assert_array_almost_equal(result.data, expected)

        result.backward(np.ones_like(result.data))
        expected_grad = 0.5 / np.sqrt(a.data)
        np.testing.assert_array_almost_equal(a.grad, expected_grad)  # type: ignore

    def test_sqrt_no_grad(self):
        a = Tensor([[4.0, 9.0]])
        result = a.sqrt()
        expected = np.sqrt(a.data)
        np.testing.assert_array_equal(result.data, expected)
        self.assertIsNone(result.grad_fn)
