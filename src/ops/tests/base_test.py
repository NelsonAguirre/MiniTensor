import unittest

import numpy as np

from ...device import Device
from ...ops import BaseOps
from ...tensor import Tensor


class TestBaseOps(unittest.TestCase):
    def setUp(self):
        self.tensor = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True, device=Device.CPU)

    def test_reshape(self):
        result = Tensor.from_TProps(BaseOps.reshape(self.tensor, (4,)))
        self.assertEqual(result._data.shape, (4,))
        self.assertEqual(len(result.grad_fn.dependencies), 1)  # type: ignore

    def test_transpose_none(self):
        result = Tensor.from_TProps(BaseOps.transpose(self.tensor))
        self.assertEqual(result._data.shape, (2, 2))

    def test_transpose_axes(self):
        t = Tensor(np.random.rand(2, 3, 4), requires_grad=True)
        result = Tensor.from_TProps(BaseOps.transpose(t, dim=(2, 0, 1)))
        self.assertEqual(result._data.shape, (4, 2, 3))

    def test_squeeze_none(self):
        t = Tensor(np.array([[[1.0], [2.0]]]), requires_grad=True)
        result = Tensor.from_TProps(BaseOps.squeeze(t, dim=None))
        self.assertEqual(result._data.shape, (2,))

    def test_squeeze_axis(self):
        t = Tensor(np.array([[[1.0], [2.0]]]), requires_grad=True)
        result = Tensor.from_TProps(BaseOps.squeeze(t, dim=2))
        self.assertEqual(result._data.shape, (1, 2))

    def test_unsqueeze(self):
        t = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        result = Tensor.from_TProps(BaseOps.unsqueeze(t, dim=1))
        self.assertEqual(result._data.shape, (2, 1))

    def test_bkwd_broadcast_scalar_tensor(self):
        t = Tensor(3.0, requires_grad=True)
        grad = np.array([[1.0, 2.0], [3.0, 4.0]])
        fn = BaseOps.backward_broadcast(t)
        self.assertEqual(fn(grad), grad.sum())

    def test_bkwd_broadcast_scalar_grad(self):
        t = Tensor(np.array([1.0]), requires_grad=True)
        grad = np.array(2.0)
        fn = BaseOps.backward_broadcast(t)
        self.assertEqual(fn(grad), grad)

    def test_bkwd_broadcast_full(self):
        t = Tensor(np.ones((1, 3)), requires_grad=True)
        grad = np.ones((2, 1, 3))
        fn = BaseOps.backward_broadcast(t)
        out = fn(grad)
        self.assertEqual(out.shape, (1, 3))
