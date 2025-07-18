import unittest

import numpy as np

from ..device import Device, DType, check_cuda
from ..tensor import Tensor


class TestTensorCoreOps(unittest.TestCase):
    def test_to_returns_self(self):
        t = Tensor([1.0, 2.0], dtype=DType.FLOAT32)
        self.assertIs(t.to(Device.CPU, DType.FLOAT32), t)

    def test_to_change_dtype(self):
        t = Tensor([1.0], dtype=DType.FLOAT32)
        t2 = t.to(Device.CPU, DType.FLOAT64)
        self.assertEqual(t2.dtype, DType.FLOAT64)
        self.assertNotEqual(t2.data.dtype, t.data.dtype)

    def test_zero_and_release_grad(self):
        t = Tensor([1.0, 2.0], requires_grad=True)
        t.grad.fill(42.0)  # type: ignore
        t.zero_grad()
        np.testing.assert_allclose(t.grad, 0.0)  # type: ignore

    def test_backward_scalar_grad(self):
        t = Tensor(3.0, requires_grad=True) + 0  # Se suma para que "grad_fn" no sea None.
        t.backward()
        self.assertEqual(t.grad, 1.0)

    def test_backward_errors(self):
        t = Tensor([1.0, 2.0])
        with self.assertRaises(RuntimeError):
            t.backward()

        t = Tensor([1.0, 2.0], requires_grad=True) + 1
        with self.assertRaises(ValueError):
            t.backward(np.ones((3,)))  # shape mismatch

        if check_cuda():
            import cupy as cp

            t = Tensor([1.0, 2.0], requires_grad=True, device=Device.CPU)
            wrong_device = cp.array([1.0, 2.0])
            with self.assertRaises(ValueError):
                t.backward(wrong_device)

    def test_repr_and_randn(self):
        t = Tensor([1.0])
        self.assertIn("Tensor(", repr(t))
        r = Tensor.randn((2, 2), requires_grad=True)
        self.assertEqual(r.shape, (2, 2))
        self.assertTrue(r.requires_grad)
