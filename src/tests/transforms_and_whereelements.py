import unittest

import numpy as np

from ..tensor import Tensor


class TestTensorTransformElementwiseOps(unittest.TestCase):
    def test__reshape(self):
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        r = t.reshape((4,))
        np.testing.assert_array_equal(r.data, [1.0, 2.0, 3.0, 4.0])

    def test_transpose_and_T(self):
        t = Tensor([[1.0, 2.0], [3.0, 4.0]])
        tr = t.transpose()
        tT = t.T
        np.testing.assert_array_equal(tr.data, [[1.0, 3.0], [2.0, 4.0]])
        np.testing.assert_array_equal(tT.data, tr.data)

    def test_squeeze_unsqueeze(self):
        t = Tensor(np.array([[[1.0], [2.0]]]))
        sq = t.squeeze()
        us = sq.unsqueeze(2)
        self.assertEqual(sq.shape, (2, 1))
        self.assertEqual(us.shape, (2, 1, 1))

    def test_where(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([3.0, 2.0, 1.0])
        mask = Tensor([1.0, 0.0, 1.0])
        w = Tensor.where(mask, a, b)
        np.testing.assert_array_equal(w.data, [1.0, 2.0, 3.0])

    def test_maximum_minimum(self):
        a = Tensor([1.0, 5.0])
        b = Tensor([3.0, 2.0])
        mx = Tensor.maximum(a, b)
        mn = Tensor.minimum(a, b)
        np.testing.assert_array_equal(mx.data, [3.0, 5.0])
        np.testing.assert_array_equal(mn.data, [1.0, 2.0])

    def test_abs(self):
        t = Tensor([-1.0, 2.0, -3.0])
        out = t.abs()
        np.testing.assert_array_equal(out.data, [1.0, 2.0, 3.0])

    def test_threshold(self):
        t = Tensor([0.5, 1.5, 0.2])
        out = t.threshold(1.0, 0.0)
        np.testing.assert_array_equal(out.data, [0.0, 1.5, 0.0])

    def test_masked_fill(self):
        t = Tensor([10.0, 20.0, 30.0])
        mask = Tensor([1.0, 0.0, 1.0])
        out = t.masked_fill(mask, 99.0)
        np.testing.assert_array_equal(out.data, [99.0, 20.0, 99.0])

    def test_sign(self):
        t = Tensor([-3.0, 0.0, 2.0])
        out = t.sign()
        np.testing.assert_array_equal(out.data, [-1.0, 0.0, 1.0])
