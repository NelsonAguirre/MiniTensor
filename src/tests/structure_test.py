import unittest
import numpy as np

from ..tensor import Tensor
from ..device import Device, DType
from ..structure import TensorLike, TProps


class TestTypes(unittest.TestCase):
    def test_tprops_props(self):
        data = np.ones((2, 2))
        tprops = TProps(
            data=data,
            requires_grad=True,
            device=Device.CPU,
            dtype=DType.FLOAT32,
        )
        self.assertEqual(tprops.props(), (data, DType.FLOAT32, True, Device.CPU, None, None))

    def test_tensorlike_conformance(self):
        tensor = Tensor([[1, 2]], requires_grad=False)
        self.assertIsInstance(tensor, TensorLike)
