import unittest

import numpy as np

from ..device import Device, DType
from ..tensor import (
    Tensor,
    data_change_to_tensor,
    verify_equal_devices,
    other_change_to_tensor,
    tensor_from_TProps,
    Other_change_Verify_devices_From_TProps,
    Other_change_Verify_devices,
)
from ..structure import TProps


class TestDecorators(unittest.TestCase):

    def test_data_change_to_tensor_tensor_passthrough(self):
        t = Tensor([1.0])
        self.assertIs(data_change_to_tensor(t, device=Device.CPU), t)

    def test_data_change_to_tensor_from_list(self):
        t = data_change_to_tensor([1.0, 2.0], device=Device.CPU)
        self.assertIsInstance(t, Tensor)
        np.testing.assert_array_equal(t.data, [1.0, 2.0])

    def test_other_change_to_tensor_casts_input(self):
        @other_change_to_tensor
        def add(self: Tensor, other: Tensor):
            return self.data + other.data

        a = Tensor([1.0])
        result = add(a, [2.0])
        np.testing.assert_array_equal(result, np.array([3.0]))

    def test_verify_equal_devices_same_device(self):
        @verify_equal_devices
        def add(self: Tensor, other: Tensor):
            return self.data + other.data

        a = Tensor([1.0], device=Device.CPU)
        b = Tensor([2.0], device=Device.CPU)
        result = add(a, b)
        np.testing.assert_array_equal(result, np.array([3.0]))

    def test_verify_equal_devices_raises_on_mismatch(self):
        @verify_equal_devices
        def add(self: Tensor, other: Tensor):
            return self.data + other.data

        a = Tensor([1.0], device=Device.CPU)
        b = Tensor([2.0], device=Device.CPU)
        b.device = Device.CUDA  # Force mismatch manually
        with self.assertRaises(ValueError):
            add(a, b)

    def test_Other_change_Verify_devices_combines_data_and_device(self):
        @Other_change_Verify_devices
        def subtract(self: Tensor, other: Tensor):
            return self.data - other.data

        a = Tensor([5.0])
        result = subtract(a, [2.0])
        np.testing.assert_array_equal(result, np.array([3.0]))

    def test_tensor_from_TProps_wraps_tprops(self):
        @tensor_from_TProps
        def make_tensor():
            return TProps(
                data=np.array([1.0]),
                requires_grad=False,
                device=Device.CPU,
                dtype=DType.FLOAT32,
            )

        t = make_tensor()
        self.assertIsInstance(t, Tensor)
        np.testing.assert_array_equal(t.data, [1.0])

    def test_Other_change_Verify_devices_From_TProps_combines_all(self):
        @Other_change_Verify_devices_From_TProps
        def add_op(self: Tensor, other: Tensor):
            return TProps(
                data=self.data + other.data,
                requires_grad=self.requires_grad or other.requires_grad,
                device=self.device,
                dtype=self.dtype,
            )

        a = Tensor([1.0])
        result = add_op(a, [2.0])
        self.assertIsInstance(result, Tensor)
        np.testing.assert_array_equal(result.data, [3.0])
