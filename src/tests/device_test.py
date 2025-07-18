import unittest
from enum import Enum

import numpy as np
import cupy as cp

from ..device import Device, DType, get_device, get_lib, get_dtype_from_lib


class TestDevice(unittest.TestCase):
    def testget_device_enum(self):
        """Test if Device Enum values are correct."""
        self.assertEqual(Device.CPU, "cpu")
        self.assertEqual(Device.CUDA, "cuda")

    def testget_device_function_valid(self):
        """Test if get_device function returns correct Device enum."""
        self.assertEqual(get_device("cpu"), Device.CPU)
        self.assertEqual(get_device(Device.CPU), Device.CPU)

        from src import device

        if device.CUDA_AVAILABLE:
            self.assertEqual(get_device("cuda"), Device.CUDA)
            self.assertEqual(get_device(Device.CUDA), Device.CUDA)

    def testget_device_function_invalid(self):
        """Test if get_device function raises an error for invalid inputs."""
        with self.assertRaises(RuntimeError):
            get_device("tpu")
        with self.assertRaises(RuntimeError):
            get_device("gpu")
        with self.assertRaises(RuntimeError):
            get_device(123)  # type: ignore

    def testget_lib_function_default(self):
        self.assertIs(get_lib(), np)  # Default is CPU

    def testget_lib_cuda_without_cupy(self):
        from src import device as devmod

        # Simulate CuPy not available
        original_flag = devmod.CUDA_AVAILABLE
        devmod.CUDA_AVAILABLE = False

        with self.assertRaises(RuntimeError):
            devmod.get_device("cuda")

        devmod.CUDA_AVAILABLE = original_flag

    def test_get_dtype_cpu(self):
        for dtype in [Device.CPU, Device.CUDA]:
            lib = get_lib(dtype)
            for enum_dtype in DType:
                self.assertEqual(
                    get_dtype_from_lib(dtype, enum_dtype),
                    getattr(lib, enum_dtype.value),
                )

    def test_get_dtype_invalid(self):
        class FakeDType(Enum):
            UNKNOWN = "unknown"

        with self.assertRaises(ValueError):
            get_dtype_from_lib(Device.CPU, FakeDType.UNKNOWN)  # type: ignore

    def testget_lib_function(self):
        """Test if get_lib function returns the correct module."""
        self.assertIs(get_lib("cpu"), np)

        from src import device

        if device.CUDA_AVAILABLE:
            self.assertIs(get_lib("cuda"), cp)

    def testget_device_function_case_insensitive(self):
        self.assertEqual(get_device("CPU"), Device.CPU)
        from src import device

        if device.CUDA_AVAILABLE:
            self.assertEqual(get_device("CuDa"), Device.CUDA)
