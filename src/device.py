from enum import Enum
from typing import Union

import numpy as np

# CUDA is AVAILABLE?

try:
    import cupy as cp

    CUDA_AVAILABLE = cp.cuda.is_available()

except ImportError:
    CUDA_AVAILABLE = False
except Exception as e:
    print(f"Ocurrió un error inesperado al verificar CUDA: '{e}'")
    CUDA_AVAILABLE = False

Scalar = Union[int, float]
Array = Union[np.ndarray, "cp.ndarray"]


class DType(Enum):
    """Enum representing supported data types for tensor valores."""

    FLOAT64 = "float64"
    FLOAT32 = "float32"
    INT64 = "int64"
    INT32 = "int32"
    INT16 = "int16"
    INT8 = "int8"
    BOOL = "bool_"


class Device(str, Enum):
    """Enumeration for supported devices."""

    CPU = "cpu"
    CUDA = "cuda"


def check_cuda() -> bool:
    r"""Check if CuPy is installed and CUDA devices are available.

    Returns:
        bool: True if CuPy is installed and a CUDA device is available, False otherwise.
    """
    return CUDA_AVAILABLE


def get_device(device: Union[Device, str]) -> Device:
    """Validates and returns the appropriate Device enum.

    Args:
        device (Union[Device, str]): The device to validate ("cpu" or "cuda").

    Raises:
        ImportError: If the device string is invalid or if CUDA is requested but unavailable.

    Returns:
        Device: The validated Device enum (Device.CPU or Device.CUDA).
    """
    if isinstance(device, Device):
        return device
    if isinstance(device, str):
        device = device.lower()
    if device not in (Device.CPU.value, Device.CUDA.value):
        raise RuntimeError("Device must be a 'cpu' or 'cuda'")

    if device == Device.CUDA.value and not CUDA_AVAILABLE:
        raise RuntimeError("CUDA operations requested but CuPy is not installed!")

    return Device(device)


def get_lib(device: Union[Device, str] = Device.CPU):
    """Returns the appropriate array library based on the device.

    Args:
        device (Union[Device, str], optional): The selected device. Defaults to 'cpu'.

    Returns:
        module: NumPy if CPU, CuPy if CUDA.
    """
    device = get_device(device)
    return cp if device == Device.CUDA else np  # type:ignore


def get_dtype_from_lib(device: Device, dtype: DType):
    """Returns the correct low-level data type function for a given device.

    Args:
        device (Device): The computational device (e.g., CPU or GPU).
        dtype (DType): The desired data type.

    Raises:
        ValueError: If the dtype is unsupported for the given device.

    Returns:
        Callable: The function or dtype handler corresponding to the device and dtype.
    """
    lib = get_lib(device)

    mapping = {
        DType.FLOAT64: lib.float64,
        DType.FLOAT32: lib.float32,
        DType.INT64: lib.int64,
        DType.INT32: lib.int32,
        DType.INT16: lib.int16,
        DType.INT8: lib.int8,
        DType.BOOL: lib.bool_,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype}' for device '{device}'")
    return mapping[dtype]
