from functools import wraps
from typing import Optional, Tuple, Union, final, Callable

from .device import (
    Device,
    DType,
    Array,
    Scalar,
    get_device,
    get_lib,
    get_dtype_from_lib,
)
from .ops import BaseOps, WhereElements, OverloadOps, MathOps, Reduce
from .structure import Data, Dim, Index, Shape, TensorLike, TProps

# from .structure import Self as TensorType
from .function import Function


def data_change_to_tensor(t: Union[Data, "Tensor"], device: Device) -> "Tensor":
    if not isinstance(t, Tensor):
        t = Tensor(t, device=device)  # type: ignore
    return t


def other_change_to_tensor(fn) -> Callable[..., "Tensor"]:
    @wraps(fn)
    def wrapper(self: "Tensor", other: Union[Data, "Tensor"], *args, **kwargs):
        other = data_change_to_tensor(other, self.device)
        return fn(self, other, *args, *kwargs)

    return wrapper


def verify_equal_devices(fn):
    @wraps(fn)
    def wrapper(self: "Tensor", other: "Tensor", *args, **kwargs):
        if self.device != other.device:
            raise ValueError(
                f"Tensors on different devices: {self.device} vs {other.device}"
            )
        return fn(self, other, *args, *kwargs)

    return wrapper


def tensor_from_TProps(fn) -> Callable[..., "Tensor"]:
    @wraps(fn)
    def wrapper(*args, **kwargs) -> "Tensor":
        props = fn(*args, **kwargs)
        return Tensor.from_TProps(props)

    return wrapper


def Other_change_Verify_devices_From_TProps(fn) -> Callable[..., "Tensor"]:
    return other_change_to_tensor(verify_equal_devices(tensor_from_TProps(fn)))


def Other_change_Verify_devices(fn):
    return other_change_to_tensor(verify_equal_devices(fn))


@final
class Tensor(TensorLike):
    float64 = DType.FLOAT64
    float32 = DType.FLOAT32
    int64 = DType.INT64
    int32 = DType.INT32
    int16 = DType.INT16
    int8 = DType.INT8
    bool_ = DType.BOOL

    def __init__(
        self,
        data: Data,
        dtype: Optional[DType] = None,
        requires_grad: bool = False,
        device: Union[Device, str] = Device.CPU,
        grad_fn: Optional[Function] = None,
        in_graph: bool = False,
    ) -> None:

        self._dtype = dtype or self.float32
        self._device = get_device(device)
        self._data = self.build_ndarray(data, self._dtype, self.device)
        self._requires_grad = requires_grad
        self._grad_fn = grad_fn
        self._in_graph = in_graph
        self.grad: Optional[Array] = None

        if requires_grad:
            lib = get_lib(self._device)
            dtype_lib = get_dtype_from_lib(self._device, self._dtype)
            self.grad = lib.zeros_like(self._data, dtype=dtype_lib)
        if in_graph:
            self.in_graph_set()

    @classmethod
    def from_TProps(cls, props: TProps) -> "Tensor":
        return cls(*(props.props()))

    # -------------
    # Core Fields
    # -------------
    @property
    def grad_fn(self) -> Optional[Function]:
        return self._grad_fn

    @grad_fn.setter
    def grad_fn(self, grad_fn):
        self._grad_fn = grad_fn

    @property
    def data(self) -> Array:
        """
        Gets the underlying array of the Tensor.

        Returns:
        Array: The underlying array of the Tensor.
        """
        return self._data

    @data.setter
    def data(self, new_data: Data) -> None:
        r"""
        Sets new data for the Tensor and resets gradients if required.

        Args:
            new_data (Data): The new array to be set as the underlying array of the Tensor.
        """
        self._data = Tensor.build_ndarray(new_data)
        if self.requires_grad:
            self.zero_grad()

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool):
        lib = get_lib(self.device)
        self._requires_grad = value
        if value:
            dtype_lib = get_dtype_from_lib(self._device, self._dtype)
            self.grad = lib.zeros_like(self.data, dtype=dtype_lib) if value else None

    @property
    def device(self) -> Device:
        return self._device

    @device.setter
    def device(self, device: Device):
        self._device = device

    @property
    def dtype(self) -> DType:
        return self._dtype

    @dtype.setter
    def dtype(self, value: DType):
        self._dtype = value
        if value is not None:
            dtype_lib = get_dtype_from_lib(self._device, self._dtype)
            self._data = self._data.astype(dtype_lib)

    @property
    def is_leaf(self) -> bool:
        """
        Checks if the Tensor is a leaf Tensor (required gradient and create for the user) in the computation graph.

        Returns:
            bool: True if the Tensor is a leaf node, False otherwise.
        """
        return self.grad_fn is None and self.requires_grad

    def item(self) -> Scalar:
        """
        Returns the Python scalar value from a tensor with one element.

        Raises:
            ValueError: If the tensor has more than one element.
        """
        if self.data.size != 1:
            raise ValueError(
                f"Cannot convert tensor with shape {self.shape} to a scalar."
            )
        return self.data.item()

    @staticmethod
    def build_ndarray(
        data: Data, dtype: DType = DType.FLOAT32, device: Device = Device.CPU
    ) -> Array:
        r"""
        Builds a numpy o cupy array from the given data.

        Args:
            data (Data): The data to be converted into a array.
            dtype (DType): The desired data type of the array.

        Returns:
            Array: The built numpy or cupy array.
        """
        lib = get_lib(device)
        dtype_lib = get_dtype_from_lib(device, dtype)

        if isinstance(data, TensorLike):
            return data.data
        elif isinstance(data, lib.ndarray):
            return data.astype(dtype_lib)  # type: ignore
        return lib.array(data, dtype=dtype_lib)

    # --- Shape Properties ---

    @property
    def shape(self) -> Shape:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def size(self) -> int:
        return self._data.size

    def __repr__(self) -> str:
        return f"Tensor({self.data}', data from: {'numpy' if self._device is Device.CPU else 'cupy'}{', requires grad' if self.requires_grad else ''}, shape={self.shape}, dtype={self.dtype.value}{', is_leaf' if self.is_leaf else ''}{', in_graph' if self._in_graph else ''})"

    def __str__(self) -> str:
        return self.data.__str__()

    def to(self, device: Union[Device, str], dtype: Optional[DType] = None) -> "Tensor":
        new_device = get_device(device)
        new_dtype = dtype or self.dtype
        new_dtype_impl = get_dtype_from_lib(new_device, new_dtype)

        if new_device == self.device and new_dtype == self.dtype:
            return self

        if new_device != self.device:
            # Explicitly convert CuPy → NumPy using `.get()`
            if self.device == Device.CUDA and new_device == Device.CPU:
                new_data = self.data.get().astype(new_dtype_impl)  # type: ignore
            # NumPy → CuPy should be safe via `cupy.array(...)`
            elif self.device == Device.CPU and new_device == Device.CUDA:
                new_data = get_lib(new_device).array(self.data, dtype=new_dtype_impl)
            else:
                raise RuntimeError(
                    f"Unsupported device transfer: {self.device} -> {new_device}"
                )
        else:
            # Only dtype conversion
            new_data = self.data.astype(new_dtype_impl)

        output_tensor = Tensor(
            data=new_data,
            requires_grad=self.requires_grad,
            device=new_device,
            dtype=new_dtype,
        )  # type: ignore
        output_tensor.grad_fn = self.grad_fn
        return output_tensor

    @property
    def in_graph(self) -> bool:
        return self._in_graph

    def in_graph_set(self):
        """
        Mark the current tensor and its dependent tensors as part of the computational graph.
        """
        self._in_graph = True
        if self.grad_fn:
            for T in self.grad_fn.dependencies:
                T._in_graph = True  # type: ignore
        else:
            raise RuntimeError("The current tensor has no dependencies")

    def zero_grad(self) -> None:
        r"""
        Resets the gradients of the Tensor to zero.
        """
        if not self.requires_grad:
            raise ValueError(
                "zero_grad can't be executed if the Tensor doesn't require gradient."
            )
        self.grad.fill(0.0)  # type: ignore

    def backward(self, grad_output: Optional[Data] = None) -> None:
        """
        Propagates the gradients of all tensors in the computational graph.

        Args:
            grad_output (Data, optional): The gradient of the output tensor. Defaults to None.
        """

        def prepare_grad_output(self, grad_output: Optional[Data]) -> Array:
            lib = get_lib(self.device)
            dtype_lib = get_dtype_from_lib(self.device, self.dtype)
            if grad_output is None:
                grad_output = lib.ones(self.shape, dtype=dtype_lib)
            else:
                if isinstance(grad_output, Tensor) or isinstance(
                    grad_output, lib.ndarray
                ):
                    if grad_output.shape != self.shape:  # type: ignore
                        raise ValueError("Shape of grad not equal")

                    if isinstance(grad_output, Tensor):
                        if grad_output.device is lib:
                            raise ValueError(
                                "The grad and the tensor hasnt equal devices"
                            )
                        grad_output = grad_output.data
                    elif isinstance(grad_output, lib.ndarray):
                        grad_output = grad_output.astype(lib.float32)  # type: ignore
                else:
                    grad_output = lib.ndarray(grad_output)  # type: ignore
            return grad_output

        def order_topo_backward(self) -> Tuple["Tensor", ...]:
            def DFS(u, order_inv, visited, visited_per_DFS) -> None:
                visited.add(id(u))
                if u.grad_fn is not None:

                    for (
                        w
                    ) in (
                        u.grad_fn.dependencies
                    ):  # Recorre todos los adyacentes siguientes del vertice 'u'

                        if id(w) not in visited:
                            visited_per_DFS.add(id(w))
                            DFS(w, order_inv, visited, visited_per_DFS)
                        elif id(w) in visited and id(w) in visited_per_DFS:
                            raise RuntimeError(
                                f"¡There is a cycle! Vertice father: {u}. The vertex {w} has already been visited"
                            )
                order_inv.append(u)

            order_inv = []
            visited = set()
            visited.add(id(self))
            for u in self.grad_fn.dependencies:
                if id(u) not in visited:
                    visited_per_DFS = set()
                    DFS(u, order_inv, visited, visited_per_DFS)
            order_inv.append(self)
            orden = tuple(order_inv[::-1])
            return orden

        # Step 0: Precaution

        if self.grad_fn is None:
            raise RuntimeError("[!] The Tensor doesn't a backpropagation function.")
        if not self.requires_grad:
            raise RuntimeError(
                "'backward()' can only be called Tensors that require gradient."
            )

        # Step 1: The 'grad' of the tensor is prepared and its value is updated
        grad_output = prepare_grad_output(self, grad_output)
        self.grad += grad_output  # El gradiente actual está formado por unos.

        # Step 2: Calculate the traversal order of the tensors.
        order = order_topo_backward(self)

        # Step 3: The tensors are traversed, and for each dependency, its gradient is updated.
        for i, T in enumerate(order):  # 'T' means 'Tensor'
            operation = T.grad_fn
            if T.is_leaf:
                continue
            grad_T = T.grad

            if get_lib(self.device).all(grad_T == 0.0):
                continue  # Dont propagate if all values of grad_T is zero.
            grads_dependencies = operation.backward(grad_T)  # type: ignore
            # Update gradiente for the inputs of the current Tensor.

            assert isinstance(grads_dependencies, tuple), "Grads must be a tuple"
            for i, dependency in enumerate(operation.dependencies):  # type: ignore
                dependency.grad += grads_dependencies[i]  # type: ignore

    @staticmethod
    def randn(
        dims: Dim = (),
        requires_grad=False,
        device: Device = Device.CPU,
    ) -> "Tensor":
        if isinstance(dims, int):
            dims = (dims,)

        data = get_lib(device).random.randn(*dims)
        return Tensor(
            data=data,
            requires_grad=requires_grad,
            device=device,
        )  # type: ignore

    # --- Unary / Structure Methods ---
    @tensor_from_TProps
    def reshape(self, shape: Shape):
        return BaseOps.reshape(self, shape)

    @tensor_from_TProps
    def transpose(self, dim: Optional[Dim] = None):
        return BaseOps.transpose(self, dim)

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    @tensor_from_TProps
    def squeeze(self, dim: Optional[Dim] = None):
        return BaseOps.squeeze(self, dim)

    @tensor_from_TProps
    def unsqueeze(self, dim: int):
        return BaseOps.unsqueeze(self, dim)

    # --- Indexing ---
    @tensor_from_TProps
    def __getitem__(self, index: Index):
        return OverloadOps.get_item(self, index)

    # --- Where Logic ---
    @staticmethod
    @tensor_from_TProps
    def where(condition: "Tensor", a: "Tensor", b: "Tensor") -> "Tensor":
        r"""
        Performs element-wise selection based on a condition.

        This function returns a tensor where each element is taken from 'a' if the corresponding element in 'condition' is True, otherwhise from 'b'.
        Its supports automatics differentation.

        Args:
            condition (Tensor): A boolean tensor with the same shape as 'a' and 'b'.
            a (Tensor): The tensor providing values where 'condition' is True.
            b (Tensor): The tensor providing values where 'condition' is False.

        Returns:
            Tensor: A tensor with the selected values based on the condition.

        Example:
            >>> a = Tensor([1, 2, 3, 4], requires_grad= True)
            >>> b = Tensor([5, 6, 7, 8], requires_grad= True)
            >>> condition = Tensor([True, False, True, False])
            >>> result = Tensor.where(condition, a, b)
            >>> print(result)
            Tensor([1 6 3 8], requires_grad=True)
        """
        return WhereElements.where(condition, a, b)  # type: ignore

    @staticmethod
    @Other_change_Verify_devices_From_TProps
    def maximum(a: "Tensor", b: "Tensor") -> "Tensor":
        return WhereElements.maximum(a, b)  # type: ignore

    @staticmethod
    @Other_change_Verify_devices_From_TProps
    def minimum(a: "Tensor", b: "Tensor") -> "Tensor":
        return WhereElements.minimum(a, b)  # type: ignore

    def threshold(self, threshold: float, value: float) -> "Tensor":
        """
        Replace values below a threshold with a specified value.
        """
        return Tensor.where(self > threshold, self, Tensor(value))  # type: ignore

    @verify_equal_devices
    def masked_fill(self, mask: "Tensor", value: float) -> "Tensor":  # type: ignore
        """
        Replace values based on a boolean mask
        """
        return Tensor.where(mask, Tensor(value), self)  # type: ignore

    def sign(self) -> "Tensor":
        """
        Element-wise sign function
        """
        return Tensor.where(self > 0, Tensor(1), Tensor.where(self < 0, Tensor(-1), Tensor(0)))  # type: ignore

    # --- Comparisons ---
    @Other_change_Verify_devices
    def __eq__(self, other: "Tensor") -> "Tensor":
        """
        Equal to operator (==).

        Creates a boolean tensor where each element is True if the corresponding
        element in self is equal to the corresponding element in other.

        Args:
            other (Data): Another tensor or scalar to compare.

        Returns:
            Tensor: A boolean tensor with the comparison results.
        """
        return Tensor(self.data == other.data, dtype=self.bool_)

    @Other_change_Verify_devices
    def __ne__(self, other: "Tensor") -> "Tensor":
        """
        Not equal to operator (!=).

        Creates a boolean tensor where each element is True if the corresponding
        element in self is not equal to the corresponding element in other.

        Args:
            other (Data): Another tensor or scal, dtype=bar to compare.

        Returns:
            Tensor: A boolean tensor with the comparison results.
        """
        return Tensor(self.data != other.data, dtype=self.bool_)

    @Other_change_Verify_devices
    def __lt__(self, other: "Tensor") -> "Tensor":
        """
        Less than operator (<).

        Creates a boolean tensor where each element is True if the corresponding
        element in self is less than the corresponding element in other.

        Args:
            other (Data): Another tensor or scalar to compare.

        Returns:
            Tensor: A boolean tensor with the comparison results.
        """
        return Tensor(self.data < other.data, dtype=self.bool_)

    @Other_change_Verify_devices
    def __le__(self, other: "Tensor") -> "Tensor":
        """
        Less than or equal to operator (<=).

        Creates a boolean tensor where each element is True if the corresponding
        element in self is less than or equal to the corresponding element in other.

        Args:
            other (Data): Another tensor or scalar to compare.

        Returns:
            Tensor: A boolean tensor with the comparison results.
        """
        return Tensor(self.data <= other.data, dtype=self.bool_)

    @Other_change_Verify_devices
    def __gt__(self, other: "Tensor") -> "Tensor":
        """
        Greater than operator (>).

        Creates a boolean tensor where each element is True if the corresponding
        element in self is greater than the corresponding element in other.

        Args:
            other (Data): Another tensor or scalar to compare.

        Returns:
            Tensor: A boolean tensor with the comparison results.
        """
        return Tensor(self.data > other.data, dtype=self.bool_)

    @Other_change_Verify_devices
    def __ge__(self, other: "Tensor") -> "Tensor":
        """
        Greater than or equal to operator (>=).

        Creates a boolean tensor where each element is True if the corresponding
        element in self is greater than or equal to the corresponding element in other.

        Args:
            other (Data): Another tensor or scalar to compare.

        Returns:
            Tensor: A boolean tensor with the comparison results.
        """
        return Tensor(self.data >= other.data, dtype=self.bool_)

    # --- Unary Math Ops ---
    @tensor_from_TProps
    def log(self) -> "Tensor":
        """
        Computes the natural logarithm (base e) of the tensor's data.
        """
        return MathOps.log(self)  # type:ignore

    @tensor_from_TProps
    def exp(self) -> "Tensor":
        """
        Computes the exp function of the tensor's data.
        """
        return MathOps.exp(self)  # type: ignore

    @tensor_from_TProps
    def tanh(self) -> "Tensor":
        """
        Computes the thanh function of the tensor's data.
        """
        return MathOps.tanh(self)  # type: ignore

    @tensor_from_TProps
    def abs(self) -> "Tensor":
        """
        Computes the element-wise absolute value of the tensor.
        """
        return MathOps.abs(self)  # type: ignore

    @tensor_from_TProps
    def sqrt(self) -> "Tensor":
        return OverloadOps.pow(self, Tensor(0.5))  # type: ignore

    @Other_change_Verify_devices_From_TProps
    def Sqrt(self, other: "Tensor") -> "Tensor":
        return OverloadOps.pow(self, other)  # type: ignore

    # --- Reduction Ops ---
    @tensor_from_TProps
    def sum(self, dim: Optional[Dim] = None, keepdims: bool = False) -> "Tensor":
        """
        Sum of dim.

        Args:
            dim: dimensions to sum.
        """
        return Reduce.sum(self, dim, keepdims)  # type:ignore

    @tensor_from_TProps
    def min(self, dim: Optional[Dim] = None, keepdims: bool = False) -> "Tensor":
        """
        Computes the minimum value along a given dimension.

        Args:
            dim (Dim or None): The dimension along.
            keepdims (bool): Whether to keep the dimensions.

        Returns:
            Tensor: A new tensor with the minimum values.
        """
        return Reduce.min(self, dim, keepdims)  # type: ignore

    @tensor_from_TProps
    def max(self, dim: Optional[Dim] = None, keepdims: bool = False) -> "Tensor":
        """
        Computes the maximum value along a given dimension.

        Args:
            dim (Dim or None): The dimension along.
            keepdims (bool): Whether to keep the dimensions.

        Returns:
            Tensor: A new tensor with the maximum values.
        """
        return Reduce.max(self, dim, keepdims)  # type:ignore

    def mean(self, dim: Optional[Dim] = None, keepdims: bool = False) -> "Tensor":
        """
        Computes the mean value along a given dimension.

        Args:
            dim (Optional[Dim]): The dimension
            keepdims (bool): Whether to keep the dimensions.

        Returns:
            Tensor: A new tensor with the mean values.
        """
        return Reduce.mean(self, dim, keepdims)  # type:ignore

    # --- Binary Ops ---
    @Other_change_Verify_devices_From_TProps
    def __matmul__(self, other: "Tensor") -> "Tensor":
        """
        Overload the '@' operator to perform matrix multiplication between two tensors.

        Args:
            other (Data): Another tensor to multiply.

        Returns:
            Tensor: The result of the matrix multiplication.
        """
        return OverloadOps.matmul(self, other)  # type: ignore

    @Other_change_Verify_devices
    def __imatmul__(self, other: Data) -> "Tensor":
        """
        Overload the '@=' operation to perform in-place matrix multiplication between two tensors.
        WARNING: In-place operations do not track gradients!
            Args:
                other (Data): Another tensor to multiply in-place.

            Returns:
                Tensor: The result of the matrix multiplication.
        """
        if self.in_graph:
            raise ValueError("Tensor is already in graph. Dont use @ in-place!")
        if other.in_graph:  # type: ignore
            raise ValueError("The other Tensor is already in graph. Dont use @ in-place!")  # type: ignore
        self.data = self.data @ other.data  # type: ignore
        return self

    @Other_change_Verify_devices_From_TProps
    def __mul__(self, other: "Tensor") -> "Tensor":
        """
        Overload the '*' operator to perform element-wise tensor multiplication.

        Args:
            other (Data): Another tensor or scalar to multiply.

        Returns:
            Tensor: The result of the element-wise tensor multiplication.
        """
        return OverloadOps.mul(self, other)  # type: ignore

    @Other_change_Verify_devices
    def __rmul__(self, other: "Tensor") -> "Tensor":
        """
        Overload the right-hand '*' operator (other*self).

        Args:
            other (Data): Another tensor or SCALAR to multiply.
            Returns:
            Tensor: The result of the element-wise tensor multiplication.
        """
        return other * self  # type: ignore

    @Other_change_Verify_devices
    def __imul__(self, other: "Tensor") -> "Tensor":
        """
        Overload the '*=' operation to perform in-place element-wise tensor multiplication.
        WARNING: In-place operations do not track gradients!

        Args:
            other (Data): Another tensor or scalar to multiply in-place

        Returns:
            Tensor: The result of the element-wise tensor multiplication.
        """
        if self.in_graph:
            raise ValueError("Tensor already in graph. Dont use mul in-place!")
        if other.in_graph:
            raise ValueError(
                "The other Tensor is already in graph. Dont use mul in-place!"
            )

        self.data = self.data * other.data
        return self

    @Other_change_Verify_devices_From_TProps
    def __add__(self, other: "Tensor") -> "Tensor":
        """
        Overload the '+' operation to perform element-wise tensor addition.

        Args:
            other (Data): The tensor to be added.

        Returns:
            Tensor: The result of the element-wise tensor addition.
        """
        return OverloadOps.add(self, other)  # type: ignore

    @Other_change_Verify_devices
    def __radd__(self, other: "Tensor") -> "Tensor":
        """
        Overload the right-hand '+' operator (other+self).

        Args:
            other (Data): Another tensor or SCALAR to add.

        Returns:
            Tensor: The result of the element-wise tensor addition.
        """
        return other + self  # type:ignore

    @Other_change_Verify_devices
    def __iadd__(self, other: "Tensor") -> "Tensor":
        """
        Overload the '+=' operation to perform in-place element-wise tensor addition.
        WARNING: In-place operations do not track gradients!

        Args:
            other (Data): Another tensor or scalar to add in-place

        Returns:
            Tensor: The result of the element-wise tensor addition.
        """
        if self.in_graph:
            raise ValueError("Tensor is already in graph. Dont use add in-place!")
        if other.in_graph:
            raise ValueError(
                "The other Tensor is already in graph. Dont use add in-place!"
            )

        self.data = self.data + other.data
        return self

    @tensor_from_TProps
    def __neg__(self) -> "Tensor":
        return OverloadOps.neg(self)  # type: ignore

    @Other_change_Verify_devices
    def __sub__(self, other: "Tensor") -> "Tensor":
        """
        Overload the '-' operator to perform element-wise tensor subtraction.
        Uses addition with negation: a-b = a+ (-b).

        Args:
            other (Data): Another tensor or scalar to subtract.

        Returns:
            Tensor: The result of the element-wise tensor subtraction.
        """
        return self + (-other)

    @Other_change_Verify_devices
    def __rsub__(self, other: "Tensor") -> "Tensor":
        """
        Overload the right-hand '-' operator (other-self).

        Args:
            other (Data): Another tensor or SCALAR to subtract.
        Returns:
            Tensor: The result of the element-wise tensor subtraction.
        """
        return other - self

    @Other_change_Verify_devices
    def __isub__(self, other: "Tensor") -> "Tensor":
        """
        Overload the '-=' operation to perform in-place element-wise tensor subtraction.
        WARNING: In-place operations do not track gradients!

        Args:
            other (Data): Another tensor or scalar to subtract in-place

        Returns:
            Tensor: The result of the element-wise tensor subtraction.
        """
        if self.in_graph:
            raise ValueError("Tensor is already in graph. Dont use sub in-place!")
        if other.in_graph:
            raise ValueError(
                "The other Tensor is already in graph. Dont use sub in-place!"
            )

        self.data = self.data - other.data
        return self

    @Other_change_Verify_devices_From_TProps
    def __pow__(self, other: "Tensor") -> "Tensor":
        """
        Overload the '**' operator to perform element-wise tensor power.

        Args:
            other (Data | Scalar): Another tensor or scalar to power.

        Returns:
            Tensor: The result of the element-wise tensor power.
        """
        return OverloadOps.pow(self, other)  # type: ignore

    @Other_change_Verify_devices
    def __rpow__(self, other: "Tensor") -> "Tensor":
        """
        Overload the right-hand '**' operator (other/self).

        Args:
            other (Data): The numerator, which can be a SCALAR or Tensor.

        Returns:
            Tensor: The result of the element-wise tensor power.
        """
        return other**self

    @Other_change_Verify_devices
    def __ipow__(self, other: "Tensor") -> "Tensor":
        """
        Overload the '**=' operation to perform in-place element-wise tensor division.
        WARNING: In-place operations do not track gradients!

        Args:
            other (Data): Another tensor or scalar to power in-place

        Returns:
            Tensor: The result of the element-wise tensor power.
        """
        if self.in_graph:
            raise ValueError("Tensor is already in graph. Dont use division in-place!")
        if other.in_graph:
            raise ValueError(
                "The other Tensor is already in graph. Dont use division in-place!"
            )

        self.data = self.data**other.data
        return self

    @Other_change_Verify_devices
    def __truediv__(self, other: "Tensor") -> "Tensor":
        """
        Overload the right-hand '/' operator.

        Args:
            other (Data): The numerator, which can be a scalar or Tensor.

        Returns:
            Tensor: The result of the element-wise tensor division.
        """
        other.dtype = Tensor.float32
        return self * (other ** (-1))

    @Other_change_Verify_devices
    def __rtruediv__(self, other: "Tensor") -> "Tensor":
        """
        Overload the right-hand '/' operator (other/self).

        Args:
            other (Data): The numerator, which can be a SCALAR or Tensor.

        Returns:
            Tensor: The result of the element-wise tensor division.
        """
        return other / self

    @Other_change_Verify_devices
    def __itruediv__(self, other: "Tensor") -> "Tensor":
        """
        Overload the '/=' operation to perform in-place element-wise tensor division.
        WARNING: In-place operations do not track gradients!

        Args:
            other (Data): Another tensor or scalar to divide in-place

        Returns:
            Tensor: The result of the element-wise tensor division.
        """
        if self.in_graph:
            raise ValueError("Tensor is already in graph. Dont use division in-place!")
        if other.in_graph:
            raise ValueError(
                "The other Tensor is already in graph. Dont use division in-place!"
            )
        self.data = self.data / other.data
        return self
