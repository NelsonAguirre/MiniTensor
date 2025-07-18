from typing import List, Dict, Optional, Tuple, Union, Any
from .device import Array
from .structure import TensorLike, FunctionLike

Grad_Backward = Union[Tuple[Array, 2], Tuple[Array, 1]]


class Ctx:
    def __init__(self):
        self._saved_data: List[Union[TensorLike, Array]] = []
        self._non_data_info: Dict[str, Any] = {}

    def save_for_backward(self, *tensors) -> None:
        """
        Save tensors or arrays for the backward pass.
        """
        for t in tensors:
            if t not in self._saved_data:
                self._saved_data.append(t)

    @property
    def saved_data(self) -> Tuple[Union[TensorLike, Array], ...]:
        """
        Return a tuple of the saved tensors in the order they were saved.
        """
        return tuple(self._saved_data)

    def save_other_info(self, key: str, value):
        self._non_data_info[key] = value

    def get_other_info(self, key: str):
        return self._non_data_info.get(key)


class Function(FunctionLike):
    """
    A base class for backpropagation functions, it defines the connections between input and output tensors.
    """

    def __init__(self, dependencies: Dict[str, TensorLike], ctx: Optional["Ctx"] = None):
        self._dependencies = dependencies
        self.ctx = ctx

    @property
    def dependencies(self) -> Tuple[TensorLike, ...]:
        """
        Returns the dependencies as a tuple.
        """
        return tuple(self._dependencies.values())

    def backward(self, grad: Array) -> Grad_Backward:
        """
        Calculate gradients of the dependencies.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        r"""
        Returns a string representation of the function.
        """
        return f"OperatorBack({self.__class__.__name__}, num_dependencies: {len(self.dependencies)}{', Ctx' if self.ctx is not None else ''})"


class Function_one_dependency(Function):
    def __init__(self, dependency: TensorLike, ctx: Optional["Ctx"] = None):
        self._dependency = dependency
        self.ctx = ctx

    @property
    def dependencies(self) -> Tuple[TensorLike]:
        return (self._dependency,)

    def backward(self, grad: Array) -> Tuple[Array]:  # type: ignore
        raise NotImplementedError
