from ..device import Array, get_lib
from ..structure import TensorLike, TProps
from ..function import Function_one_dependency, Ctx
from typing import Tuple


class MathOps:
    @staticmethod
    def log(tensor: TensorLike) -> TProps:
        lib = get_lib(tensor.device)
        data = lib.log(tensor.data)
        grad_fn = None
        in_graph = False
        if tensor.requires_grad:
            grad_fn = LogBackward(tensor)
            in_graph = True

        return TProps(data, tensor.dtype, tensor.requires_grad, tensor.device, grad_fn, in_graph)

    @staticmethod
    def exp(tensor: TensorLike) -> TProps:
        lib = get_lib(tensor.device)
        data = lib.exp(tensor.data)

        grad_fn = None
        in_graph = False
        if tensor.requires_grad:
            ctx = Ctx()
            ctx.save_for_backward(data)
            grad_fn = ExpBackward(tensor, ctx)
            in_graph = True
        return TProps(data, tensor.dtype, tensor.requires_grad, tensor.device, grad_fn, in_graph)

    @staticmethod
    def tanh(tensor: TensorLike) -> TProps:
        lib = get_lib(tensor.device)
        data = lib.tanh(tensor.data)
        grad_fn = None
        in_graph = False

        if tensor.requires_grad:
            ctx = Ctx()
            ctx.save_for_backward(data)
            grad_fn = TanhBackward(tensor, ctx)
            in_graph = True
        return TProps(data, tensor.dtype, tensor.requires_grad, tensor.device, grad_fn, in_graph)

    @staticmethod
    def abs(tensor: TensorLike) -> TProps:
        lib = get_lib(tensor.device)
        data = lib.abs(tensor.data)
        grad_fn = None
        in_graph = False

        if tensor.requires_grad:
            grad_fn = AbsBackward(tensor)
            in_graph = True

        return TProps(data, tensor.dtype, tensor.requires_grad, tensor.device, grad_fn, in_graph)


class LogBackward(Function_one_dependency):
    def backward(self, grad: Array) -> Tuple[Array]:
        return (grad / self.dependencies[0].data,)


class ExpBackward(Function_one_dependency):
    def backward(self, grad: Array) -> Tuple[Array]:
        output = self.ctx.saved_data[0]  # type: ignore
        return (output * grad,)


class TanhBackward(Function_one_dependency):
    def backward(self, grad: Array) -> Tuple[Array]:
        output = self.ctx.saved_data[0]  # type: ignore
        return ((1 - output**2) * grad,)


class AbsBackward(Function_one_dependency):
    def backward(self, grad: Array) -> Tuple[Array]:
        dependency = self.dependencies[0]
        lib = get_lib(dependency.device)
        data = dependency.data
        return (lib.sign(data) * grad,)
