from typing import Optional, Tuple
from src.device import Array, get_lib
from src.structure import Dim, TensorLike, TProps
from src.function import Function_one_dependency, Ctx


class Reduce:
    @staticmethod
    def sum(
        tensor: TensorLike, dim: Optional[Dim] = None, keepdims: bool = False
    ) -> TProps:
        lib = get_lib(tensor.device)
        data = lib.sum(tensor.data, axis=dim, keepdims=keepdims)
        grad_fn = None
        in_graph = False

        if tensor.requires_grad:
            ctx = Ctx()
            ctx.save_other_info("dim", dim)
            ctx.save_other_info("keepdims", keepdims)
            grad_fn = SumBackward(tensor, ctx)
            in_graph = True

        return TProps(
            data, tensor.dtype, tensor.requires_grad, tensor.device, grad_fn, in_graph
        )

    @staticmethod
    def mean(
        tensor: TensorLike, dim: Optional[Dim] = None, keepdims: bool = False
    ) -> TensorLike:
        num_values = tensor.size if dim is None else tensor.data.shape[dim]
        return tensor.sum(dim=dim, keepdims=keepdims) / num_values

    @staticmethod
    def max(
        tensor: TensorLike, dim: Optional[Dim] = None, keepdims: bool = False
    ) -> TProps:
        lib = get_lib(tensor.device)
        data = lib.max(tensor.data, axis=dim, keepdims=keepdims)
        grad_fn = None
        in_graph = False
        if tensor.requires_grad:
            ctx = Ctx()
            ctx.save_for_backward(data)
            ctx.save_other_info("dim", dim)
            ctx.save_other_info("keepdims", keepdims)
            grad_fn = MaxMinBackward(tensor, ctx)
            in_graph = True

        return TProps(
            data, tensor.dtype, tensor.requires_grad, tensor.device, grad_fn, in_graph
        )

    @staticmethod
    def min(
        tensor: TensorLike, dim: Optional[Dim] = None, keepdims: bool = False
    ) -> TProps:
        lib = get_lib(tensor.device)
        data = lib.min(tensor.data, axis=dim, keepdims=keepdims)
        grad_fn = None
        in_graph = False

        if tensor.requires_grad:
            ctx = Ctx()
            ctx.save_for_backward(data)
            ctx.save_other_info("dim", dim)
            ctx.save_other_info("keepdims", keepdims)
            grad_fn = MaxMinBackward(tensor, ctx)
            in_graph = True
        return TProps(
            data, tensor.dtype, tensor.requires_grad, tensor.device, grad_fn, in_graph
        )


class SumBackward(Function_one_dependency):
    def backward(self, grad: Array) -> Tuple[Array]:
        dim = self.ctx.get_other_info("dim")  # type: ignore
        keepdims = self.ctx.get_other_info("keepdims")  # type: ignore
        x = self.dependencies[0]
        x_data = x.data
        lib = get_lib(x.device)
        full_dx_local = lib.ones_like(x_data)
        if dim is None:
            return (full_dx_local * grad,)
        grad_expanded = grad if keepdims else lib.expand_dims(grad, axis=dim)

        return (full_dx_local * grad_expanded,)


class MaxMinBackward(Function_one_dependency):
    def backward(self, grad: Array) -> Tuple[Array]:
        x = self.dependencies[0]
        x_data = x.data
        output = self.ctx.saved_data[0]  # type: ignore
        dim = self.ctx.get_other_info("dim")  # type: ignore
        keepdims = self.ctx.get_other_info("keepdims")  # type: ignore

        lib = get_lib(x.device)

        output_expanded = output if keepdims or dim is None else lib.expand_dims(output, axis=dim)  # type: ignore
        mask = x_data == output_expanded
        num_max = lib.sum(mask, axis=dim, keepdims=keepdims) if dim is not None else lib.sum(mask)  # type: ignore
        grad_expanded = (
            grad if keepdims is True or dim is None else lib.expand_dims(grad, dim)
        )
        return ((mask / num_max) * grad_expanded,)
