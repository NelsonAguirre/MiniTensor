from src.structure import Index, TensorLike, TProps
from src.function import Ctx, Function, Function_one_dependency, Grad_Backward
from src.device import Array, get_lib, get_dtype_from_lib

from src.ops.base import BaseOps

from typing import Tuple, Callable, Union


class OverloadOps:
    @staticmethod
    def get_item(tensor: TensorLike, index: Index):
        r"""
        Perform indexing on the tensor.

        This method allows for selecting specific elements from the tensor using another tensor or a NumPy array as an index.

        Args:
            index (IndexType): The index or indices to select from the tensor.

        Returns:
            Tensor: A new tensor with the selected elements.

        Example:
            >>> a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
            >>> indices = Tensor([0, 2])
            >>> result = a[0][indices]
            >>> print(result)
            Tensor([1,3], requires_grad=True, shape=(2,))
        """

        if isinstance(index, tuple):
            idx = []
            for i in index:
                if isinstance(i, TensorLike):
                    idx.append(i.data)
                else:
                    idx.append(i)
            index = tuple(idx)

        if isinstance(index, TensorLike):
            index = index.data

        data = tensor.data[index]
        requires_grad = tensor.requires_grad
        grad_fn = None
        in_graph = False

        if requires_grad:
            ctx = Ctx()
            ctx.save_other_info("index", index)
            grad_fn = GetItemBackward(tensor, ctx)
            in_graph = True

        return TProps(
            data=data,
            dtype=tensor.dtype,
            device=tensor.device,
            grad_fn=grad_fn,
            in_graph=in_graph,
        )

    @staticmethod
    def neg(tensor: TensorLike) -> TProps:
        """
        Overload the unary '-' operator to negate a tensor.
        This allows defining substraction as addition with negation.
        """

        data = -tensor.data
        requires_grad = tensor.requires_grad
        grad_fn = None
        in_graph = False
        if requires_grad:
            grad_fn = NegBackward(tensor)
            in_graph = True
        return TProps(
            data=data,
            dtype=tensor.dtype,
            device=tensor.device,
            grad_fn=grad_fn,
            in_graph=in_graph,
        )

    @staticmethod
    def add(a: TensorLike, b: TensorLike) -> TProps:
        """
        Adds two tensors element-wise.

        Args:
            a (Tensor): The first tensor.
            b (Tensor): The second tensor.

        Returns:
            Tensor: The result of the element-wise addition.
        """

        data = a.data + b.data
        requires_grad = a.requires_grad or b.requires_grad

        grad_fn = None
        in_graph = False
        if requires_grad:
            dependencies = dict()
            if a.requires_grad:
                dependencies["a"] = a
            if b.requires_grad:
                dependencies["b"] = b
            if len(dependencies) != 0:
                grad_fn = AddBackward(dependencies)
                in_graph = True
        return TProps(
            data=data,
            dtype=a.dtype,
            requires_grad=requires_grad,
            grad_fn=grad_fn,
            in_graph=in_graph,
        )

    @staticmethod
    def mul(a: TensorLike, b: TensorLike) -> TProps:
        """
        Performs element-wise multiplications between two tensors and resturns the result.
        Handles tensors that require gradient by defining the backward for backpropagation.

        Args:
            a (Tensor): The first tensor to be multiplied.
            b (Tensor): The second tensor to be multiplied.

        Returns:
            Tensor: The tensor result of the element-wise multiplication.
        """
        data = a.data * b.data
        requires_grad = a.requires_grad or b.requires_grad

        grad_fn = None
        in_graph = False
        if requires_grad:
            dependencies = dict()
            ctx = Ctx()

            if a.requires_grad:
                dependencies["a"] = a
                ctx.save_other_info("b_data", b.data)
            if b.requires_grad:
                dependencies["b"] = b
                ctx.save_other_info("a_data", a.data)

            if len(dependencies) != 0:
                grad_fn = MulBackward(dependencies, ctx)
                in_graph = True
        return TProps(
            data=data, requires_grad=requires_grad, grad_fn=grad_fn, in_graph=in_graph
        )

    @staticmethod
    def pow(a: TensorLike, b: TensorLike) -> TProps:
        r"""
        Computes the element-wise power of the tensor's data.

        The operation applies the power function element-wise, raising each element in the tnesor to the given power 'pow'

        Args:
            a (Tensor): The tensor whose data will be raised to the power.
            b (Tensor): The exponent to which each element in the tensor will be raised.

        Returns:
            Tensor: A new tensor with the element-wise power
        """
        data = a.data**b.data
        requires_grad = a.requires_grad or b.requires_grad
        grad_fn = None
        in_graph = False

        if requires_grad:
            dependencies = dict()
            ctx = Ctx()
            if a.requires_grad:
                dependencies["a"] = a
                ctx.save_other_info("b_data", b.data)
            if b.requires_grad:
                dependencies["b"] = b
                ctx.save_other_info("a_data", a.data)
            if len(dependencies) != 0:
                grad_fn = PowBackward(dependencies, ctx)
                in_graph = True

        return TProps(
            data=data,
            dtype=a.dtype,
            device=a.device,
            requires_grad=requires_grad,
            grad_fn=grad_fn,
            in_graph=in_graph,
        )

    @staticmethod
    def matmul(a: TensorLike, b: TensorLike) -> TProps:
        """
        Performs matrix multiplication between two tensors and resturns the result.

        Args:
            a (Tensor): The first tensor to be multiplied.
            b (Tensor): The second tensor to be multiplied.

        Returns:
            Tensor: The tensor result of the matrix multiplication.
        """

        data = a.data @ b.data
        requires_grad = a.requires_grad or b.requires_grad

        grad_fn = None
        in_graph = False
        if requires_grad:
            dependencies = dict()
            ctx = Ctx()
            if a.requires_grad:
                dependencies["a"] = a
                ctx.save_other_info("b_data", b.data)
            if b.requires_grad:
                dependencies["b"] = b
                ctx.save_other_info("a_data", a.data)
            if len(dependencies) != 0:
                grad_fn = MatmulBackward(dependencies, ctx)
                in_graph = True
        return TProps(data, a.dtype, requires_grad, a.device, grad_fn, in_graph)


class GetItemBackward(Function_one_dependency):
    def backward(self, grad: Array) -> Tuple[Array]:
        """
        Backward pass for tensor indexing.

        This function construct a zero tensor of the same shape as teh original
        tensor and assigns (with np.add.at) the incoming gradient only to the indexed positions.

        Args:
          grad (np.ndarray): The incoming gradient.

        Returns:
          np.ndarray: Gradient propagated back to the indexed positions.
        """
        if not isinstance(self.ctx, Ctx):
            raise RuntimeError("")
        index: Index = self.ctx.get_other_info("index")
        a = self.dependencies[0]
        lib = get_lib(a.device)
        dtype = get_dtype_from_lib(a.device, a.dtype)

        full_grad = lib.zeros_like(a.data, dtype=dtype)
        full_grad[index] = grad
        full_grad = full_grad

        return (full_grad,)


class NegBackward(Function_one_dependency):
    def backward(self, grad: Array) -> Tuple[Array]:
        return (-grad,)


class AddBackward(Function):
    def backward(self, grad: Array) -> Grad_Backward:
        grads = []
        if "a" in self._dependencies:
            a = self._dependencies["a"]

            backward_function = BaseOps.backward_broadcast(a)
            grads.append(backward_function(grad))
        if "b" in self._dependencies:
            b = self._dependencies["b"]
            backward_function = BaseOps.backward_broadcast(b)
            grads.append(backward_function(grad))
        return tuple(grads)


class MulBackward(Function):
    def backward(self, grad: Array) -> Grad_Backward:
        if not isinstance(self.ctx, Ctx):
            raise RuntimeError("not found ctx")
        grads = []
        ctx = self.ctx
        if "a" in self._dependencies:
            a_data = self._dependencies["a"].data
            b_data = ctx.get_other_info("b_data")
            a_grad_extended = grad * b_data
            backward_function = BaseOps.backward_broadcast(a_data)
            grads.append(backward_function(a_grad_extended))

        if "b" in self._dependencies:
            b_data = self._dependencies["b"].data
            a_data = ctx.get_other_info("a_data")
            b_grad_extended = grad * a_data
            backward_function = BaseOps.backward_broadcast(b_data)
            grads.append(backward_function(b_grad_extended))
        return tuple(grads)


class PowBackward(Function):
    def backward(self, grad: Array) -> Grad_Backward:
        if not isinstance(self.ctx, Ctx):
            raise RuntimeError("Not found Ctx")
        grads = []
        ctx = self.ctx
        if "a" in self._dependencies:
            a_data = self._dependencies["a"].data
            b_data = ctx.get_other_info("b_data")
            a_grad_extended = grad * b_data * (a_data ** (b_data - 1))  # type: ignore
            backward_function = BaseOps.backward_broadcast(a_data)
            grads.append(backward_function(a_grad_extended))

        if "b" in self._dependencies:
            b_data = self._dependencies["b"].data
            a_data = ctx.get_other_info("a_data")
            lib = get_lib(self.dependencies[0].device)
            b_grad_extended = grad * (a_data**b_data) * lib.log(a_data)  # type: ignore
            backward_function = BaseOps.backward_broadcast(b_data)
            grads.append(backward_function(b_grad_extended))
        return tuple(grads)


class MatmulBackward(Function):
    def backward_broadcast_matmul(
        self, tensor: Union[TensorLike, Array]
    ) -> Callable[[Array], Array]:
        def _backward(grad: Array) -> Array:
            # 1: Calculamos las dimensiones "añadidas"
            ndim_added = max(0, grad.ndim - tensor.ndim)
            if ndim_added > 0:
                grad = grad.sum(axis=tuple(range(ndim_added)), keepdims=False)
            # 2: Calculamos las dimesiones "expandidas", para las dimensiones de los batches.
            reduce_axes = tuple(
                dim
                for dim in range(tensor.ndim - 2)
                if tensor.shape[dim] == 1 and grad.shape[dim] > 1
            )
            if reduce_axes:
                grad = grad.sum(axis=reduce_axes, keepdims=False)
            # 3 Asegurarnos que el gradiente final tenga la misma forma que el tensor.
            if grad.shape != tensor.shape:
                grad = grad.reshape(tensor.shape)
            return grad

        return _backward

    def backward(self, grad: Array) -> Grad_Backward:
        if not isinstance(self.ctx, Ctx):
            raise RuntimeError("not found ctx")
        grads = []
        ctx = self.ctx
        if "a" in self._dependencies:
            a = self._dependencies["a"]
            a_data = a.data
            b_data = ctx.get_other_info("b_data")
            if a_data.ndim == 1 and b_data.ndim == 1:  # type: ignore
                grads.append(b_data * grad)
            else:
                lib = get_lib(a.device)
                a_grad_extended = (
                    grad @ b_data.swapaxes(-1, -2)  # type:ignore
                    if b_data.ndim > 1  # type:ignore
                    else lib.outer(grad, b_data)  # type: ignore
                )
                backward_function = self.backward_broadcast_matmul(a_data)
                grads.append(backward_function(a_grad_extended))

        if "b" in self._dependencies:
            b_data = self._dependencies["b"].data
            a_data = ctx.get_other_info("a_data")
            if a_data.ndim == 1 and b_data.ndim == 1:  # type: ignore
                grads.append(a_data * grad)
            else:
                b_grad_extended = a_data.swapaxes(-1, -2) @ grad  # type: ignore
                backward_function = self.backward_broadcast_matmul(b_data)
                grads.append(backward_function(b_grad_extended))
        return tuple(grads)
