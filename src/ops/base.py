from typing import Callable, Tuple, Union, Optional

from src.device import Array, get_lib
from src.structure import Dim, Shape, TProps, TensorLike
from src.function import Ctx, Function_one_dependency


class BaseOps:
    @staticmethod
    def backward_broadcast(
        tensor: Union[Array, TensorLike],
    ) -> Callable[[Array], Array]:
        r"""
        Backward closure function to sum across broadcasted dimenesions.

        When performing operations between tensor of different shapes, broadcasting is used to align their shapes. This function ensures
        that gradient are correctly summed over the broadcasted dimensions during the backward pass.

        Args:
            tensor (Tensor): The tensor involved in the operation, used to handle its shape during backward gradient compution.

        Returns:
            _backward (function): A function that computes the gradient, summing over broadcasted dimensions to match the original tensor's shape.
        """

        def _backward(grad: Array) -> Array:
            """
            Backward function that calculates the gradient of the tensor's shape.

            Args:
            grad (Array): The global gradient in the extend version of the tensor.

            Returns:
            Array: The gradient of the tensor's shape.
            """
            # Caso 1: El tensor es un escalar. En este caso el gradiente debería ser un escalar también, para ello sumamos todos los elementos. En caso de ya ser un escalar la suma no le hará efecto.
            if tensor.ndim == 0:
                return grad.sum()
            # Caso 2: El gradiente es un escalar. Si el gradiente es un escalar, entonces no ha habido broadcast. El broadcast siempre añade dimensiones, no las elimina
            if grad.ndim == 0:
                return grad
            # Caso 3: Hay dimensiones expandidas o creadas.
            # 3.1: Calculamos las dimensiones "añadidas"
            ndim_added = max(0, grad.ndim - tensor.ndim)
            if ndim_added > 0:
                grad = grad.sum(axis=tuple(range(ndim_added)), keepdims=False)
            # 3.2: Calculamos las dimesiones "expandidas"
            reduce_axes = tuple(
                dim
                for dim in range(tensor.ndim)
                if tensor.shape[dim] == 1 and grad.shape[dim] > 1
            )
            if reduce_axes:
                grad = grad.sum(axis=reduce_axes, keepdims=False)
            # 3.3 Asegurarnos que el gradiente final tenga la misma forma que el tensor.
            if grad.shape != tensor.shape:
                grad = grad.reshape(tensor.shape)
            return grad

        return _backward

    @staticmethod
    def reshape(tensor: TensorLike, shape: Shape) -> TProps:
        r"""
        Reshapes the tensor to the specified 'shape'.

        Args:
            shape: The new shape of the tensor.

        Returns:
            tensor: A tensor with the specified shape.
        """

        data = tensor.data.reshape(shape)
        requires_grad = tensor.requires_grad
        grad_fn = None
        in_graph = False
        dtype = tensor.dtype

        if requires_grad:
            ctx = Ctx()
            ctx.save_other_info("shape", tensor.shape)
            grad_fn = ReshapeBackward(tensor, ctx)
            grad_fn = grad_fn
            in_graph = True
        return TProps(
            data=data,
            dtype=dtype,
            requires_grad=requires_grad,
            device=tensor.device,
            grad_fn=grad_fn,
            in_graph=in_graph,
        )

    @staticmethod
    def transpose(tensor: TensorLike, dim: Optional[Dim] = None) -> TProps:
        lib = get_lib(tensor.device)

        data = lib.transpose(tensor.data, dim)
        requires_grad = tensor.requires_grad
        grad_fn = None
        in_graph = False

        if requires_grad:
            ctx = Ctx()
            ctx.save_other_info("dim", dim)
            grad_fn = TransposeBackward(tensor, ctx)
            in_graph = True
        return TProps(
            data=data,
            dtype=tensor.dtype,
            requires_grad=requires_grad,
            device=tensor.device,
            grad_fn=grad_fn,
            in_graph=in_graph,
        )

    @staticmethod
    def squeeze(tensor, dim: Optional[Dim] = None) -> TProps:
        r"""
        Removes dimensions of size 1 from the specified 'dim'. When 'dim' is None its remove all dimensiones of size 1 of the tensor.

        Args:
            dim (None, int or Tuple[int], optional): The dimensions to remove. Defaults to None.

        Returns:
            Tensor: A tensor with the specified dimensions removed.
        """
        lib = get_lib(tensor.device)

        data = lib.squeeze(tensor.data, dim)
        requires_grad = tensor.requires_grad
        grad_fn = None
        in_graph = False

        if requires_grad:
            ctx = Ctx()
            ctx.save_other_info("dim", dim)
            grad_fn = SqueezeBackward(tensor, ctx)
            in_graph = True
        return TProps(
            data=data,
            dtype=tensor.dtype,
            requires_grad=requires_grad,
            device=tensor.device,
            grad_fn=grad_fn,
            in_graph=in_graph,
        )

    @staticmethod
    def unsqueeze(tensor, dim: Dim) -> TProps:
        r"""
        Adds dimensions of size 1 to the specified dims.

        Args:
            dim (int or Tuple[int]): The dimensions to add.

        Returns:
            Tensor: A tensor with the specified dimensions added.
        """
        lib = get_lib(tensor.device)

        data = lib.expand_dims(tensor.data, dim)
        requires_grad = tensor.requires_grad
        grad_fn = None
        in_graph = False
        if requires_grad:
            ctx = Ctx()
            ctx.save_other_info("dim", dim)
            grad_fn = UnsqueezeBackward(tensor, ctx)
            in_graph = True
        return TProps(
            data=data,
            dtype=tensor.dtype,
            requires_grad=requires_grad,
            grad_fn=grad_fn,
            in_graph=in_graph,
        )


class ReshapeBackward(Function_one_dependency):
    def backward(self, grad: Array) -> Tuple[Array]:
        if not isinstance(self.ctx, Ctx):
            raise RuntimeError("Undefine ctx")
        shape = self.ctx.get_other_info("shape")
        return (grad.reshape(shape),)


class TransposeBackward(Function_one_dependency):
    def backward(self, grad: Array) -> Tuple[Array]:
        if not isinstance(self.ctx, Ctx):
            raise RuntimeError("Undefine ctx")
        dim = self.ctx.get_other_info("dim")
        lib = get_lib(self.dependencies[0].device)

        if dim is None:
            return (lib.transpose(grad),)
        else:
            inv_axes = tuple(lib.argsort((dim)))
            return (lib.transpose(grad, inv_axes),)


class SqueezeBackward(Function_one_dependency):
    def backward(self, grad: Array) -> Tuple[Array]:
        if not isinstance(self.ctx, Ctx):
            raise RuntimeError("Undefine ctx")
        dim = self.ctx.get_other_info("dim")
        lib = get_lib(self.dependencies[0].device)
        if dim is None:
            return (grad.reshape(self.dependencies[0].data.shape),)
        return (lib.expand_dim(grad, dim),)  # type: ignore


class UnsqueezeBackward(Function_one_dependency):
    def backward(self, grad: Array) -> Tuple[Array]:
        if not isinstance(self.ctx, Ctx):
            raise RuntimeError("Undefine ctx")
        lib = get_lib(self.dependencies[0].device)

        dim = self.ctx.get_other_info("dim")
        return (lib.squeeze(grad, axis=dim),)
