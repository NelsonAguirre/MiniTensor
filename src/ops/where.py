from src.device import Array, get_lib
from src.structure import TensorLike, TProps
from src.function import Ctx, Grad_Backward, Function


class WhereElements:
    @staticmethod
    def where(condition: TensorLike, a: TensorLike, b: TensorLike) -> TProps:
        assert isinstance(a, TensorLike) and isinstance(
            b, TensorLike
        ), "The arguments must be tensors"
        lib = get_lib(a.device)

        data = lib.where(condition.data, a.data, b.data)
        requires_grad = a.requires_grad or b.requires_grad
        grad_fn = None
        in_graph = False

        if requires_grad:
            dependencies = dict()
            ctx = Ctx()

            if a.requires_grad:
                dependencies["a"] = a
                ctx.save_for_backward(condition)

            if b.requires_grad:
                dependencies["b"] = b
                ctx.save_for_backward(condition)

            if len(dependencies) != 0:
                grad_fn = WhereBackward(dependencies, ctx)
                in_graph = True
        return TProps(
            data=data,
            requires_grad=requires_grad,
            dtype=a.dtype,
            device=a.device,
            grad_fn=grad_fn,
            in_graph=in_graph,
        )

    @staticmethod
    def maximum(a: TensorLike, b: TensorLike) -> TProps:
        r"""
        Apply element-wise max operation: max(a: 'Tensor', b: 'Tensor') -> 'Tensor'
        Returns a Tensor with the result of element-wise maximum.
        """
        return WhereElements.where(a > b, a, b)

    @staticmethod
    def minimum(a: TensorLike, b: TensorLike) -> TProps:
        r"""
        Apply element-wise max operation: min(a: 'Tensor', b: 'Tensor') -> 'Tensor'
        Returns a Tensor with the result of element-wise minimum.
        """
        return WhereElements.where(a > b, b, a)


class WhereBackward(Function):
    def backward(self, grad: Array) -> Grad_Backward:
        grads = []
        ctx = self.ctx
        if not isinstance(ctx, Ctx):
            raise RuntimeError("Ctx not found")

        lib = get_lib(self.dependencies[0].device)
        if "a" in self._dependencies:
            condition = ctx.saved_data[0].data
            grads.append(
                lib.where(condition, grad, 0.0)
            )  # Conserva los elementos de "grad" si se cumple la condición.

        if "b" in self._dependencies:
            condition = ctx.saved_data[0].data
            grads.append(lib.where(condition, 0.0, grad))
        return tuple(grads)
