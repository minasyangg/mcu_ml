"""Operator and gradient implementations."""
import sys
sys.path.append('/root/.virtualenvs/jupyter/lecture4/python')
from numbers import Number
from typing import Optional, List
from needle.autograd import NDArray
from needle.autograd import Op, Tensor, Value, TensorOp
from scipy.special import softmax



import needle as ndl

import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        if all([a.ndim == 1, b.ndim == 1]):
            return array_api.divide(a, b)
        if a.ndim == 1:
            a = a.reshape((-1, 1))
        elif b.ndim == 1:
            b = b.reshape((-1, 1))            
        return array_api.divide(a, b)

    def gradient(self, out_grad, node):
        nom, den = node.inputs
        return ndl.divide(out_grad, den), out_grad * ndl.divide(ndl.negate(nom), ndl.power_scalar(den, 2))



def divide(a, b):
  return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return array_api.divide(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return ndl.divide_scalar(out_grad, self.scalar)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
      if self.axes == None:
        return array_api.swapaxes(a, len(a.shape)-1, len(a.shape)-2)
      else:
        return array_api.swapaxes(a, self.axes[0], self.axes[1])

    def gradient(self, out_grad, node):
      input_shape = array_api.array(node.inputs[0].shape)
      out_shape = array_api.array(out_grad.shape)
      axis = tuple(array_api.argwhere(input_shape != out_shape).flat)
      return ndl.transpose(out_grad, axes=axis)



def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION
        

    def gradient(self, out_grad, node):
        a = node.inputs[0]

        if a.ndim == 1 and out_grad.ndim != 1:
            try:
                grad = ndl.summation(out_grad, axes=1)
            except ValueError:
                print(f"GRAD_RESHAPE: out_grad: {out_grad.shape} node: {a.shape}")
                 

        elif a.ndim == out_grad.ndim:
            try:
                grad = ndl.reshape(out_grad, a.shape)
            except ValueError:
                print(f"GRAD_RESHAPE: out_grad: {out_grad.shape} node: {a.shape}")

        if grad.shape == a.shape:
            return grad
        else:
            print("fff")
            print(f"node_shape {node.shape} != grad_shape {grad.shape}")
        
       


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        value = None
        try:
            value = array_api.broadcast_to(a, self.shape)
        except ValueError:
            value = array_api.broadcast_to(a.T, self.shape)

        if value is not None:
            return value
        else: print("BROADCASTING: error")

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        grad = None
        if a.ndim == 0:
            return out_grad
        else:
            inp_shape = node.inputs[0].shape
            if len(inp_shape) == len(out_grad.shape) and len(inp_shape) == 2:
                return ndl.reshape(ndl.summation(out_grad, 1), inp_shape)
            elif len(inp_shape) == len(out_grad.shape) and len(inp_shape) == 3 and inp_shape[-1] != out_grad.shape[-1]:
                return ndl.reshape(ndl.summation(out_grad, 2), inp_shape)
            elif len(inp_shape) < len(out_grad.shape):
                return ndl.reshape(ndl.summation(out_grad), inp_shape)
          


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        grad = None
        if out_grad.ndim == 0 and a.ndim == 0:
            grad = out_grad
        if out_grad.ndim == 1 and  a.ndim != 1:
            try:
                out_grad = out_grad.reshape((-1, 1))
            except ValueError:
                print(f'SUMMATION grad err: out_grad: {out_grad.shape} node: {a.shape}')  
        if out_grad.ndim == 0 and a.ndim != 0:
            try:
                grad = ndl.broadcast_to(out_grad, a.shape)
            except ValueError:
                print(f'SUMMATION grad err: out_grad: {out_grad.shape} node: {a.shape}')
        elif out_grad.ndim == 2 and a.ndim == 2:  
            try:
                grad = ndl.broadcast_to(out_grad, a.shape)
            except ValueError:
                print(f'SUMMATION grad err: out_grad: {out_grad.shape} node: {a.shape}')
        elif a.ndim == 3 and out_grad.ndim != a.ndim:
            try:
                grad = ndl.broadcast_to(ndl.summation(out_grad, axes = 1), a.shape)
            except ValueError:
                print(f'SUMMATION grad err: out_grad: {out_grad.shape} node: {a.shape}')

        if grad == None:
            print('SUMMATION: grad is None')
        elif grad.shape == a.shape:
            return grad
        elif grad.shape != a.shape:
            print(f"{grad.shape} != {a.shape}")
        else:
            print('SUMMATION grad err')
            
        

def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # if a.shape == b.shape:
        #     return array_api.matmul(a, array_api.transpose(b))
        # else:
        try:
            return array_api.matmul(a, b)
        except ValueError:
            return array_api.matmul(a, array_api.transpose(b))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        grads = []
        a, b = node.inputs

        if out_grad.ndim == b.ndim:
            if out_grad.shape[-1] == b.shape[0]:
                try:
                    first_grad = ndl.matmul(out_grad, b)
                except ValueError:
                    print(f'MATNUL: out_grad_shape = {out_grad.shape}')
                    print(f'MATNUL: B_shape = {b.shape}')
            elif any([x in out_grad.shape for x in b.shape]) or any([x in b.shape for x in out_grad.shape]):
                try:
                    first_grad = ndl.matmul(out_grad, ndl.transpose(b))
                except ValueError:
                    first_grad = ndl.matmul(ndl.traspose(out_grad), b)
            else:
                print(f'MATNUL: out_grad_shape = {out_grad.shape}')
                print(f'MATNUL: B_shape = {b.shape}')

        if out_grad.ndim == a.ndim:
            if out_grad.shape[-1] == a.shape[0]:
                try:
                    second_grad = ndl.matmul(out_grad, a)
                except ValueError:
                    print(f'MATNUL: out_grad_shape = {out_grad.shape}')
                    print(f'MATNUL: A_shape = {a.shape}')
            elif any([x in out_grad.shape for x in a.shape]) or any([x in a.shape for x in out_grad.shape]):
                try:
                    second_grad = ndl.matmul(out_grad, ndl.transpose(a))
                except ValueError:
                    second_grad = ndl.matmul(ndl.transpose(out_grad), a)
            else:
                print(f'MATNUL: out_grad_shape = {out_grad.shape}')
                print(f'MATNUL: A_shape = {a.shape}')

        if first_grad.shape == a.shape:
            grads.append(first_grad)
        elif all([x in first_grad.shape for x in a.shape]):
            grads.append(ndl.transpose(first_grad))
            
        else:
            print(f"MATNUL: first_grad's shape {second_grad.shape} != {a.shape}")

        if second_grad.shape == b.shape:
            grads.append(second_grad)
        elif all([x in second_grad.shape for x in b.shape]):
            grads.append(ndl.transpose(second_grad))
        else:
            print(f"MATNUL: second_grad's shape {second_grad.shape} != {b.shape}")
        
        if None not in grads:
            return grads
        else:
            print('MATNUL: None in {grads}')

        # if len(lhs.shape) <=2 and len(rhs.shape) <= 2:
        #   if lhs.shape[0] > rhs.shape[0]:
        #     return out_grad @ ndl.transpose(rhs), out_grad @ lhs
        #   elif lhs.shape[0] < rhs.shape[0]:
        #     return out_grad @ ndl.transpose(rhs), ndl.transpose(out_grad @ lhs)
        #   elif lhs.shape[0] ==  out_grad.shape[1] and rhs.shape[0] == out_grad.shape[1]:
        #     return ndl.broadcast_to((out_grad @ rhs), node.shape), out_grad @ lhs
        # elif len(lhs.shape) > 2 and len(rhs.shape) > 2 and len(lhs.shape) == len(rhs.shape):
        #   return out_grad @ ndl.transpose(rhs, axes=(2, 3)), ndl.transpose(out_grad, axes=(2, 3)) @ lhs
        # elif len(lhs.shape) != len(rhs.shape):
        #   if len(lhs.shape) > len(rhs.shape) and lhs.shape[-1] == rhs.shape[-2]:
        #     return out_grad @ ndl.transpose(rhs), ndl.transpose(out_grad, axes=(2, 3)) @ lhs
        #   elif len(lhs.shape) < len(rhs.shape) and lhs.shape[-1] == rhs.shape[-2] and len(lhs.shape) == 2 and len(rhs.shape) == 3:
        #     grad_l = ndl.matmul(out_grad, ndl.transpose(rhs, axes=(1, 2)))
        #     grad_r = ndl.matmul(out_grad, lhs)
        #     return ndl.summation(grad_l, axes=0), ndl.transpose(grad_r, axes=(1, 2))
        #   elif len(lhs.shape) == 3 and len(rhs.shape) == 4:
        #     return ndl.summation(ndl.matmul(out_grad, ndl.transpose(rhs, axes=(2,3))), axes=0), ndl.transpose(ndl.matmul(out_grad, ndl.broadcast_to(lhs, (3,3,2,1))), axes=(2,3))
        #   elif len(lhs.shape) == 2 and len(rhs.shape) == 4 and lhs.shape[-1] == rhs.shape[-2]:
        #     return ndl.summation(ndl.matmul(out_grad, ndl.transpose(rhs, axes=(2,3))), axes=(0,1)), ndl.transpose(ndl.matmul(out_grad, lhs), axes=(2,3))



def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
      return array_api.negative(a)
        

    def gradient(self, out_grad, node):
        return ndl.mul_scalar(out_grad, -1) 



def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        grad = ndl.divide(out_grad, a)

        if grad.shape != a.shape:
            print(f'LOG: {grad.shape} != {a.shape}')

        else: return grad

def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0]

        # grad = ndl.matmul(out_grad, ndl.exp(a))
        grad = out_grad * ndl.exp(a)

        if grad.shape != a.shape:
            print(out_grad.shape, a.shape)
            print(f'EXP: {grad.shape} != {a.shape}')
        else: return grad



def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):
    def compute(self, a):
        return array_api.where(a > 0, 1, 0)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        grad = a - a 

        return grad



def relu(a):
    return ReLU()(a)





class Softmax(TensorOp):
    def compute(self, a, axes):
        self.axes = axes
        return softmax(a, axes=axes)
        
    def gradient(self, out_grad, node):
        return out_grad



def softmax(a, axes=None):
    return Softmax()(a, axes)


