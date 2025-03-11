import numpy as np
import tensorflow as tf
from keras.src import ops
from tensorflow.python.ops import gen_math_ops
from Dens import Dens
class LinearDens(Dens):
    def __init__(self, wieights, bias):
        super().__init__(wieights, bias)

    def outputs(self, inputs):
        inputs_shape = inputs.shape
        if (inputs_shape[1] != self.weights.shape[0]):
            raise ValueError(
                f"inputs.shape {inputs_shape} can not matmul weights.shape {self.weights.shape}."
            )
        return ops.matmul(inputs, self.weights) + self.bias

    def calculate_gradients(self, grad_out, inputs):
        grad_weights = gen_math_ops.mat_mul(inputs, grad_out, transpose_a=True, grad_b=True)
        grad_bias =  gen_math_ops.mat_mul(tf.ones((32,1), tf.float32), grad_out, transpose_a = True, grad_b = True)
        grad = gen_math_ops.mat_mul(grad_out, self.weights, transpose_b=True, grad_a = True)
        return grad_weights, grad_bias, grad