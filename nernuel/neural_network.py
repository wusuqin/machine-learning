import numpy as np
import math
import tensorflow as tf
import math

from keras.src.distribution import initialize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
from keras.src import ops
from ReluDense import ReluDens
from LinearDens import LinearDens
import math

X_train = tf.Variable([[ 5.55075659,-1.04564419],[ 5.3231534, -2.99183439], [-3.10014381, -3.31564409], [-2.4230153, -0.93551791],
                    [-4.9033676, -0.38310963], [-0.57449011,-2.66475512], [ 0.29859758 ,2.84908785], [-6.72596243, 3.58509537],
                    [ 6.2418555, -0.29225312], [-1.01498214, -3.70046527], [-3.61615283, 2.76038508],[ 2.05069979,  1.46312554],
                    [ 3.83671763, -2.0213351 ],[ 5.7827765, -3.7932379 ],[ 3.41800471, -3.24311418],[-3.42177445, 2.10749794],
                    [-2.28564551, -1.46163252], [ 2.11347211, 0.07883028], [-2.005778,  -2.46747897],[ 0.31077063, 1.14724314],
                    [-5.10069672, 2.30379318], [-6.26405266, 3.52790535], [ 1.81050091, 0.95522163], [-5.76404783, 1.22481149],
                    [ 4.61298353, -3.92730539],[ 0.60213256, 3.01912738],[-4.86570341, 0.89314453],[-5.97071094, 2.47055962],
                    [ 4.58983098, -3.05507534], [-4.08389663, -1.06221829], [ 2.8988813, 1.64515036], [ 0.97080728,  3.39405598]])
y_train = tf.Variable([3,3, 1, 1, 1, 1, 2, 0, 3, 1, 0, 2, 3, 3, 3, 0, 1, 2, 1, 2, 0, 0, 2, 0, 3, 2, 0, 0, 3, 1, 2, 2])
y_output = np.zeros((32, 4))
for i in range(32):
    for j in range(4):
        y_output[i][j] = 1 if j==y_train[i] else 0
w1 = tf.Variable([[0.41752017,-1.1821599], [0.06095338, 1.0677675]])
b1 = tf.Variable([0,0], dtype=tf.float32)
w2 = tf.Variable([[-0.16419482, 0.4498222, 0.28154945, -0.07789922], [-0.8060634, -0.7707381, 0.8299103, -0.30200005]])
b2 = tf.Variable([0,0,0,0], dtype=tf.float32)

beta1 = 0.9
beta2 = 0.999

momentums = [tf.Variable([[0,0], [0,0]], dtype=tf.float32),tf.Variable([0,0], dtype=tf.float32),
             tf.Variable([[0,0,0,0], [0,0,0,0]], dtype=tf.float32), tf.Variable([0,0,0,0], dtype=tf.float32)]

velocities = [tf.Variable([[0,0], [0,0]], dtype=tf.float32),tf.Variable([0,0], dtype=tf.float32),
             tf.Variable([[0,0,0,0], [0,0,0,0]], dtype=tf.float32), tf.Variable([0,0,0,0], dtype=tf.float32)]

def calculate_outputs(outputs):
    loss_out = np.zeros(32)
    for i in range(32):
        loss_out[i] = math.log(outputs[i][y_train[i]])*-1
    return  loss_out

def calculate_gradients(weights, iteration):
    w1 = weights[0]
    b1 = weights[1]
    w2 = weights[2]
    b2 = weights[3]

    dens1 = ReluDens(w1, b1)
    dens2 = LinearDens(w2, b2)
    outputs1 = dens1.outputs(X_train)
    outputs2 = dens2.outputs(outputs1)
    softmax_output = tf.nn.softmax(outputs2)
    loss_out = calculate_outputs(softmax_output)
    print(f"{iteration} step: loss is {np.sum(loss_out)/32}")
    grad2_out = tf.subtract(softmax_output, y_output)*0.03125
    grad2_weights, grad2_bias, grad_ = dens2.calculate_gradients(grad2_out, outputs1)
    grad2_ = tf.Variable(tf.zeros([grad_.shape[0], grad_.shape[1]]))
    for i in range(grad_.shape[0]):
        for j in range(grad_.shape[1]):
            grad2_[i,j].assign(0 if dens1.out[i,j]<0 else grad_[i,j])
    grad1_weights, grad1_bias, _ = dens1.calculate_gradients(grad2_, X_train)
    return [grad1_weights, grad1_bias, grad2_weights, grad2_bias]

def calculate_weights_bias(weights, grads, iteration):
    beta1_power = ops.power(beta1, iteration+1)
    beta2_power=ops.power(beta2, iteration+1)
    alpha = 0.01 * ops.sqrt(1 - beta2_power) / (1 - beta1_power)
    for i  in range(4):
        momentums[i] =momentums[i] + ops.multiply(ops.subtract(grads[i], momentums[i]), 1-beta1)
        velocities[i] = velocities[i] + ops.multiply(ops.subtract(ops.square(grads[i]), velocities[i]), 1 - beta2)
        weights[i] = weights[i] - (ops.divide(ops.multiply(momentums[i], alpha), ops.add(ops.sqrt(velocities[i]), 1e-7) ))
    return weights

weights = [w1,b1,w2,b2]
for j in range(200):
  grads = calculate_gradients(weights, j)
  weights = calculate_weights_bias(weights, grads, j)





