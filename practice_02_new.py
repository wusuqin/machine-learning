import numpy as np
import math
import tensorflow as tf
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from autils import *
from lab_utils_multiclass_TF import *
from sklearn.datasets import make_blobs
from tensorflow.keras.activations import linear, relu, sigmoid

centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
m = 32
std = 1.0
#X_train, y_train = make_blobs(n_samples=m, centers=centers, cluster_std=std,random_state=30)
X_train = np.array([[ 5.55075659,-1.04564419],[ 5.3231534, -2.99183439], [-3.10014381, -3.31564409], [-2.4230153, -0.93551791],
                    [-4.9033676, -0.38310963], [-0.57449011,-2.66475512], [ 0.29859758 ,2.84908785], [-6.72596243, 3.58509537],
                    [ 6.2418555, -0.29225312], [-1.01498214, -3.70046527], [-3.61615283, 2.76038508],[ 2.05069979,  1.46312554],
                    [ 3.83671763, -2.0213351 ],[ 5.7827765, -3.7932379 ],[ 3.41800471, -3.24311418],[-3.42177445, 2.10749794],
                    [-2.28564551, -1.46163252], [ 2.11347211, 0.07883028], [-2.005778,  -2.46747897],[ 0.31077063, 1.14724314],
                    [-5.10069672, 2.30379318], [-6.26405266, 3.52790535], [ 1.81050091, 0.95522163], [-5.76404783, 1.22481149],
                    [ 4.61298353, -3.92730539],[ 0.60213256, 3.01912738],[-4.86570341, 0.89314453],[-5.97071094, 2.47055962],
                    [ 4.58983098, -3.05507534], [-4.08389663, -1.06221829], [ 2.8988813, 1.64515036], [ 0.97080728,  3.39405598]])

y_train = np.array([3,3, 1, 1, 1, 1, 2, 0, 3, 1, 0, 2, 3, 3, 3, 0, 1, 2, 1, 2, 0, 0, 2, 0, 3, 2, 0, 0, 3, 1, 2, 2])
#plt_mc(X_train, y_train,4,centers,std=1.0)
model = Sequential(
    [
        tf.keras.Input(shape=(2,)),
        Dense(2, activation = 'relu',   name = "L1"),
        Dense(4, activation = 'linear', name = "L2")
    ],name = "my_model"
)
w1 = np.array([[0.41752017,-1.1821599], [0.06095338, 1.0677675]])
b1 = np.array([0,0])
w2 = np.array([[-0.16419482, 0.4498222, 0.28154945, -0.07789922], [-0.8060634, -0.7707381, 0.8299103, -0.30200005]])
b2 = np.array([0,0,0,0])
model.layers[0].set_weights([w1, b1])
model.layers[1].set_weights([w2, b2])
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
)

model.fit(
    X_train,y_train,
    epochs=200
)