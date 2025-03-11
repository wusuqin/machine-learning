# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import ctypes
import math, copy
import matplotlib.pyplot as plt
from lab_utils_uni import plt_divergence, plt_house_x, plt_contour_wgrad, plt_gradients
plt.style.use("./deeplearning.mplstyle")

def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost = np.longdouble(0)
    for i in range(m):
        cost += (w*x[i]+ b - y[i])**2
    return np.longdouble(cost/(2*m))

def compute_gradient(x,y,w,b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        dj_dw += (w*x[i]+b-y[i])*x[i]
        dj_db += w*x[i] + b -y[i]
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return dj_dw, dj_db

def gradient_descent(x,y,w_in,b_in,alpha,num_iters,cost_function,gradient_function):
    w = copy.deepcopy(w_in)
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x,y,w,b)
        w= w - alpha*dj_dw
        b = b - alpha*dj_db
        if i < 100000:
            J_history.append(cost_function(x,y,w,b))
            p_history.append([w,b])
        if i%math.ceil(num_iters/10) == 0:
            print(f"Interation: {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw:{dj_dw:0.3e}, dj_db:{dj_db:0.3e}, ",
                  f"w: {w:0.3e}, b:{b: 0.5e}")
    return w,b,J_history,p_history
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
w_int = 0
b_int = 0
interations = 10
tmp_alpha = 8.0e-1
w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_int, b_int,tmp_alpha, interations, compute_cost,compute_gradient )
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist[:100])
ax2.plot(1000+np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs.iteration(start)")
ax2.set_title("Cost vs.iteration(end)")
ax1.set_ylabel("Cost")
ax2.set_ylabel("Cost")
ax1.set_xlabel("interation step")
ax2.set_xlabel("interation step")
print(f"1000 sqrft house  predication {w_final*1.0 + b_final:0.1f} thousand dollars")
print(f"1200 sqrft house  predication {w_final*1.2 + b_final:0.1f} thousand dollars")
print(f"2000 sqrft house  predication {w_final*2.0 + b_final:0.1f} thousand dollars")
fig, ax = plt.subplots(1,1,figsize=(12,6))
plt_contour_wgrad(x_train, y_train, p_hist, ax, w_range=[180, 220, 0.5], b_range=[80, 120, 0.5],
            contours=[1,5,10,20],resolution=0.5)
plt_divergence(p_hist,J_hist,x_train,y_train)
plt.show()