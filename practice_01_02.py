import copy, math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
import matplotlib.pyplot as plt
plt.style.use("./deeplearning.mplstyle")
dlblue='#0096ff';dlorange='#FF9300';dldarked='#c0000';dlmagenta='#FF40FF';dlpurple='#7030A0';
np.set_printoptions(precision=2)
from lab_utils_multi import load_house_data, compute_cost, run_gradient_descent
from lab_utils_multi import norm_plot, plt_contour_multi, plt_equal_scale, plot_cost_i_w

x_train, y_train = load_house_data()
x_features = ['size(sqrft)', 'bedrooms', 'floors', 'age']
fix, ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(x_train[:,i], y_train)
    ax[i].set_xlabel(x_features[i])
def zscore_normalize_features(x):
    mu = np.mean(x,axis=0)
    sigma = np.std(x, axis=0)
    x_norm = (x-mu) / sigma
    return x_norm, mu,sigma
scaler = StandardScaler()
x_norm, a,b = zscore_normalize_features(x_train)
x_norm = scaler.fit_transform(x_train)
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(x_norm, y_train)
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")
print(f"model parameters from previous lab: w: [110.56 -21.27 -32.71 -37.97], b: 363.16")

