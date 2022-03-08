from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt
import pandas as pd
from random import shuffle


F = 8
L = 8

# Exercise 1
model = CmdStanModel(stan_file='code_1.stan')
sample = model.sample({'M': F})
stan_data = pd.DataFrame()
stan_data['lambda'] = sample.stan_variable('lambda')
y_sim = sample.stan_variable('y_sim')
for i in range(F):
    stan_data[f'y_sim_{i}'] = y_sim[:, i]
for series in stan_data:
    plt.hist(stan_data[series])
    plt.title(series)
    plt.show()

# Exercise 2
models = {i: CmdStanModel(stan_file=f'code_{i}.stan') for i in (2, 3)}

# Use N=0 for both models
stan_y = [0]*8 + [1]*7 + [2]
shuffle(stan_y)
stan_data = {'N': 16,
             'y': stan_y}
try:
    samples = {i: models[i].sample(stan_data) for i in models}
except RuntimeError:
    print("Value out of bounds, runtime error is thrown")
    samples = {i: models[i].sample({'N': 16, 'y': [0]*8 + [1]*8}) for i in models}
for sample in samples.values():
    theta = sample.stan_variable('theta')
    summary = sample.summary()
    plt.hist(theta, bins=20)
    plt.axvline(summary['5%']['theta'], color='r')
    plt.axvline(summary['50%']['theta'], color='g')
    plt.axvline(summary['95%']['theta'], color='b')
    plt.axvline(theta.mean(), color='y')
    plt.show()

