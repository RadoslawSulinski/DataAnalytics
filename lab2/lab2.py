from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt
import pandas as pd


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