from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt
import pandas as pd


normal = pd.read_csv('normal.csv', index_col=0, header=0)
bins_no = 50
model = CmdStanModel(stan_file='stan1.stan')
sample = model.sample(data={'N': 1, 'y': [normal['value'][0]]})
plt.hist(sample.stan_variable('mu'), bins=bins_no)
plt.title('mu 1 sample')
plt.show()
plt.hist(sample.stan_variable('sigma'), bins=bins_no)
plt.title('sigma 1 sample')
plt.show()
# For one sample the answer is basically always mu = sample value, sigma = 0

mu = 0.2
sigma = 0.4
stan_data = {'N': len(normal['value']), 'y': normal['value'].to_list()}
sample = model.sample(data=stan_data)
plt.hist(sample.stan_variable('mu'), bins=bins_no)
plt.title('mu all samples')
plt.show()
plt.hist(sample.stan_variable('sigma'), bins=bins_no)
plt.title('sigma all samples')
plt.show()
# The more samples we give, the better the fit

# Exercise 2
coin = pd.read_csv('coin.csv', index_col=0, header=0)
model = CmdStanModel(stan_file='stan2.stan')
data = {'N': len(coin['Toss_Result']), 'y': coin['Toss_Result'].to_list()}
sample = model.sample(data=data)
plt.hist(sample.stan_variable('theta'), bins=bins_no)
plt.title('theta')
plt.show()


# Exercise 3
model3 = CmdStanModel(stan_file='stan3.stan')
sample3 = model3.sample({'N': len(normal['value']), 'y': normal['value'].to_list()})

