data {
	int N;
	vector[N] price;
	real scores[N];
}

parameters {
	real beta;
	real<upper=0> gamma;
	real<lower=0> sigma;
}

transformed parameters {
	vector[N] mu = square(price)*gamma+price*beta;
}

model {
	beta ~ normal(10, 0.3);
	sigma ~ normal(12000, 500);
	gamma ~ normal(-0.0004, 0.00004);
	scores ~ normal(mu, sigma);
}

generated quantities {
	real score[N];
	vector[N] log_lik;
	array[N] real y_hat;
	for (i in 1:N){
		score[i] = normal_rng(mu[i], sigma);
		log_lik[i] = normal_lpdf(scores[i] | mu[i], sigma);
		y_hat[i] = normal_rng(mu[i], sigma);
	}
}