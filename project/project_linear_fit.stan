data {
	int N;
	vector[N] price;
	real scores[N];
}

parameters {
	real alpha;
	real beta;
	real<lower=0> sigma;
}

transformed parameters {
	vector[N] mu = price*beta+alpha;
}

model {
	alpha ~ normal(9500, 500);
	beta ~ normal(6.5, 0.5);
	sigma ~ normal(12000, 500);
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