data {
	int N;
	real price[N];
}

generated quantities {
	real alpha = normal_rng(9500, 500);
	real beta = normal_rng(6.5, 0.5);
	real sigma = normal_rng(12000, 500);
	real score[N];
	for (i in 1:N){
		score[i] = normal_rng(price[i]*beta+alpha, sigma);
	}
}