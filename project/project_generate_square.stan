data {
	int N;
	real price[N];
}

generated quantities {
	real beta = normal_rng(10, 0.3);
	real sigma = normal_rng(12000, 500);
	real gamma = normal_rng(-0.0004, 0.00004);
	real score[N];
	for (i in 1:N){
		score[i] = normal_rng(price[i]*price[i]*gamma + price[i]*beta, sigma);
	}
}