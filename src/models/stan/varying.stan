data {
    int<lower=1> n_people;
    int<lower=1> n_chocs;
    array[n_people, n_chocs] int rankings;
}

transformed data {
    array[n_people, n_chocs] int rankings_argsort;
	for (i in 1:n_people){
 		rankings_argsort[i] = sort_indices_asc(rankings[i]);
	}
}

parameters {
    vector[n_chocs] choc_mus_fitted; // mean latent ratings for chocolates
    vector<lower=0>[n_chocs] choc_sigmas_fitted; // sd of latent ratings for chocolates
    array[n_people] ordered[n_chocs] ratings; // latent ratings for each person

    real choc_sigmas_alpha;
    real choc_sigmas_denom;
}

transformed parameters {
    real choc_mus_std = sd(choc_mus_fitted);
    vector[n_chocs] choc_mus_adj;
    choc_mus_adj = choc_mus_fitted ./ choc_mus_std;

    real choc_sigmas_beta;
    choc_sigmas_beta = choc_sigmas_alpha / choc_sigmas_denom;
}

model {
    choc_mus_fitted ~ normal(0, 1); // prior on mean chocolate latent ratings

    choc_sigmas_alpha ~ gamma(5, 1);
    choc_sigmas_denom ~ gamma(20, 10);

    choc_sigmas_fitted ~ gamma(5,choc_sigmas_beta); // prior on sd of chocolate latent ratings
    
    for (i in 1:n_people){
        ratings[i][rankings_argsort[i]] ~ normal(choc_mus_adj, choc_sigmas_fitted);
    }
}