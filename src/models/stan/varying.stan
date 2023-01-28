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

    real<lower=0> choc_sigmas_alpha; // hyperparameter for sd of chocolate latent ratings
    real<lower=0> choc_sigmas_mean; // hyperparameter for sd of chocolate latent ratings
}

transformed parameters {

    // standardise the scale of choc_mus_fitted to ensure sd does not blow up
    real choc_mus_std = sd(choc_mus_fitted);
    vector[n_chocs] choc_mus_adj;
    choc_mus_adj = choc_mus_fitted ./ choc_mus_std;

    real choc_sigmas_beta; // hyperparameter for sd of chocolate latent ratings
    choc_sigmas_beta = choc_sigmas_alpha / choc_sigmas_mean;
}

model {
    choc_mus_fitted ~ normal(0, 1); // prior on mean chocolate latent ratings

    choc_sigmas_alpha ~ gamma(5, 1); // hyperprior on alpha of distribution of sd of chocolate latent ratings
    choc_sigmas_mean ~ gamma(10, 4); // hyperprior on mean of distribution of sd of chocolate latent ratings

    choc_sigmas_fitted ~ gamma(choc_sigmas_alpha,choc_sigmas_beta); // prior on sd of chocolate latent ratings
    
    for (i in 1:n_people){
        // model the ratings given by each person as normal based on mean and sd per chocolate

        // ratings parameters per person are re-ordered by argsort based on the ratings given by that person
        // so that the ratings for that chocolate are matched to the mean and sd for that chocolate

        ratings[i][rankings_argsort[i]] ~ normal(choc_mus_adj, choc_sigmas_fitted);
    }
}