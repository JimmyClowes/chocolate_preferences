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
    real choc_mus_fitted[n_chocs]; // mean latent ratings for chocolates
    array[n_people] ordered[n_chocs] ratings; // latent ratings for each person
}

transformed parameters {
    real choc_mus_std = sd(choc_mus_fitted);
    vector[n_chocs] choc_mus_adj;
    choc_mus_adj = to_vector(choc_mus_fitted) ./ choc_mus_std;
}

model {
    choc_mus_fitted ~ normal(0, 1); // prior on mean chocolate latent ratings
    
    for (i in 1:n_people){
        ratings[i][rankings_argsort[i]] ~ normal(choc_mus_adj, 1);
    }
}