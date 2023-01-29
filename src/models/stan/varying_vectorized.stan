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
}

transformed parameters {
    real choc_mus_std = sd(choc_mus_fitted);
    vector[n_chocs] choc_mus_adj;
    choc_mus_adj = choc_mus_fitted ./ choc_mus_std;

    matrix[n_people, n_chocs] ratings_matrix;
    for (i in 1:n_people){
        ratings_matrix[i] = to_row_vector(ratings[i]);
    }

    vector[n_people*n_chocs] ratings_vector;
    ratings_vector = to_vector(ratings_matrix);

    matrix[n_people, n_chocs] choc_mus_matrix;
    for (i in 1:n_people){
        choc_mus_matrix[i] = to_row_vector(choc_mus_adj[rankings[i]]);
    }

    vector[n_people*n_chocs] choc_mus_vector;
    choc_mus_vector = to_vector(choc_mus_matrix);

    matrix[n_people, n_chocs] choc_sigmas_matrix;
    for (i in 1:n_people){
        choc_sigmas_matrix[i] = to_row_vector(choc_sigmas_fitted[rankings[i]]);
    }

    vector[n_people*n_chocs] choc_sigmas_vector;
    choc_sigmas_vector = to_vector(choc_sigmas_matrix);
}

model {
    choc_mus_fitted ~ normal(0, 1); // prior on mean chocolate latent ratings

    choc_sigmas_fitted ~ gamma(5,10); // prior on sd of chocolate latent ratings
    
    ratings_vector ~ normal(choc_mus_vector, choc_sigmas_vector);
}