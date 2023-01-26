with pm.Model() as model:

    choc_sigmas_fitted = pm.Gamma("choc_sigmas_fitted",
                                        alpha=5,
                                        beta=10,
                                        shape=(n_chocs))
    
    # overall distribution of mean preference values for chocolates
    choc_mus_fitted = pm.Normal("choc_mus_fitted",
                            0,
                            1,
                            shape=(n_chocs))

    # calculate standard deviation of fitted mu values to normalise scale of person ratings
    choc_mus_std = pm.Deterministic("choc_mus_std",
                                    pt.std(choc_mus_fitted))

    choc_mus_mean = pm.Deterministic("choc_mus_mean",
                                pt.mean(choc_mus_fitted))

    # map order of chocolate mus to individual ranks for use in ordered transform
    choc_mus_mapped = pm.Deterministic("choc_mus_mapped",
                                        (choc_mus_fitted[choc_rankings_true])/choc_mus_std)

    # map order of chocolate sigmas to individual ranks for use in ordered transform
    choc_sigmas_mapped = pm.Deterministic("choc_sigmas_mapped",
                                            choc_sigmas_fitted[choc_rankings_true])

    #individual distributions of preferences across chocolates, hierarchically related to overall means      
    person_ratings = pm.Normal("person_ratings",
                                choc_mus_mapped,
                                0.2,
                                transform=pm.distributions.transforms.univariate_ordered,
                                initval=np.repeat(np.linspace(-2,2,num=n_chocs)[None,:], n_people, axis=0),
                                shape=(n_people, n_chocs))
    
    idata = pm.sample(draws=200)