#!/usr/bin/env python
# coding: utf-8

# # Fitting a model to simulated data
# 
# Having settled on a generative model I think represents the data generating process reasonably well, the next step is to simulate some data from the generative model, and then attempt to write an estimating model that can recover the known parameter values from the simulated data. If the estimating model can not recover the known parameters from simulated data, there is no reason to believe it would do a good job identifying true parameter values when fitting to real data. And, if you're willing to believe the generative model assumed is a reasonable representation of the true data generating process that produces the real data, seeing the estimating model recover known parameters from simulated data should give you confidence in the parameters the model estiamtes on the real data.
# 
# 
# ### Drawing simulated data from the generative model
# First, some data is simulated from the generative model, in the same way as it was in the previous step.

# In[1]:


from src.data.generative import SimGenerative
sim = SimGenerative(n_people=40,
                    n_chocs=15,
                    seed=321)
sim.draw()


# ### Fitting the estimating model to the simulated data
# 
# I used the Stan language to write a model to estimate the parameter values of interest. A Stan model must have at least `data`, `parameters` and `model` blocks, plus optional other code blocks.
# 
# The `data` block specifies what data needs to be passed into the model, which here is the number of people ranking chocolates, the number of chocolates, and the rank given by each person to each chocolate.
# 
# The `parameters` specifies the shape and type of the parameters being estimated in the model. The key type in this model that allows the model to be fit to rank data is the `ordered` type, specified as `array[n_people] ordered[n_chocs]`. This means that the model will estimate values for an array of `n_people` lots of ordered values of `n_chocs`, i.e. that each person's underlying ratings must decrease monotonically with their rankings. So the model requires that  the order of each person's ratings respects the order of their rankings.
# 
# The `model` block specfies how the parameters are assumed to behave in realtion to the data. The specification here should be exactly analagous to the generative model which was specified in numpy earlier in the project. So, as was the case there, each person's ratings for each chocolate are specified to be drawn from a normal distribution based on the population mean and standard deviation for that chocolate. By using the fact that each person's ratings are ordered in conjunction with the assumption each chocolate has such a distribution of appeal across people, the parameters of those distributions of appeal can be estiamted.

# In[2]:


import src.models.stan_models as sm
model = sm.StanModel(filename='choc_model.stan')


# In[3]:


print(model.code())


# With the model specified as above, it can be fitted to the data. Here I pass the ranking data to the model for fitting, as well as specifying an initial `step_size` for sampling, which I found made the model perform better than leaving the `step_size` at its default value.
# 
# Since I have created a `StanModel` class that inherits the underlying modelling capability from `cmdstanpy` (the interface for using Stan in python), I call the `fit` method, which behing the scenes uses [Markov Chain Monte Carlo (MCMC) sampling](https://mc-stan.org/docs/reference-manual/hmc.html) to sample estimates of the parameter values.

# In[4]:


model.fit(sim.choc_rankings,
          step_size=0.01)


# ## Visualising the model fit to the simulated data
# 
# With the model fitted to the data, we can now check whether the model has done a good job of recovering the known paramater values that were used to generate the simulated data.
# 
# This first visualisation shows the known values of the `mu` parameter in the standard distribution of appeal for each chocolate in the simulation (red lines) superimposed on the spread of paramter values sampled by the model during the model fitting process.
# 
# The model can be seen to have performed reasonably well, with the true values generally around the centre of the sampled values for each chocolate, and with the model samples focussing in higher regions for cases where the true value is high and lower regions where the true value is low. On this basis, we have good reason to trust the model's estimates of the mean value of appeal for the chocolates across people, i.e. the model is reasonably well able to glean information about which chocolates have systematically more and less appeal.

# In[5]:


model.viz_samples_violin(stan_var='choc_mus_fitted',
                         yaxis_title='underlying appeal',
                         actuals=sim.choc_mus)


# This second visualisation is similar to the first, but rather than showing the performance of the model in relation to the `mu` parameters of the normal distribution of appeal for each chocolate, it shows performance in relation to the `sigma` paramters, i.e. how much variation there is in appeal across people.
# 
# Again, the true values are generally around the centre of the sampled values, and the model focusses on higher regions where the values are truly higher and vice versa. So we also have good reason to trust the ability of the model to glean information about how much variation there is in the appeal of each chocolate.

# In[6]:


model.viz_samples_violin(stan_var='choc_sigmas_fitted',
                            yaxis_title='variation in underlying appeal',
                            actuals=sim.choc_sigmas)

