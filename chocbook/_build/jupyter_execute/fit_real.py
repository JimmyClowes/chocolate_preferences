#!/usr/bin/env python
# coding: utf-8

# # Fitting a model to real data
# 
# With a model that has been found to be able to reliably recover known parameters from simulated data, we're now in a position to apply that model to real data in order to learn about the parameters we're interested in. As a reminder, the aims of this project were to determine whether it is possible to learn anything about the appeal ratings and the rankings at the population level from the rank data collected at the individual level (the data below, as seen earlier in the exploratory data analysis).
# 
# So, here I fit the model to the real data, and then visualise the model fit to explore:
# - What we can infer from the model about the population mean appeal for each chocolate
# - What we can infer from the model about the variability in appeal for each chocolate across people
# - What the implications are from the model fitting about the probability each chocolate really is at each rank at the population level

# In[1]:


from src.data.make_dataset import read_processed_data

ranking_df = read_processed_data()
ranking_df


# In[2]:


import src.models.stan_models as sm

model = sm.StanModel(filename='choc_model.stan')
model.fit(ranking_df,
          step_size=0.01)


# # Drawing inferences from the fitted model
# 
# With the model fitted to the real data, we are now in a position to see what it is possible to infer at the population level. At this point, it's probably worth considering what the population is.
# 
# ### What is the population here?
# 
# When drawing inferences about the population level ratings and rankings fitted by the model, the question naturally arises "what is the population?" That's partly a question for the reader to answer - what are you willing to believe the population is that the people who participated in this ranking exercise are representative of? Just the 10 people involved? All South London based thirtysomethings who like to spend their evenings carrying out pointless exercises with whatever props are at hand? All people who eat chocolate?
# 
# Clearly it's reasonable to draw inferences from this modelling about aggregate preferences among the population of the ten partipants, and it's probably reasonable to extend that to drawing inferences about people of the same age group and demographics that took part. I would suggest however that the group of participants in the exercise wouldn't be representative of chocolate eaters in wider society, and that a larger and tailored data collection exercise would be needed to draw inferences relating to a wider group.
# 
# ## Population mean appeal for chocolates
# 
# This first visualition shows the model sample values for the population mean appeal for each chocolate. Since we only have ranking data from 10 people, there is quite a wide range of values for each chocolate. With more data from more people doing the ranking, it may be possible for the model to fit a tighter range of values for the mean appeal for the chocolates, but with just the 10 people's ranking data in this exercise it is essentially leaving the possibility open that rankings for each chocolate could still be high or low.
# 
# Nevertheless, there are some things it does seem to be possible to infer here. The mean appeal for Maltesers appears to be systematically higher than that for all other chocolates, showing that the model has confirmed the expectation from the exploratory data analysis that Maltesers should be at the most appealing end of the scale.

# In[3]:


model.viz_samples_violin('choc_mus_fitted',
                         yaxis_title='underlying appeal',
                         xaxis_labels=True)


# ## Population variation in appeal for chocolates
# 
# Here, the standard deviation parameter of the distribition of appeal for each chocolate as fitted by the model is visualised. This parameter tells us how much appeal is rated differently for a chocolate across the people rating it.
# 
# There are a couple of things of interest here. First, the values sampled by the model are generally larger than 2, and almost all larger than 1. Given that the model specified the _mean_ of the standard deviation in the chocolate population level appeals to be 1, the fact that the chocolate _standard deviations_ in appeal across people are almost all greater than 2 suggests that the amount of variation in ratings at the individual level is greater than the amount of variation in the average ratings per chocolate. What that means is that there is a lot of variation in the preferences of individual people - there is generally not a lot of accordance between people that some chocolates are better than others.
# 
# 
# Second, comparing the spread of values for the standard deviations sampled by the model, it is apparent that some chocolates have lower amounts of variation in appeal than most, and some have more than most. In particular, Wispa appears to have the least amount of variation in its appeal rating across the people ranking the chocolates, suggesting most people generally rate Wispa at about the same place on their appeal scale as each other. The standard deviations for Bounty and Eclair are larger than for other chocolates, meaning ratings vary more for those chocolates. This makes sense to me - I really dislike coconut flavour, but other people like it so I would expect it to have a relatively large amount of variation in appeal.

# In[4]:


model.viz_samples_violin('choc_sigmas_fitted',
                         yaxis_title='variation in underlying appeal',
                         xaxis_labels=True)


# ## Implications for probabilities of where the chocolates rank for the population
# 
# Finally, with the model having sampled values of the population level mean appeal for each chocolate, we can analyse the order of those values in each of the MCMC samples the model carried out to get samples of the population level ranking of the chocolates.
# 
# Based on this, there are runaway winners and losers for what could be considered the best and worst of the chocolates among this population. There's more than a 50% probability that Malteser is the top ranked of the chocolates for the population, and it's more than 80% likely it is at least in the top 3. Eclair has a 40% probability of being the viewed as the worst of the chocolates among this population.
# 
# The limited amount of variation in appeal for Wispa seen above, plus its mean appeal being in the middle of the scale, leads to it being extremely unlikley to be the best ranked or the worst ranked - it is firmly somewhere in the middle of the rankings. No one in this population is going to thank you for throwing them a Wispa, but they aren't going to throw it back at you either.
# 
# Twix and Dairy milk both put in a strong showing, with a lot of probability density at the top end of the rankings, and their probability of being the top ranked chocolate among the population is only limited by Maltesers having dominated the probability for top rank.

# In[5]:


model.viz_pop_ranking_samples()

