#!/usr/bin/env python
# coding: utf-8

# # Building a generative model
# 
# Having visually inspected the ranking data collected, I now have an improved understanding of how rankings varied across the people taking part. I want to build a generative model that I believe is:<br>
# (a) theoretically plausible in terms of how people's rankings come about; and<br>
# (b) consistent with the patterns observed in the ranking data.
# 
# To build the generative model, I'm going to make some choices about what phenomena I think are involved in the process of generating the rankings, and I'm going to choose statistical distributions that I consider well suited to representing those phenomena.
# 
# The aspects of the data generating process I decided to include in the generative model were:
# - There are multiple people ranking chocolates (10 in the observed data, but it could be any number)
# 
# - There are multiple chocolates being ranked (17 in the observed data, but it could be any number)
# 
# - People's _rankings_ of chocolates is based on ordering of unobserved _ratings_ people give to each chocolate on a continuous latent scale (based on a [widely used approach to modelling rank data](https://en.wikipedia.org/wiki/Thurstonian_model))
# 
# - Each chocolate has some mean appeal and variance of appeal on the latent appeal scale
#   - Some chcocolates are generally agreed to be nicer than others, so they would have a higher mean appeal on the latent scale
#   - Some chocolates divide opinion more, so they would have a higher variance in appeal on the latent scale
# 
# - Each person has their own individual appeal rating for each chocolate drawn from the appeal distributions for each chocolate
# 
# - The _ranking_ each person produces is based on the ordering of their appeal _ratings_ for each chocolate from highest to lowest
# 
# ### Conception of the latent rating scale
# 
# My conception of the latent rating scale for the chocolates is that each individual has a conception of a one-dimensonsal rating space along which they place the chocolates being rated according to how much appeal each chocolate holds for that person. In the specification of the generative model, this scale is constrained by assumptions that ratings are normally distributed, so a strong preference or dislike for a given chocolate will be constrained in how far from the other ratings it can stray. Some constraint like this is necessary in order to model rankings from multiple people on a common scale.
# 
# With this in mind, the ratings given by individual people should be considered ratings across that person's own private appeal scale, i.e. the person rates the chocolate near the bottom of what appeal they think it is possible a chocolate could have, near the top of what appeal they think a chocolate could have, or somewhere in the middle. In this sense, ratings from different people can not be compared in absolute terms, i.e. person A rating chocolate X 2.3 while person B rates chocolate X 1.9 can not be understood to mean person A likes chocolate X more than person B does in absolute terms. What it does mean is that person A considered chocolate X to be higher up their private scale of how good a chocolate can be than person B does up their private scale.
# 
# It may or may not be the case that humans experience the consumption of chocolate in a common way and there there is some absolute rating scale that could be understood in common terms across people rating chocolates, but that philosophical question is outside the remit of this exercise.

# ## Defining the generative model

# To give myself the ability to readily generate data based on the generative model in a flexible way throughout the project, I set up a `SimGenerative` class to generate ranking data based on the generative model. Here a couple of function definitions show how I have specified the generative model. First, in the `__init__` function, the number of people and chocolates involved is defined, as well as a seed for random number generation to allow reproducibility, and some hyperparameter choices that will be used in the distributions within the generative model.

# In[1]:


import inspect
from src.data.generative import SimGenerative, SimViz

print(inspect.getsource(SimGenerative.__init__))


# Then, the `draw` method of the `SimGenerative` class defines the process for how rankings come about.
# 
# - `choc_mus` sets the mean appeal ratings for each chocolate at the population level, i.e. the appeal level of each chocolate averaged across all people, and is drawn from a normal distribution defined by the hyperparameters provided
# 
# - `choc_sigmas` sets the standard deviation in appeal ratings for each chocolate at the population level, i.e. how much variation there is in the appeal of each chocolate across all people, and is drawn from a gamma distribution defined by the hyperparameters provided
# 
# - `choc_ratings` sets the individual appeal ratings for each person relating to each chocolate, drawn from a normal distribution centred on the population mean appeal for each chocolate and with standard deviation of the standard deviation in appeal for that chocolate
# 
# - `choc_rankings` sets the rankings each person gives to each chocolate, determined by ordering the ratings each person gave to each chocolate from highest to lowest

# In[2]:


print(inspect.getsource(SimGenerative.draw))


# # Drawing samples from the generative model
# 
# Instantiating a `SimGenerative` object and drawing a simulation from it, we can see how the ratings look. Here a simulation is generated with the generative model using 100 people and 20 chocolates for demonstration purposes. In the data frame produced, we can see a rating is generated by each person for each chocolate. A person giving chocolate X a lower rating than chocolate Y will cause that person's ranking of chocolate X to be worse (i.e. a higher rank index value) than their ranking for chocolate Y. Rankings are assigned with the lowest rank index number being the highest rated chocolate and the highest rank index number being the lowest rated chocolate for each person, so rank 0 always means that person's favourite, i.e. highest rated, chocolate.

# In[3]:


sim = SimGenerative(n_people=100,
                    n_chocs=20,
                    seed=321)
sim.draw()
sim.ratings_rankings_df


# ## Visualising population mean and variation in appeal

# In[4]:


sim_viz = SimViz(sim)


# The mean appeal at the population level for chocolates is defined to be normally distributed, so most chocolates have an average appeal somewhere in the middle of the appeal scale, while a few chocolates lie at the extremeties with very low or very high average appeal. The mean and standard deviation assumed for this normal distribution is entirely arbitrary, and serves only to define the size of the appeal space across which people rate chocolates. Moving the centre of this distribution higher or lower, or making its spread wider or narrower, makes no difference to the interpretation of the ratings since the scale of the latent ratings is entirely arbitrary. For this reason, a normal distribution with mean 0 and standard deviation 1 serves well since it is then easy to interpret ratings relative to 0.

# In[5]:


sim_viz.plot_var('choc_mus',
                 marginal='rug',
                 title='Population mean preferences for chocolates')


# The other population level aspect of the generative model is the standard deviation in appeal for each chocolate. This describes how much variation there is in individual people's ratings of the appeal of a given chcolate. These standard deviations differ because some chocolates are specified to have similar ratings across people, whereas others are specified to be more divisive causing a wider spread in appeal ratings across people.
# 
# The population level standard deviation in ratings was defined to be gamma distributed, allowing for a range of possible values, but ensuring that all values are positive. The mean of the gamma distribution is α/β, where α and β are its shape and scale parameters respectively. Numpy defines the scale parameter as 1/β, so by choosing shape and scale 5 and 0.5, the mean of the population standard deviations in appeal is set to 2.5. More information and interactive visualiastions of the gamma distribution can be found at https://distribution-explorer.github.io/continuous/gamma.html
# 
# Unlike the situation above where the centre and spread of the population mean appeals per chocolate was explained as arbitrary, the scale of variation in appeal ratings across people per chocolate is now relative to the scale of variation in population level mean appeal across chocolates. So, since the population mean appeals were assumed to have standard devition 1, choosing a gamma distribution with mean 2.5 for the population level standard deviations in appeal means that the individual variation in ratings is on average assumed to be about 2.5 times the size of the population level variation in ratings. So, if a chocolate has a population mean rating of -1, people's ratings will average -1 and most people will give that chocolate a negative rating, but a reasonable number may still give it a strongly positive rating.
# 
# Here we can see most chocolates have a standard deviaition in appeal in the region of 2.5, but some do have standard deviations greater than 4 or less than 2.

# In[6]:


sim_viz.plot_var('choc_sigmas',
                 marginal='rug',
                 title='Population standard deviations in preferences for chocolates')


# ## Visualising individual ratings
# 
# Individuals' ratings of chocolates are defined to be normally distributed, with the means and standard deviations of those normal distributions being the results of the population effects described above. So, each chocolate has a population level mean and standard deviation in appeal, and then the individual people have appeal ratings for each chocolate drawn from the normal distributions defined by those parameters.
# 
# Here we can see the total spread of all rating values given by people for all chocolates, as well as colour coding to show which chocolates fell into which areas on the appeal scale.
# 
# Due to the compounding effect of the individual variation on top of the population level variation in appeal, the range of values on the appeal scale is wider at the individual level than it was at the population level, with some ratings below -5 or above 5. The ratings remain constrained however to a range of values in a region of the scale centred around 0, since the assumptions in each level of the generative model limit how wide the ratings can get.

# In[7]:


sim_viz.plot_ratings(color='choc',
                     marginal='rug')


# Plotting the ratings given by all people to each chocolate separately highlights the effects of the assumptions in the generative model. The location of ratings along the scale for chocolates differs, with some chocolates having high appeal and other chocolates low appeal. It is also apparent that for some chocolates the appeal ratings are tightly concentrated in one part of the appeal scale, while for others there is a wide spread of ratings. This is the results of the generative model assuming that population level standard deviaitions in appeal differ across chocolates, so that some chocolates are rated similarly by everyone, while other chocolates are more divisive and have a wide range of ratings given.

# In[8]:


sim_viz.plot_ratings(facet_col='choc',
                     title='Distribution of all ratings given per chocolate')


# Visualising ratings given by a selection of individual participants in a similar way, we can see the nature of the distributions of ratings does not behave in the same way as the distributions per chocolate. This is a result of the fact that no individual level variation is assumed in the generative model, i.e. that all people are rating chocolates on a scale that is invariant to the population level scale (implicitly assumed to be their own private appeal rating scale bounded by how bad or good they consider it possible for a chocolate to be). So the ratings given by individuals tend to be spread across the appeal scale largeley within the bounds of -5 to +5.

# In[9]:


sim_viz.plot_ratings(facet_col='person',
                     facets_limit={'var': 'person',
                                   'records': 20},
                     title='Distribution of ratings given by a selection of people')


# ## Visualising ratings and rankings
# 
# Again visualising just a selection of people, here we explore the relationship between ratings and rankings. The generative model assumes that each person's ranking of the chocolates is given by the ordering of the ratings that person implicitly gives to the chocolates on the unobserved latent rating scale. Consequently, The relationship between rankings and ratings is monotonically decreasing, i.e. a higher rating for A than B must lead to a better (i.e. lower index) ranking of A than B.
# 
# This visualisation is interesting because it also shows us something about how our observation of only the ranking - and not the rating - data limits our information about the relative appeal people find in the chocolates. The rank alone tells us only which chocolates have more appeal than others, but nothing about how much more appeal. There are cases in this visualisation where we can see reasonably large 'jumps' in rating from one chocolate in the ranking to the next, either where a person rates a specific chocolate as having much more or much less appeal than others, or where a person seems to consider a group of chocolates better than another group of chocolates, with a noticeable 'gap' in ratings between the two groups.

# In[10]:


sim_viz.plot_ratings_rankings(facets_limit={'var': 'person',
                                            'records': 20})


# ## Visualising the rankings only
# 
# The simulated data from the generative model gives richer information than the real data we have from the chocolate ranking process where we observe only the rank each person gives to each chocolate. While it is important that the generative model being assumed is plausible in terms of whether it seems like a reasonable theoretical conception of the data generating process and how people rate chocolates, it is also important that the data generating process is capable of producing ranking data that behaves like the real ranking data we have observed.
# 
# To give some assurance that the generative model can produce data like the real observed data, we can apply the same visualisations to the simulated data as we did to the real observed data in the exploratory data analysis step. In order to ensure the comparison with the real data is like for like, I first subset the simulated data to include only 10 people.

# In[11]:


import src.visualization.viz_rankings as vizrank
rank_prior_df = sim_viz.sim.ratings_rankings_df.query("person < 10").copy()


# Visualising the mean ranks given to the chocolates in the simulated process, it appears that a pattern similar to that in the real data is capable of being produced. Here, one chocolate has a much higher mean rank than the others, then a group of chocolate are closely following, before a large number of chocolates have very similar mean ranks, and finally a few chocolates perform substantially worse in turn. This is similar to what was observed in the real data, i.e. that many chocolates do not have much between them in terms of mean rank, while a few perform markedly better or worse.

# In[12]:


vizrank.plot_rank_means(rank_prior_df)


# Also replicating the visualisation from the exploratory data analysis of frequency of being ranked in the top or bottom 5, again it appears the generative model is capable of producing ranking data that performs similarly to the real observed ranking data.
# 
# Many chocolates appear in both the top 5 and the bottom 5 for different people, while some chocolates appear exclusively in people's top or bottom 5s, but not both. Based on this, there is no reason to reject the generative model as a representation of the data generating process that produced the real rankings.

# In[13]:


vizrank.plot_top_bottom_n(rank_prior_df, n=5)

