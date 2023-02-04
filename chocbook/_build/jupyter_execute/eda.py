#!/usr/bin/env python
# coding: utf-8

# # Exploratory data analysis
# Prior to any modelling, some exploratory data analysis helps to identify whether there are any unexpected quirks in the data, and also to identify what the ball park expectations of the outcomes of the modelling should be.
# 
# ## Read in processed data
# The data from the rankings decided by each person is transformed into a DataFrame format with a record per person and chocolate. Each participant and each chocolate are given an index value, and the rank that participant allocated that chocolate is explicitly included numerically, from rank 0 being that person's most preferred chocolate, rank 1 the next and so on.

# In[1]:


from src.data.make_dataset import read_processed_data
ranking_df = read_processed_data()
ranking_df


# ## Mean rankings of chocolates across people
# As a starting point, the mean ranks of the 17 chocolates across the 10 people is visualised. This suggests that when we come to see the modelled population-level appeal of the chocolates, we should expect to see Maltesers and Twix toward the top end of appeal and Eclair toward the bottom end of appeal. It also suggests the generative model should be capable of producing ranking outcomes where a small number of chocolates are strongly preferreed or disliked by all the people, while many are quite closely bunched in the middle of the rankings.

# In[2]:


import src.visualization.viz_rankings as vizrank
vizrank.plot_rank_means(ranking_df)


# ## Frequency of occurrece of ranking in top 5 / bottom 5 per chocolate
# Next, the frequency of appearance among top 5 preferences and bottom 5 preferences for each chocolate is visualised. What we can learn from this is that the generative model should be capable of producing outcomes where appeal for some chocolates is bunched at one end of the scale for almost all participants. The generative model should also be capable of producing outcomes where some chocolates are ranked in the top 5 by some people but in the bottom 5 by other people. This suggests that for some chocolates the individual effect should be larger than the differences in population level effects, since the individual variaition in the appeal is enough to make chocolates higher or lower in the rankings in a way that is not due to a commonly held preference across participants.

# In[3]:


vizrank.plot_top_bottom_n(ranking_df, 5)

