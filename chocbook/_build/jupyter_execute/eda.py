#!/usr/bin/env python
# coding: utf-8

# # Exploratory data analysis
# Prior to any modelling, some exploratory data analysis helps to identify whether there are any unexpected quirks in the data, and also to identify what the ball park expectations of the outcomes of the modelling should be.

# In[1]:


from src.data.make_dataset import read_processed_data


# ## Read in processed data
# 
# The data from the rankings decided by each participant is transformed into a DataFrame format with a record per participant and chocolate. Each participant and each chocolate are given an index value, and the rank that participant allocated that chocolate is explicitly included numerically.

# In[2]:


ranking_df = read_processed_data()
ranking_df


# ## Visual 1: Mean rankings of chocolates across participants
# This visualises the mean rankings of the 17 chocolates across the 10 participants.
# 
# This suggests that when analysing the population-level preferences, we should expect to see Maltesers and Twix toward the top end of preferences and Eclair toward the bottom end of preferences. It also suggests the generative model should be capable of producing ranking outcomes where a small number of chocolates are strongly preferreed or disliked by all the participants, while many are quite closely bunched in the middle of the rankings.

# In[3]:


import src.visualization.viz_rankings as vizrank
vizrank.plot_rank_means(ranking_df)


# ## Visual 2: Frequency of occurrece of ranking in top 5 / bottom 5 per chocolate
# This visualises the number of participants who ranked each chocolate in their top 5 preferences and bottom 5 preferences.
# 
# From this, the generative model should be capable of producing outcomes where appeal for some chocolates is bunched at one end of the scale for almost all participants. The generative model should also be capable of producing outcomes where some chocolates are ranked in the top 5 by some people but in the bottom 5 by other people. This suggests that for some chocolates the individual effect should be larger than the differences in population level effects, since the individual variaition in the appeal is enough to make chocolates higher or lower in the rankings in a way that is not due to a commonly held preference across participants.

# In[4]:


vizrank.plot_top_bottom_n(ranking_df, 5)

