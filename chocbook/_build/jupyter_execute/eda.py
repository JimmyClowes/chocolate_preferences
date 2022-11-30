#!/usr/bin/env python
# coding: utf-8

# # Exploratory data analysis
# Prior to any modelling, some exploratory data analysis helps to identify whether there are any unexpected quirks in the data, and also to identify what the ball park expectations of the outcomes of the modelling should be.

# In[1]:


from src.data.make_dataset import read_processed_data
import plotly.express as px


# ## Read in processed data
# 
# The data from the rankings decided by each participant is transformed into a DataFrame format with a record per participant and chocolate. Each participant and each chocolate are given an index value, and the rank that participant allocated that chocolate is explicitly included numerically.

# In[2]:


ranking_df = read_processed_data()
ranking_df


# ## Visual 1: Mean rankings of chocolates across participants
# This visualises the mean rankings of the 17 chocolates across the 10 participants.
# 
# This suggests that when analysing the population-level preferences, we should expect to see Maltesers and Twix toward the top end of preferences and Eclair toward the bottom end of preferences. It also suggests a small number of chocolates are strongly preferreed or disliked by all the participants, while many are quite closely bunched in the middle of the rankigns. This means that when generating priors for the model fitting process, a few outliers with most chocolates having mean attractivenss toward the middle of the distribution is a reasonable representation, whereas an extremely divisive spread of mean rankings would not be a good representation.

# In[3]:


fig = px.bar(ranking_df.groupby('choc')[['rank']].mean().sort_values('rank'))

fig.update_layout(
{
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)'
},
showlegend=False)

fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
fig.update_yaxes(showline=True, linewidth=1, linecolor='black',title='mean rank')

fig.show()


# ## Visual 2: Frequency of occurrece of ranking in top 5 / bottom 5 per chocolate
# This visualises the number of participants who ranked each chocolate in their top 5 preferences and bottom 5 preferences.
# 
# This suggests that when analysing the population-level preferences, we should expect Malteser and Eclair to have unimodal distributions, with most participants sharing the same view. In contrast, we should expect bimodal distributions for Snickers and Bounty, where participants had strongly opposing views.

# In[4]:


ranking_df['top_5'] = ranking_df['rank'] <= 4
ranking_df['bottom_5'] = ranking_df['rank'] >= 12


# In[5]:


fig = px.bar(ranking_df[['choc', 'top_5', 'bottom_5']].melt(id_vars='choc').groupby(['choc','variable'])['value'].sum().reset_index(),
       y='choc',
       x='value',
       facet_col='variable')

fig.update_layout(
{
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)'
},
showlegend=False)

fig.update_xaxes(showline=True, linewidth=1, linecolor='black', title='frequency')
fig.update_yaxes(showline=True, linewidth=1, linecolor='black',title='choc')

fig.show()

