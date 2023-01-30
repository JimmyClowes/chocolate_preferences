import pandas as pd
import plotly.express as px

def plot_rank_means(ranking_df):

    ranking_df['choc'] = ranking_df['choc'].astype('string')

    fig = px.bar(ranking_df.groupby(['choc'])[['rank']].mean().sort_values('rank').reset_index(),
                 x='choc',
                 y='rank',
                 template='simple_white')

    fig.update_layout(showlegend=False,
                      yaxis_title='Mean rank')

    return fig

def plot_top_bottom_n(ranking_df,
                        n):

    top_n = 'top_{n}'.format(n=n)
    bottom_n = 'bottom_{n}'.format(n=n)

    ranking_df[top_n] = ranking_df['rank'] <= n-1
    ranking_df[bottom_n] = ranking_df['rank'] >= max(ranking_df['rank']) - n

    fig = px.bar(ranking_df[['choc', top_n, bottom_n]].melt(id_vars='choc').groupby(['choc','variable'])['value'].sum().reset_index(),
       y='choc',
       x='value',
       facet_col='variable',
       template='simple_white')

    fig.update_layout(showlegend=False,
                        xaxis_title='frequency',
                        yaxis_title='choc')

    return fig