import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import os
from cmdstanpy import CmdStanModel

import git
git_root = git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel")

stan_model_dir = os.path.join(git_root, 'src', 'models', 'stan')

class StanModel(CmdStanModel):

    def __init__(self,
                 filename,
                 plot_config={},
                 **kwargs):

        self.filename = os.path.join(stan_model_dir,
                                     filename)

        self.plot_config = plot_config

        super().__init__(stan_file = self.filename)

    def fit(self,
            choc_rankings,
            **kwargs):

        if isinstance(choc_rankings, pd.DataFrame):
            self.data_df = choc_rankings

            self.choc_lookup = self.data_df[['choc_idx','choc']].drop_duplicates().sort_values('choc_idx')

            choc_rankings_array = np.array([choc_rankings[choc_rankings['person_idx']==i]['choc_idx'].
                                            to_numpy() for i in range(choc_rankings['person'].nunique())])
        else:
            choc_rankings_array = choc_rankings

        self.data =  {'n_people': choc_rankings_array.shape[0],
                    'n_chocs': choc_rankings_array.shape[1],
                    'rankings': np.flip(choc_rankings_array, axis=1).astype('int')}
        
        self.fit = self.sample(self.data,
                               **kwargs)

    def viz_samples_violin(self,
                           stan_var,
                           yaxis_title,
                           xaxis_labels=False,
                           actuals=None):

        fig = px.violin(self.fit.stan_variable(stan_var),
                        **self.plot_config)

        if xaxis_labels is True:

            fig.update_layout(xaxis = dict(tickmode = 'array',
                                           tickvals = self.choc_lookup['choc_idx'].tolist(),
                                           ticktext = self.choc_lookup['choc'].tolist()
                                           ))

        if actuals is not None:
            for i in range(0, len(actuals)):

                fig.add_shape(
                    type='line',
                    x0=(i-0.5),
                    y0=(actuals[i]),
                    x1=i+0.5,
                    y1=(actuals[i]),
                    line=dict(
                        color='Red',
                    )
                )

        fig.update_layout({
                           'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                           'paper_bgcolor': 'rgba(0, 0, 0, 0)'
                           },
                           showlegend=False)
                           
        fig.update_xaxes(showline=True,
                        linewidth=1,
                        linecolor='black',
                        title='chocolate',
                        tickangle=270)

        fig.update_yaxes(showline=True,
                        linewidth=1,
                        linecolor='black',
                        title=yaxis_title)

        return fig

    def viz_pop_ranking_samples(self,
                                n_rows=4,
                                n_cols=5):

        sample_argsort = np.argsort(-self.fit.stan_variable('choc_mus_fitted'), axis=1)

        fig = make_subplots(rows=n_rows,
                            cols=n_cols,
                            shared_xaxes='all',
                            shared_yaxes='all',
                            subplot_titles=self.choc_lookup['choc'].tolist())

        for i in range(self.data['rankings'].shape[1]):

            row_idx = (i//n_cols) + 1
            col_idx = (i % n_cols) + 1
            fig.append_trace(
                go.Histogram(x=np.where(sample_argsort==i)[1],
                                histnorm='probability',
                                name=str(i)),
                row=row_idx,
                col=col_idx
            )

        fig.update_layout(
        {
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)'
        },
        showlegend=False,
        font=dict(
                size=7
            ))

        fig.update_annotations(font_size=12)

        fig.update_xaxes(showline=True,
                            showticklabels=True,
                            linewidth=1,
                            linecolor='black',
                            tick0=0,
                            dtick=1,
                            tickangle=270
                            )
        fig.update_yaxes(showline=True,
                            showticklabels=True,
                            linewidth=1,
                            linecolor='black',
                            tick0=0,
                            dtick=0.25)

        fig.show()