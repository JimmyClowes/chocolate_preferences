import numpy as np
import plotly.express as px

import os
from cmdstanpy import CmdStanModel

import git
git_root = git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel")

stan_model_dir = os.path.join(git_root, 'src', 'models', 'stan')

class StanModel(CmdStanModel):

    def __init__(self,
                 filename,
                 **kwargs):

        self.filename = os.path.join(stan_model_dir,
                                     filename)

        super().__init__(stan_file = self.filename)

    def fit(self,
            choc_rankings,
            **kwargs):

        self.data = data={'n_people': choc_rankings.shape[0],
                          'n_chocs': choc_rankings.shape[1],
                          'rankings': np.flip(choc_rankings, axis=1).astype('int')}
        
        self.fit = self.sample(self.data,
                               **kwargs)

    def viz_samples_violin(self,
                           stan_var,
                           yaxis_title,
                           actuals=None):

        fig = px.violin(self.fit.stan_variable(stan_var))

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
                        title='chocolate')
        fig.update_yaxes(showline=True,
                        linewidth=1,
                        linecolor='black',
                        title=yaxis_title)

        return fig