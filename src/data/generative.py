import numpy as np
import pandas as pd
import plotly.express as px

class SimGenerative():

    def __init__(self,
                 n_people=10,
                 n_chocs=17,
                 seed=123,
                 hyperparams={'choc_mus': {'loc': 0, 'scale': 1},
                              'choc_sigmas': {'shape': 5, 'scale': 0.5}}):

        self.n_people = n_people
        self.n_chocs = n_chocs
        self.seed = seed
        self.hyperparams = hyperparams

    def draw(self):

        np.random.seed(self.seed)

        self.choc_mus = np.random.normal(**self.hyperparams['choc_mus'],
                                         size=self.n_chocs)
        self.choc_sigmas = np.random.gamma(**self.hyperparams['choc_sigmas'],
                                           size=self.n_chocs)

        self.choc_ratings = np.random.normal(self.choc_mus[np.newaxis,:],
                                             self.choc_sigmas[np.newaxis,:],
                                             size=(self.n_people,  self.n_chocs))

        # negative of ratings is taken so that lower values are given higher rank indices
        self.choc_rankings = np.argsort(-self.choc_ratings, axis=1)

        self._make_ratings_rankings_df()

    def _ratings_to_df(self):

        return (pd.DataFrame(self.choc_ratings).
                            reset_index(names='person').
                            melt(id_vars='person',
                                 var_name='choc',
                                 value_name='rating')
                                 )

    def _rankings_to_df(self):

        return (pd.DataFrame(self.choc_rankings).
                            reset_index(names='person').
                            melt(id_vars='person',
                                 var_name='rank',
                                 value_name='choc')
                                 )

    def _make_ratings_rankings_df(self):

        self.ratings_rankings_df = pd.merge(self._ratings_to_df(),
                                            self._rankings_to_df(),
                                            on=["person", "choc"])


class SimViz():

    def __init__(self,
                 sim,
                 plot_config={'template': 'simple_white'}):
                      
        if isinstance(sim, SimGenerative):
            self.sim = sim
        else:
            raise Exception("sim must be an object of the SimGenerative class")

        self.plot_config = plot_config
        
    def plot_var(self,
                 var,
                 **kwargs):

        fig = px.histogram(getattr(self.sim, var),
                           **self.plot_config,
                           **kwargs)

        fig.update_layout(showlegend=False,
                            xaxis_title=var)
        
        return fig

    def plot_ratings(self,
                     facet_col=None,
                     facet_col_wrap=5,
                     facets_limit=None,
                     **kwargs):

        plot_data = self.sim.ratings_rankings_df

        if facets_limit is not None:
            plot_data = plot_data[plot_data[facets_limit['var']] < facets_limit['records']]

        fig = px.histogram(plot_data,
                            x='rating',
                            facet_col=facet_col,
                            facet_col_wrap=facet_col_wrap,
                            **self.plot_config,
                            **kwargs)

        return fig

    def plot_ratings_rankings(self,
                                facet_col_wrap=5,
                                facets_limit=None,
                                **kwargs):

        plot_data = self.sim.ratings_rankings_df

        if facets_limit is not None:
            plot_data = plot_data[plot_data[facets_limit['var']] < facets_limit['records']]

        fig = px.scatter(plot_data,
                            x='rank',
                            y='rating',
                            color='choc',
                            facet_col='person',
                            facet_col_wrap=facet_col_wrap,
                            **self.plot_config,
                            **kwargs)
        
        fig.update_layout(showlegend=False)

        fig.update_traces(marker={'size': 4})

        return fig