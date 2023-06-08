import itertools

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.stats import linregress
from project.Data import Data, GeographicScale, FileType, DataMean
from project.Fitness import Algorithms


class FitnessGeoPlot:
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
        self.fig = make_subplots(rows=1, cols=2, subplot_titles=('Green Fitness', 'Patent Fitness'),
                                 specs=[[{'type': 'choropleth'}, {'type': 'choropleth'}]])

    def make_plot(self):
        self._draw_map(data=self.data1, row=1, col=1)
        self._draw_map(data=self.data2, row=1, col=2)
        return self.fig

    def _draw_map(self, data, row, col):
        fitness, complexity = Algorithms.compute_fitness_complexity(data=data)
        fitness = fitness[fitness['fitness'] > 0]
        self.fig.add_trace(self.get_choropleth_trace(fitness), row=row, col=col)
        self.fig.update_layout(title_text=f'Green fitness vs patent fitness in {self.data1.year[0]}',
                               coloraxis=dict(colorscale='Reds', colorbar=dict(title="Log(Fitness)", orientation='v')))

    def get_choropleth_trace(self, fitness):
        trace = go.Choropleth(
            locationmode='ISO-3',
            locations=fitness.index,
            z=np.log(fitness.values.flatten() + 0.1),
            text=fitness.index,
            colorscale='Reds',
            autocolorscale=False,
            coloraxis="coloraxis")
        return trace


class ComplexityNetworkPlot:
    def __init__(self, data):
        self.data = data

    def make_plot(self):
        pass


class MultiYearCorrelationPlot:
    def __init__(self, data_green1, data_green2, data_patents1, data_patents2):
        self.data_green1 = data_green1
        self.data_green2 = data_green2
        self.data_patents1 = data_patents1
        self.data_patents2 = data_patents2
        self.fig = go.Figure()

    def plot_correlation_fitness_ranking_green_patents(self):
        CorrelationPlotPatents(data_green=self.data_green1, data_patents=self.data_patents1, fig=self.fig,
                               year=self.data_green1.year).plot_correlation_fitness_ranking_green_patents()
        CorrelationPlotPatents(data_green=self.data_green2, data_patents=self.data_patents2, fig=self.fig,
                               year=self.data_green2.year).plot_correlation_fitness_ranking_green_patents()
        self._layout()
        return self.fig

    def _layout(self):
        self.fig.update_layout(
            title_text=f'Green fitness vs patent fitness', height=950)
        self.fig.update_xaxes(title_text='Green fitness')
        self.fig.update_yaxes(title_text='Patent fitness')


class CorrelationPlotPatents:
    def __init__(self, data_green, data_patents, fig, year):
        self.data_green = data_green
        self.data_patents = data_patents
        self.fig = fig
        self.year = year

    def plot_correlation_fitness_ranking_green_patents(self):
        fitness_green, _ = Algorithms.compute_fitness_complexity(data=self.data_green)
        fitness_patents, _ = Algorithms.compute_fitness_complexity(data=self.data_patents)
        fitness_green_ranking = fitness_green['fitness'].rank(ascending=True)
        fitness_patents_ranking = fitness_patents['fitness'].rank(ascending=True)
        number_of_patents = 10 * np.log(self.data_green.data.sum(axis=1) + 10)
        #diversification = 3 * np.sum(Algorithms._mcp(data=self.data_green), axis=1)
        self.plot_scatter(x=fitness_green_ranking, y=fitness_patents_ranking, labels=fitness_green.index,
                          marker_sizes=number_of_patents)

    def plot_scatter(self, x, y, labels, marker_sizes):
        trace_scatter = go.Scatter(x=x, y=y, mode='markers+text', name=f'Countries {self.year}',
                                   marker=dict(size=marker_sizes), hovertext=labels, text=labels)
        slope, intercept = linregress(x, y)[0:2]
        trace_regression = go.Scatter(x=x, y=slope * x + intercept, mode='lines',
                                      name=f'Regression {self.year}')
        self.fig.add_trace(trace_scatter)
        self.fig.add_trace(trace_regression)


class NestedPlot:
    def __init__(self, data_green1, data_green2, data_patents1, data_patents2):
        self.data_green1 = data_green1
        self.data_green2 = data_green2
        self.data_patents1 = data_patents1
        self.data_patents2 = data_patents2
        self.fig = make_subplots(rows=2, cols=2, subplot_titles=("Green fitness 1990-1995", "Patent fitness 1990-1995", "Green fitness 2014-2019",
                                                                  "Patent fitness 2014-2019"))

    def make_plot(self):
        self._plot_matrix(data=self.data_green1, row=1, col=1)
        self._plot_matrix(data=self.data_patents1, row=1, col=2)
        self._plot_matrix(data=self.data_green2, row=2, col=1)
        self._plot_matrix(data=self.data_patents2, row=2, col=2)
        self.fig.update_layout(title_text="Country-patent matrix", height=950)
        return self.fig

    def _plot_matrix(self, data, row, col):
        mcp = Algorithms._mcp(data=data)
        fitness, complexity = Algorithms.compute_fitness_complexity(data=data)
        mcp_df = pd.DataFrame(mcp, index=data.data.index, columns=data.data.columns)
        rank_sorted_fitness = np.argsort(fitness.values.flatten())[::-1]
        rank_sorted_complexity = np.argsort(complexity.values.flatten())
        sorted_mcp_df = mcp_df.iloc[rank_sorted_fitness, rank_sorted_complexity]
        heatmap_trace = go.Heatmap(z=sorted_mcp_df.values[::-1], x=sorted_mcp_df.columns, y=sorted_mcp_df.index, colorbar=None)
        self.fig.add_trace(heatmap_trace, row=row, col=col)


class PlotsPresentation:
    @staticmethod
    def get_data():
        data_green1 = DataMean(years=list(range(1990, 1995)), geo_scale=GeographicScale.COUNTRY, file_type=FileType.Y)
        data_green2 = DataMean(years=list(range(2014, 2019)), geo_scale=GeographicScale.COUNTRY, file_type=FileType.Y)
        data_patents1 = DataMean(years=list(range(1990, 1995)), geo_scale=GeographicScale.COUNTRY,
                                 file_type=FileType.AH)
        data_patents2 = DataMean(years=list(range(2014, 2019)), geo_scale=GeographicScale.COUNTRY,
                                 file_type=FileType.AH)
        return data_green1, data_green2, data_patents1, data_patents2
    @staticmethod
    def generate_figure_multiyear_correlation_plot():
        data_green1, data_green2, data_patents1, data_patents2 = PlotsPresentation.get_data()
        multiyear_correlation_plot = MultiYearCorrelationPlot(data_green1=data_green1, data_green2=data_green2,
                                                              data_patents1=data_patents1, data_patents2=data_patents2)
        fig = multiyear_correlation_plot.plot_correlation_fitness_ranking_green_patents()
        return fig

    @staticmethod
    def generate_fitness_geo_plots():
        data_green1, data_green2, data_patents1, data_patents2 = PlotsPresentation.get_data()
        fig_1990 = FitnessGeoPlot(data1=data_green1, data2=data_patents1).make_plot()
        fig_2015 = FitnessGeoPlot(data1=data_green2, data2=data_patents2).make_plot()
        return fig_1990, fig_2015

    @staticmethod
    def generate_nestedness_plots():
        data_green1, data_green2, data_patents1, data_patents2 = PlotsPresentation.get_data()
        nested_plot = NestedPlot(data_green1=data_green1, data_green2=data_green2,
                                 data_patents1=data_patents1, data_patents2=data_patents2)
        fig = nested_plot.make_plot()
        return fig


if __name__ == '__main__':
    PlotsPresentation.generate_nestedness_plots().show()
