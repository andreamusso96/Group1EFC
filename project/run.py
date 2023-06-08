from dash import Dash, dcc, html
import dash_mantine_components as dmc


external_stylesheets = [dmc.theme.DEFAULT_COLORS]
app = Dash(__name__, external_stylesheets=external_stylesheets)


if __name__ == '__main__':
    from Plots import PlotsPresentation
    fig_geo_1990, fig_geo_2015 = PlotsPresentation.generate_fitness_geo_plots()
    nestedness_fig = PlotsPresentation.generate_nestedness_plots()
    fig_correlation = PlotsPresentation.generate_figure_multiyear_correlation_plot()
    figures = [fig_geo_1990, fig_geo_2015, nestedness_fig, fig_correlation]

    app.layout = html.Div(
        style={'overflowY': 'scroll', 'height': '2500px'},  # This enables scrolling
        children=[dcc.Graph(figure=fig) for fig in figures]
    )

    app.run_server(debug=True)