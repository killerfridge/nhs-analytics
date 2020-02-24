import json
import pandas as pd
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objs as go
import os
import flask
import textwrap
import datetime

server = flask.Flask(__name__)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server)

MAPBOX_TOKEN = 'pk.eyJ1Ijoia2lsbGVyZnJpZGdlIiwiYSI6ImNrNGUzNnBjdTA4c2czZWxoNjJ3MjY4ajYifQ.Nf_B_iK2zst6vGvgYT6YRQ'
px.set_mapbox_access_token(MAPBOX_TOKEN)

DATA_DIR = os.path.join('.', 'data')
MAP_DIR = os.path.join('.', 'maps')


def text_wrapper(text):
    list_text = textwrap.wrap(text, width=20)
    return '<br>'.join(list_text)


def df_from_map(path):
    """Takes a geojson file and pulls just the properties into a dataframe"""
    tmp_df = pd.read_json(path)
    tmp_list = []
    for i in range(len(tmp_df)):
        tmp_list.append(pd.DataFrame(tmp_df.iloc[i, 1]['properties'], index=[0]))
    return_df = pd.concat(tmp_list)
    return_df.set_index('objectid', inplace=True)
    return_df.columns = ['Code', 'Name', 'Easting', 'Northing', 'long', 'lat', 'area', 'length']

    return return_df


# df_cancer = df_from_map(os.path.join(MAP_DIR, 'cancer.json'))
df_stp = df_from_map(os.path.join(MAP_DIR, 'stp.json'))
df_ccg = df_from_map(os.path.join(MAP_DIR, 'ccg.json'))
df_local = df_from_map(os.path.join(MAP_DIR, 'nhs_local.json'))

df_universe = pd.read_csv(os.path.join(DATA_DIR, 'account universe.csv'))
df_universe['Name'] = df_universe['Name'].apply(text_wrapper)

# Load the cancer dataframe
df_cancer = pd.read_csv(os.path.join(DATA_DIR, 'cancer.csv'))

df_cancer.columns = ['ODS Code', 'Measure', 'Percentage', 'Value', 'Total', 'Year', 'Month']

df_cancer = df_cancer[df_cancer['Year'] >= 2018]

df_cancer = pd.merge(
    df_cancer, df_universe, left_on='ODS Code', right_on='Organisational Code', how='left'
)

# Load the waiting times
df_waiting = pd.read_csv(os.path.join(DATA_DIR, 'waiting.csv'),
                         dtype={
                             'Under 6': 'int16',
                             'Over 6': 'int16',
                             'Over 13': 'int16',
                         }, parse_dates=['Period'], dayfirst=True)

df_waiting.columns = [
    'Organisational Code',
    'Period',
    'Diagnostic Test',
    'Under 6 Weeks Waiting List',
    'Over 6 Weeks Waiting List',
    'Over 13 Weeks Waiting List',
    'Total on Waiting List',
    'Waiting List Activity',
    'Total Activity',
]

df_waiting = pd.merge(df_waiting, df_universe, on='Organisational Code', how='inner')


def waiting_times_map(measure, boundary, measure_type='volume', test=None, agg_func='sum'):
    if test:
        if isinstance(test, str):
            test = [test]
        df = df_waiting[df_waiting['Diagnostic Test'].isin(test)].copy()
    else:
        df = df_waiting.copy()

    df = df[[boundary_codes[boundary], boundary_names[boundary], measure, 'Total on Waiting List']]

    df.columns = ['Code', 'Name', 'Measure', 'Total']

    df_groupby = df.groupby(['Code', 'Name']) \
        .agg({'Measure': agg_func, 'Total': agg_func}).reset_index()

    df_groupby['Percentage'] = df_groupby['Measure'] / df_groupby['Total']

    if measure_type == 'volume':
        df_groupby.columns = ['Code', 'Name', 'Value', 'Total', 'Percentage']
        return df_groupby[['Code', 'Name', 'Value', 'Total']]

    if measure_type == 'percentage':
        df_groupby = df_groupby[['Code', 'Name', 'Percentage', 'Total']]
        df_groupby.columns = ['Code', 'Name', 'Value', 'Total']
        return df_groupby


def waiting_times_scatter(measure, test=None, ):
    df = df_waiting.copy()

    if test:
        if isinstance(test, str):
            test = [test]
        df = df[df['Diagnostic Test'].isin(test)]

    df = df[[
        'Organisational Code',
        'Name',
        "Period",
        'Diagnostic Test',
        'lat',
        'long',
        'CCG Code',
        'STP Code',
        'NHS Local Code',
        'Cancer Alliance Code',
        measure,
        'Total on Waiting List',
    ]]

    df.columns = [
        'Organisational Code',
        'Name',
        "Period",
        'Diagnostic Test',
        'lat',
        'long',
        'CCG Code',
        'STP Code',
        'NHS Local Code',
        'Cancer Alliance Code',
        'Value',
        'Total',
    ]

    return df


cancer_measures = [
    '31 Day Wait for first treatment By Cancer',
    '62 Day Wait for first treatment By Cancer'
]

waiting_measures = [
    'Under 6 Weeks Waiting List',
    'Over 6 Weeks Waiting List',
    'Over 13 Weeks Waiting List',
    'Total Activity',
]

measures = cancer_measures + waiting_measures


def clean_json(json_object):
    """Takes a json file, and promotes certain data to the first level of the object"""
    for feature in json_object['features']:
        code = [i for i in feature['properties'].keys() if i[-2:] == 'cd'][0]
        name = [i for i in feature['properties'].keys() if i[-2:] == 'nm'][0]
        feature['code'] = feature['properties'][code]
        feature['name'] = feature['properties'][name]
        feature['id'] = feature['properties'][code]

    return json_object


# Load the boundary data
with open(os.path.join(MAP_DIR, 'stp.json'), 'r') as f:
    stp_json = clean_json(json.load(f))

with open(os.path.join(MAP_DIR, 'ccg.json'), 'r') as f:
    ccg_json = clean_json(json.load(f))

with open(os.path.join(MAP_DIR, 'cancer.json'), 'r') as f:
    cnn_json = clean_json(json.load(f))

with open(os.path.join(MAP_DIR, 'nhs_local.json'), 'r') as f:
    nhs_local_json = clean_json(json.load(f))

views = {
    'ccg': ccg_json,
    'stp': stp_json,
    'cancer': cnn_json,
    'local': nhs_local_json
}

df_views = {
    'ccg': df_ccg,
    'stp': df_stp,
    'cancer': df_cancer,
    'local': df_local
}

boundary_codes = {
    'ccg': 'CCG Code',
    'stp': 'STP Code',
    'local': 'NHS Local Code',
    'cancer': 'Cancer Alliance Code'
}

boundary_names = {
    'ccg': 'CCG Name',
    'stp': 'STP Name',
    'local': 'NHS Local Name',
    'cancer': 'Cancer Alliance Name'
}


def filters():
    filters = html.Div(
        children=[
            html.P(
                'NHS England analytics covers January 2018 to October 2019 and will be updated as new data comes in'),
            html.H5('Map Grouping'),
            dcc.Dropdown(
                options=[
                    {'label': 'CCG', 'value': 'ccg'},
                    {'label': 'Cancer Alliances', 'value': 'cancer'},
                    {'label': 'STP', 'value': 'stp'},
                    {'label': 'NHS Local Office', 'value': 'local'}
                ],
                multi=False,
                clearable=False,
                value='stp',
                id='boundary'
            ),
            html.Hr(),
            html.H5('Measure'),
            dcc.Dropdown(
                options=[
                    {'label': 'Cancer', 'value': 'Cancer'},
                    {'label': 'Diagnostic Imaging', 'value': 'Diagnostic'},
                ],
                value='Cancer',
                multi=False,
                clearable=False,
                id='measure-type'
            ),
            dcc.Dropdown(
                options=[
                    {'label': x, 'value': x} for x in cancer_measures
                ],
                multi=False,
                clearable=False,
                value='31 Day Wait for first treatment By Cancer',
                id='measure',
                optionHeight=70
            ),
            html.Hr(),
            html.Div(
                [
                    dcc.Dropdown(
                        options=[
                            {'label': x, 'value': x} for x in df_waiting['Diagnostic Test'].unique()
                        ],
                        clearable=True,
                        multi=True,
                        value=[x for x in df_waiting['Diagnostic Test'].unique()],
                        id='diagnostic-filter'
                    ),
                    html.H5('Group by Diagnostic Test'),
                    dcc.Dropdown(
                        options=[
                            {'label': 'Grouped', 'value': 'Grouped'},
                            {'label': 'Ungrouped', 'value': 'Ungrouped'},
                        ],
                        clearable=False,
                        multi=False,
                        value='Grouped',
                        id='diagnostic-grouping'
                    )
                ],
                id='diagnostic-tests',
                style={'display': 'none'}
            ),
        ],
        className='filters'
    )

    return filters


app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(
                    'NHS England Analytics'
                )
            ],
            className='row',
        ),
        html.Div(
            [
                html.Div(
                    [filters()], className='two columns'
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [dcc.Graph(id='map-left')],
                                    className='five columns',
                                ),
                                html.Div(
                                    [dcc.Graph(id='map-right')],
                                    className='five columns'
                                )
                            ], className='row'
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [dcc.Graph(id='graph-bottom')],
                                    className='ten columns'
                                )
                            ],
                            className='row'
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [dcc.Graph(id='graph-bottom-ts')],
                                    className='ten columns'
                                )
                            ],
                            className='row'
                        )
                    ],
                    className='ten columns',
                ),
            ],
            className='row'
        )
    ]
)


@app.callback(
    [Output('measure', 'options'), Output('measure', 'value')],
    [Input('measure-type', 'value')]
)
def update_measures(measure_type):
    if measure_type == 'Cancer':
        return [{'label': x, 'value': x} for x in cancer_measures], cancer_measures[0]
    if measure_type == 'Diagnostic':
        return [{'label': x, 'value': x} for x in waiting_measures], waiting_measures[0]


@app.callback(
    Output('map-left', 'figure'),
    [Input('boundary', 'value'),
     Input('measure', 'value'),
     Input('diagnostic-filter', 'value')]
)
def boundary_update(boundary, measure, diag):
    if measure in cancer_measures:
        # TODO add a year filter, currently using all years for measure
        df = df_cancer[df_cancer['Measure'] == measure]
        df = df.groupby([
            boundary_codes[boundary],
            boundary_names[boundary],
            'Measure'
        ]).agg({'Value': 'sum'}).reset_index()
        df.columns = ['Code', 'Name', 'Measure', 'Value']
    elif measure in waiting_measures:
        df = waiting_times_map(measure, boundary, test=diag)
        df = df.groupby('Code').sum().reset_index()

    map_json = views[boundary]

    fig = go.Figure(
        go.Choroplethmapbox(
            geojson=map_json,
            locations=df.Code,
            z=df.Value,
            colorscale='Viridis',
            marker_line_width=0,
            marker_opacity=0.5,
        )
    )

    # figure = px.choropleth_mapbox(
    #     df,
    #     geojson=map_json,
    #     locations='Code',
    #     color='area',
    #     mapbox_style='light',
    #     center={'lat': 53, 'lon': -1.5},
    #     zoom=5,
    #     hover_name='Name',
    #     hover_data=['area'],
    # )
    #
    # figure.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    fig.update_layout(
        mapbox_style='light',
        mapbox_accesstoken=MAPBOX_TOKEN,
        mapbox_zoom=5,
        mapbox_center={'lat': 53, 'lon': -1.5},
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        clickmode='event+select'
    )

    return fig


@app.callback(
    Output('map-right', 'figure'),
    [Input('boundary', 'value'),
     Input('measure', 'value'),
     Input('map-left', 'selectedData'),
     Input('diagnostic-filter', 'value')
     ]
)
def sites_update(boundary, measure, area, diag):
    if measure in cancer_measures:
        # TODO Year filter
        df = df_cancer[df_cancer['Measure'] == measure]
    elif measure in waiting_measures:
        df = waiting_times_scatter(measure, test=diag)
    if not area:
        df = df
    else:
        filter_points = []
        for point in area['points']:
            filter_points.append(point['location'])
        df = df[df[boundary_codes[boundary]].isin(filter_points)]

    df = df.groupby(['Organisational Code', 'Name', 'lat', 'long']).agg({'Value': 'sum', 'Total': 'sum'}).reset_index()
    df['Percentage'] = (df['Value'] / df['Total']) * 100

    df['scale'] = df['Value'] / df['Value'].max()

    figure = go.Figure(
        go.Scattermapbox(
            lat=df['lat'],
            lon=df['long'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=df['scale'] * 30,
                opacity=.8,
                showscale=True,
                color=df['Percentage'],
                colorscale='Magma',
            ),
            text=df['Value'],
            customdata=df['Name'],
            name=measure,
            hovertemplate=
            'Name: %{customdata}<br>' +
            'Volume: %{text:,.0}<br>' +
            'Breaches: %{marker.color:.2f}%'
        )
    )

    # figure = px.scatter_mapbox(
    #     df,
    #     lat='lat',
    #     lon='long',
    #     color='Percentage',
    #     size='Value',
    #     center={'lat': 53, 'lon': -1.5},
    #     hover_name='Name',
    #     zoom=5,
    # )

    # figure.update_layout(
    # margin={"r": 0, "t": 0, "l": 0, "b": 0}, mapbox_accesstoken=MAPBOX_TOKEN, mapbox_style='light',
    # mapbox_zoom=5)

    figure.update_layout(
        mapbox_style='light',
        mapbox_accesstoken=MAPBOX_TOKEN,
        mapbox_zoom=5,
        mapbox_center={'lat': 53, 'lon': -1.5},
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        clickmode='event+select'
    )

    return figure


@app.callback(
    Output('graph-bottom', 'figure'),
    [Input('measure', 'value'), Input('map-right', 'selectedData'),
     Input('diagnostic-filter', 'value'), Input('diagnostic-grouping', 'value')]
)
def bottom_graph(measure, sites, diag, grouping):
    if measure in cancer_measures:
        df = df_cancer[df_cancer['Measure'] == measure]
    if measure in df_waiting:
        df = waiting_times_scatter(measure, test=diag)
        if grouping == 'Ungrouped':
            df = df.groupby([
                'Name',
                'Period',
            ]).sum().reset_index()

    site_list = []
    if sites:
        for point in sites['points']:
            site_list.append(point['customdata'])

    df = df[df['Name'].isin(site_list)]

    df['Percentage'] = df['Value'] / df['Total']

    df.sort_values("Name", inplace=True)

    if measure in waiting_measures:
        if grouping == 'Grouped':
            return px.box(df, x='Name', y='Percentage', points='all', color='Diagnostic Test')
        else:
            return px.box(df, x='Name', y='Percentage', points='all')
    else:
        return px.box(df, x='Name', y='Percentage', points='all')


@app.callback(
    Output('graph-bottom-ts', 'figure'),
    [Input('measure', 'value'), Input('map-right', 'selectedData'),
     Input('diagnostic-filter', 'value')]
)
def ts_graph(measure, sites, diag):
    if measure in cancer_measures:
        df = df_cancer[df_cancer['Measure'] == measure]
        df['Period'] = df.apply(lambda x: datetime.date(x['Year'], x['Month'], 1), axis=1)
    elif measure in df_waiting:
        df = waiting_times_scatter(measure, test=diag)
    else:
        raise ValueError('Value not in list')

    site_list = []

    if sites:
        for point in sites['points']:
            site_list.append(point['customdata'])

    df = df[df['Name'].isin(site_list)]

    df = df.groupby(['Name', 'Period']).agg({'Value': 'sum', 'Total': 'sum'}).reset_index()

    print(df.head())

    fig = go.Figure(
        data=[
            go.Bar(
                x=df.loc[df['Name'] == name, 'Period'],
                y=df.loc[df['Name'] == name, 'Value'],
                name=name
            ) for name in site_list
        ]
    )

    return fig


@app.callback(
    Output('diagnostic-tests', 'style'),
    [Input('measure-type', 'value')]
)
def diagnostic_filters(measure):
    if measure == 'Diagnostic':
        return None
    else:
        return {'display': 'none'}


if __name__ == "__main__":
    server.run(debug=True)
