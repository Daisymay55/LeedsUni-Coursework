#importing the required components to run the app
from scipy import interpolate
import dash_table
import dash_core_components as mydcc
import dash_bootstrap_components as dbc
from datetime import datetime as dt
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'
from scipy.integrate import odeint
import random

### Code has been adapted from work by Henri Froese [https://towardsdatascience.com/building-an-interactive-dashboard-to-simulate-coronavirus-scenarios-in-python-ed23100e0046]

############################################ The Model ################################################
 
def deriv(y, t, r0_y_interpolated, N, gamma, sigma, beta, alpha, rho):
    S, E, I, R, D = y
    
# the differential equations that make the data run, used for the main graph
# (S=susceptible, E=Exposed, I=Infected, R=Recovered, D=Dead)       
    dSdt = -beta(t) * I * S / N
    dEdt = beta(t) * I * S / N - sigma * E
    dIdt = sigma * E - (1 - alpha) * gamma * I - alpha * rho * I
    dRdt = (1-alpha) * gamma * I
    dDdt = alpha * rho * I
    return dSdt, dEdt, dIdt, dRdt, dDdt


#the variables for the equations above (alpha,beta,gamma,sigma & rho)
beta = 4.0  
rho = 1.0/10 
alpha = round(random.uniform(0.3, 0.7),1)

#gamma
n= [i for i in range(4,10)]
for _ in range(1):
    gamma = float(random.choice(n))
    gamma = round((1/gamma),1)

#sigma    
n= [i for i in range(1,7)]
for _ in range(1):
    sigma = float(random.choice(n))
    sigma = round((1/sigma),1)
        

# Defining the R0 value - used for the R0 Graph
def logistic_R_0(t, R_0_start, k, x0, R_0_end):
    return (R_0_start-R_0_end) / (1 + np.exp(-k*(-t+x0))) + R_0_end

#defining how the model runs and how long for (Jan-Sept)
def Model(initial_cases, initial_date, N, R_0_start, k, x0, R_0_end, r0_y_interpolated=None):
    days = 274
    def beta(t):
        return logistic_R_0(t, R_0_start, k, x0, R_0_end) * gamma
        
    diff = int((np.datetime64("2020-01-01") - np.datetime64(initial_date)) / np.timedelta64(1, "D"))
    if diff > 0:
        r0_y_interpolated = [r0_y_interpolated[0] for _ in range(diff-1)] + r0_y_interpolated
    elif diff < 0:
        r0_y_interpolated = r0_y_interpolated[(-diff):]

    last_date = np.datetime64(initial_date) + np.timedelta64(days-1, "D")
    missing_days_r0 = int((last_date - np.datetime64("2020-09-30")) / np.timedelta64(1, "D"))
    r0_y_interpolated += [r0_y_interpolated[-1] for _ in range(missing_days_r0+1)]

    y0 = N-initial_cases, initial_cases, 0.0, 0.0, 0.0
    t = np.linspace(0, days, days)
    print(t)
    ret = odeint(deriv, y0, t, args=(r0_y_interpolated, N,
                                        gamma, sigma, beta, alpha, rho))
    S, E, I, R, D = ret.T
    R_0_over_time = r0_y_interpolated
    total_CFR = [0] + [100 * D[i] / sum(sigma*E[:i]) if sum(
        sigma*E[:i]) > 0 else 0 for i in range(1, len(t))]
    



    dates = pd.date_range(start=np.datetime64(initial_date), periods=days, freq="D")

    return dates, S, E, I, R, D, R_0_over_time, total_CFR


############################################ the dash app layout ################################################
external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Plague Simulation"


# these are the controls where the parameters can be tuned.
# They are not placed on the screen here, we just define them.
# Each separate input is placed in its own "dbc.FormGroup" and gets a "dbc.Label" for the name.
# The controls are wrappen in a "dbc.Card" for presentation.
controls = dbc.Card(
    [
     #Date Outbreak Begins
        dbc.FormGroup(
            [                
                dbc.Label('Date of first infection'),
                html.Br(),
                dcc.DatePickerSingle(
                    day_size=39,  # how big the date picker appears
                    display_format="DD.MM.YYYY",
                    date='2020-01-01',
                    id='initial_date',
                    min_date_allowed=dt(2020, 1, 1),
                    max_date_allowed=dt(2020, 9, 30),
                    initial_visible_month=dt(2020, 1, 15),
                    placeholder="test"
                ),
            ]
        ),
     #How many cases started the infection e.g if part of bioweapon intial cases will be higher  
        dbc.FormGroup(
            [
                dbc.Label("Initial Cases"),
                dbc.Input(
                    id="initial_cases", type="number", placeholder="initial_cases",
                    min=1, max=1_000_000, step=1, value=10,
                )
            ]
        ),
    #Total population - default is UK population in 2019
        dbc.FormGroup(
            [
                dbc.Label("Population"),
                dbc.Input(
                    id="population", type="number", placeholder="population",
                    min=10_000, max=1_000_000_000, step=10_000, value=66_650_000, #Default is the population of the UK as per 2019
                )
            ]
        ),
        
        # this is the input where the R value can be changed over time.
        # It is implemented as a table where the date is in the first column,
        # and users can change the R value on that date in the second column.
        dbc.FormGroup(
            [
                dbc.Label('Reproduction rate R over time'),
                dash_table.DataTable(
                    id='r0_table',
                    columns=[
                        {"name": "Date", "id": "Date"},
                        {"name": "R value", "id": "R value",
                         "editable": True, "type": "numeric"},
                    ],
                    data=[
                        {
                            "Date": i[0],
                            "R value": i[1],
                        }
                        for i in [("2020-01-01", 3.2), ("2020-02-01", 2.9), ("2020-03-01", 3.0), ("2020-04-01", 3.1), ("2020-05-01", 3.0), ("2020-06-01", 3.4), ("2020-07-01", 3.1), ("2020-08-01", 2.9), ("2020-09-01", 3.2)]
                    ],
                    style_cell_conditional=[
                        {'if': {'column_id': 'Date'},
                         'width': '5px'},
                        {'if': {'column_id': 'R value'},
                         'width': '10px'},
                    ],
                    style_cell={'textAlign': 'left',
                                'fontSize': 16, 'font-family': 'Helvetica'},
                    style_header={
                        'backgroundColor': 'white',
                        'fontWeight': 'bold'
                    },

                ),
            ]
        ),
        dbc.Button("Apply", id="submit-button-state",
                   color="primary", block=True)
    ],
    body=True,
)

# layout for the whole page
app.layout = dbc.Container(
    [
        # first, a jumbotron for the description and title
        dbc.Jumbotron(
            [
                dbc.Container(
                    [
                        html.H1("Plague Simulation", className="display-3"),
                        html.P(
                            "Interactively simulate different Plague outbreaks. ",
                            className="lead",
                        ),
                        html.Hr(className="my-2"),
                        dcc.Markdown('''

                            You can freely tune the date the first infection occured, how many are initially infected, the total population, and the reproduction rate R over time.
                            As default it is set to 10 initial cases, this can be made higher if required, e.g. use of a bioweapon, or lower if desired. 
                            The total population is set to that of the UK, as of 2019, but this can be adjusted depending on user preference e.g. global population or a particular town/city. 
                            R rate is currently set to between 2.9-3.4 which is the range of historic bubonic plague outbreaks; although this can be made higher to simulate a second wave, or made lower to mimick lockdowns.
                            
                            '''
                                     )
                    ],
                    fluid=True,
                )
            ],
            fluid=True,
            className="jumbotron bg-white text-dark"
        ),
        # now onto the main page, i.e. the controls on the left
        # and the graphs on the right.
        dbc.Row(
            [
                # here we place the controls we just defined,
                # and tell them to use up the left 3/12ths of the page.
                dbc.Col(controls, md=3),
                # now we place the graphs on the page, taking up
                # the right 9/12ths.
                dbc.Col(
                    [
                        # the main graph that displays plague over time.
                        dcc.Graph(id='main_graph'),
                        # the graph displaying the R values the user inputs over time.
                        dcc.Graph(id='r0_graph'),
                        # the next two graphs don't need as much space, so we
                        # put them next to each other in one row.
                        dbc.Row(
                            [
                                # the graph for the fatality rate over time.
                                dbc.Col(dcc.Graph(id='cfr_graph'), md=6),
                                # the graph for the daily deaths over time.
                                dbc.Col(dcc.Graph(id="deaths_graph"), md=6)

                            ]
                        ),
                    ],
                    md=9
                ),
            ],
            align="top",
        ),
    ],
    # fluid is set to true so that the page reacts nicely to different sizes etc.
    fluid=True,
)



############################################ the dash app callbacks ################################################


@app.callback(
    [dash.dependencies.Output('main_graph', 'figure'),
     dash.dependencies.Output('cfr_graph', 'figure'),
     dash.dependencies.Output('r0_graph', 'figure'),
     dash.dependencies.Output('deaths_graph', 'figure'),
     ],
     
    [dash.dependencies.Input('submit-button-state', 'n_clicks')],

    [dash.dependencies.State('initial_cases', 'value'),
     dash.dependencies.State('initial_date', 'date'),
     dash.dependencies.State('population', 'value'),
     dash.dependencies.State('r0_table', 'data'),
     dash.dependencies.State('r0_table', 'columns')
     ]
)

# this is where the model and layout are combined to produce the app visuals.

def update_graph(_, initial_cases, initial_date, population, r0_data, r0_columns):
    
    last_initial_date, last_population  = "2020-01-15", 1_000_000 
    if not (initial_date and population):
        initial_date, population = last_initial_date, last_population


    r0_data_x = [datapoint["Date"] for datapoint in r0_data]
    r0_data_y = [datapoint["R value"] if ((not np.isnan(datapoint["R value"])) and (datapoint["R value"] >= 0))  else 0 for datapoint in r0_data]
    f = interpolate.interp1d([0, 1, 2, 3, 4, 5, 6, 7, 8], r0_data_y, kind='linear')
    r0_x_dates = pd.date_range(start=np.datetime64("2020-01-01"), end=np.datetime64("2020-09-30"), freq="D")
    r0_y_interpolated = f(np.linspace(0, 8, num=len(r0_x_dates))).tolist()

    dates, S, E, I, R, D, R_0_over_time, total_CFR = Model(initial_cases, initial_date, population, 3.0, 0.01, 50, 2.3, r0_y_interpolated)

    return {  # return graph for compartments, graph for fatality rates, graph for reproduction rate, and graph for deaths over time
        'data': [
            {'x': dates, 'y': S.astype(int), 'type': 'line', 'name': 'susceptible'},
            {'x': dates, 'y': E.astype(int), 'type': 'line', 'name': 'exposed'},
            {'x': dates, 'y': I.astype(int), 'type': 'line', 'name': 'infected'},
            {'x': dates, 'y': R.astype(int), 'type': 'line', 'name': 'recovered'},
            {'x': dates, 'y': D.astype(int), 'type': 'line', 'name': 'dead'},
        ],
        'layout': {
            'title': 'Compartments over time'
        }
        }, {
        'data': [
            {'x': dates, 'y': total_CFR, 'type': 'line',
                'name': 'total'}
        ],
        'layout': {
            'title': 'Fatality rate over time (%)',
            }
        }, {
        'data': [
            {'x': dates, 'y': R_0_over_time, 'type': 'line', 'name': 'susceptible'}
        ],
        'layout': {
            'title': 'Reproduction Rate R over time',
            }
        }, {
        'data': [
            {'x': dates, 'y': [0] + [D[i]-D[i-1] for i in range(1, len(dates))], 'type': 'line', 'name': 'total'},
             ],
        'layout': {
            'title': 'Deaths per day'
        }
        }

   
if __name__ == '__main__':
    app.run_server(debug=True)
