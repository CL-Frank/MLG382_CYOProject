import os
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import joblib
import io
import base64
from dash import dash_table
import pickle

BASE_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(BASE_DIR, '..', 'artifacts')

# Helper function to build file paths
def get_artifact(filename):
    return os.path.join(ARTIFACTS_DIR, filename)

# # Load the saved model and preprocessors
model = joblib.load(get_artifact('best_housing_model.pkl'))
scaler_X = joblib.load(get_artifact('scaler_X.pkl'))
training_columns = joblib.load(get_artifact('training_columns.pkl'))

# Define unique cities and zipcodes (from notebook's EDA)
cities = ['Seattle', 'Renton', 'Bellevue', 'Redmond', 'Kirkland', 'Issaquah', 'Kent', 
          'Auburn', 'Sammamish', 'Federal Way', 'Shoreline', 'Woodinville', 'Maple Valley', 
          'Mercer Island', 'Burien', 'Snoqualmie', 'Kenmore', 'Des Moines', 'North Bend', 
          'Covington', 'Duvall', 'Lake Forest Park', 'Bothell', 'Newcastle', 'Tukwila', 
          'Vashon', 'Enumclaw', 'Carnation', 'Normandy Park', 'Fall City', 'Milton', 
          'Black Diamond', 'Ravensdale', 'Clyde Hill', 'Yarrow Point', 'Medina', 
          'Pacific', 'Algona', 'Skykomish', 'Preston', 'Beaux Arts Village', 'Hunts Point']
zipcodes = [98001, 98002, 98003, 98004, 98005, 98006, 98007, 98008, 98010, 98011, 
            98014, 98019, 98022, 98023, 98024, 98027, 98028, 98029, 98030, 98031, 
            98032, 98033, 98034, 98038, 98039, 98040, 98042, 98045, 98047, 98050, 
            98051, 98052, 98053, 98055, 98056, 98057, 98058, 98059, 98065, 98070, 
            98072, 98074, 98075, 98077, 98092, 98102, 98103, 98105, 98106, 98107, 
            98108, 98109, 98112, 98115, 98116, 98117, 98118, 98119, 98122, 98125, 
            98126, 98133, 98136, 98144, 98146, 98148, 98155, 98166, 98168, 98177, 
            98178, 98188, 98198, 98199, 98224, 98288]

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container(fluid=True, className="py-5 bg-light", children=[
    dbc.Row([
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    html.H1("House Price Prediction", className="text-center text-primary mb-4"),

                    dbc.Row([
                        dbc.Col([
                            dbc.Label("No. Bedrooms"),
                            dbc.Input(id='bedrooms', type='number', value=3, min=1),

                            dbc.Label("No. Bathrooms", className="mt-3"),
                            dbc.Input(id='bathrooms', type='number', value=2, min=0.5, step=0.25),

                            dbc.Label("Square Footage (Living)", className="mt-3"),
                            dbc.Input(id='sqft_living', type='number', value=2000, min=100),

                            dbc.Label("No. Floors", className="mt-3"),
                            dbc.Input(id='floors', type='number', value=1, min=1, step=0.5),

                            dbc.Label("Waterfront", className="mt-3"),
                            dcc.Dropdown(
                                id='waterfront',
                                options=[
                                    {'label': 'No', 'value': 0},
                                    {'label': 'Yes', 'value': 1}
                                ],
                                value=0
                            ),

                            dbc.Label("View 0-5", className="mt-3"),
                            dcc.Dropdown(
                                id='view',
                                options=[{'label': str(i), 'value': i} for i in range(5)],
                                value=0
                            )
                        ], md=6),
                        dbc.Col([
                            dbc.Label("Year Built"),
                            dbc.Input(id='year_built', type='number', value=2000, min=1900, max=2014),

                            dbc.Label("Has the House Been Renovated?", className="mt-3"),
                            dcc.Dropdown(
                                id='is_renovated',
                                options=[
                                    {'label': 'No', 'value': 0},
                                    {'label': 'Yes', 'value': 1}
                                ],
                                value=0
                            ),

                            dbc.Label("Condition", className="mt-3"),
                            dcc.Dropdown(
                                id='condition',
                                options=[{'label': str(i), 'value': i} for i in range(1, 6)],
                                value=3
                            ),

                            dbc.Label("Square Footage (Basement)", className="mt-3"),
                            dbc.Input(id='sqft_basement', type='number', value=0, min=0),

                            dbc.Label("City", className="mt-3"),
                            dcc.Dropdown(
                                id='city',
                                options=[{'label': city, 'value': city} for city in cities],
                                value='Seattle'
                            ),

                            dbc.Label("Zipcode", className="mt-3"),
                            dcc.Dropdown(
                                id='zipcode',
                                options=[{'label': str(z), 'value': z} for z in zipcodes],
                                value=98103
                            )
                        ], md=6)
                    ]),
                    dbc.Button("Predict", id="predict-button", color="primary", className="mt-4 w-100", n_clicks=0),
                    html.Div(id="prediction-output", className="mt-4 text-center text-success fs-5"),
                    html.Hr(),
                    html.H4("Batch Prediction (CSV Upload)", className="mt-4"),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select a CSV File')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin-bottom': '20px'
                        },
                        multiple=False
                    ),
                    html.Div(id='upload-output')
                                    ]),
                                    className="shadow-sm p-4"
                                )
            
        ], width=10, lg=8, xl=6, className="mx-auto")
        
        
    ])
    
])

def process_prediction(input_df, is_batch=False):
    # Validation (only for single prediction)
    if not is_batch:
        errors = []
        row = input_df.iloc[0]
        if row['year_built'] < 1900 or row['year_built'] > 2014:
            errors.append("Year built must be between 1900 and 2014.")
        if row['bedrooms'] < 1:
            errors.append("Bedrooms must be at least 1.")
        if row['bathrooms'] < 0.5:
            errors.append("Bathrooms must be at least 0.5.")
        if row['sqft_living'] < 100:
            errors.append("Living area must be at least 100 sqft.")
        if row['floors'] < 1:
            errors.append("Floors must be at least 1.")
        if row['sqft_basement'] < 0:
            errors.append("Invalid basement square footage.")
        if row['condition'] not in [1, 2, 3, 4, 5]:
            errors.append("Condition must be 1–5.")
        if row['view'] not in [0, 1, 2, 3, 4]:
            errors.append("View must be 0–4.")
        if row['city'] not in cities:
            errors.append("Invalid city.")
        if row['zipcode'] not in zipcodes:
            errors.append("Invalid zipcode.")
        if errors:
            return {"error": errors}

    # Derived features
    input_df['house_age'] = 2014 - input_df['year_built']
    input_df['sqft_living_bathrooms'] = input_df['sqft_living'] * input_df['bathrooms']

    # One-hot encode
    input_df = pd.get_dummies(input_df, columns=['city', 'zipcode'], prefix=['city', 'zipcode'])
    for col in training_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[training_columns]

    # Scale
    numerical_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'house_age',
                      'condition', 'sqft_basement', 'sqft_living_bathrooms']
    input_df[numerical_cols] = scaler_X.transform(input_df[numerical_cols])

    # Predict
    predictions = model.predict(input_df)
    return predictions

# Define the callback to predict price
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('bedrooms', 'value'),
    State('bathrooms', 'value'),
    State('sqft_living', 'value'),
    State('floors', 'value'),
    State('waterfront', 'value'),
    State('view', 'value'),
    State('year_built', 'value'),
    State('is_renovated', 'value'),
    State('condition', 'value'),
    State('sqft_basement', 'value'),
    State('city', 'value'),
    State('zipcode', 'value')],
)

def single_predict(n_clicks, bedrooms, bathrooms, sqft_living, floors, waterfront, view,
                   year_built, is_renovated, condition, sqft_basement, city, zipcode):
    if n_clicks:
        df = pd.DataFrame([{
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'sqft_living': sqft_living,
            'floors': floors,
            'waterfront': waterfront,
            'view': view,
            'year_built': year_built,
            'is_renovated': is_renovated,
            'condition': condition,
            'sqft_basement': sqft_basement,
            'city': city,
            'zipcode': zipcode
        }])
        result = process_prediction(df, is_batch=False)
        if isinstance(result, dict) and 'error' in result:
            return dbc.Alert("\n".join(result['error']), color="danger", dismissable=True)
        return dbc.Alert(f"Predicted House Price: ${result[0]:,.2f}", color="success", dismissable=True)
    return ""

    
@app.callback(
    Output('upload-output', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def batch_predict(contents, filename):
    if contents is None:
        return ""

    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        # Validate required columns
        required_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront', 'view',
                            'year_built', 'is_renovated', 'condition', 'sqft_basement',
                            'city', 'zipcode']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            return dbc.Alert(f"Missing columns: {', '.join(missing)}", color="danger")

        # Feature engineering
        df['house_age'] = 2014 - df['year_built']
        df['sqft_living_bathrooms'] = df['sqft_living'] * df['bathrooms']
        df = pd.get_dummies(df, columns=['city', 'zipcode'], prefix=['city', 'zipcode'])

        # Ensure same column order
        for col in training_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[training_columns]

        # Scale numerical features
        numerical_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'house_age',
                          'condition', 'sqft_basement', 'sqft_living_bathrooms']
        df[numerical_cols] = scaler_X.transform(df[numerical_cols])

        # Predict prices
        predictions = model.predict(df)

        # Format output
        results_df = pd.DataFrame({
            "Row": range(1, len(predictions) + 1),
            "Predicted Price": [f"${format(pred, ',.2f')}" for pred in predictions]
        })

        return dash_table.DataTable(
            data=results_df.to_dict("records"),
            columns=[{"name": col, "id": col} for col in results_df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'},
            page_size=10
        )

    except Exception as e:
        return dbc.Alert(f"Error processing file: {str(e)}", color="danger")

# Run the app
if __name__ == "__main__":
    print("Launching Dash app...")
    try:
        # app.run(debug=True, host='127.0.0.1', port=8050)
        app.run(debug=True)
    except Exception as e:
        print("Failed to start server:", e)