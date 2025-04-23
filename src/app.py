import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import joblib

# Load the saved model and preprocessors
model = joblib.load('../artifacts/best_housing_model.pkl')
scaler_X = joblib.load('../artifacts/scaler_X.pkl')
training_columns = joblib.load('../artifacts/training_columns.pkl')

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
                    html.Div(id="prediction-output", className="mt-4 text-center text-success fs-5")
                ]),
                className="shadow-sm p-4"
            )
        ], width=10, lg=8, xl=6, className="mx-auto")
    ])
])

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
     State('zipcode', 'value')]
)
def predict_price(n_clicks, bedrooms, bathrooms, sqft_living, floors, waterfront, view, 
                  year_built, is_renovated, condition, sqft_basement, city, zipcode):
    if n_clicks > 0:
        # --- Validation Block ---
        errors = []

        if year_built is None or not (1900 <= year_built <= 2014):
            errors.append("Year built must be between 1900 and 2014.")

        if bedrooms is None or bedrooms < 1:
            errors.append("Number of bedrooms must be at least 1.")

        if bathrooms is None or bathrooms < 0.5:
            errors.append("Number of bathrooms must be at least 0.5.")

        if sqft_living is None or sqft_living < 100:
            errors.append("Living area must be at least 100 sqft.")

        if floors is None or floors < 1:
            errors.append("Number of floors must be at least 1.")

        if sqft_basement is None:
            errors.append("Basement square footage must be provided.")
        elif sqft_basement < 0:
            errors.append("Basement square footage cannot be negative.")

        if condition is None or condition not in [1, 2, 3, 4, 5]:
            errors.append("Condition must be between 1 and 5.")

        if view is None or view not in [0, 1, 2, 3, 4]:
            errors.append("View must be between 0 and 4.")

        if city not in cities:
            errors.append("Invalid city selected.")

        if zipcode not in zipcodes:
            errors.append("Invalid zipcode selected.")

        if errors:
            return dbc.Alert("\n".join(errors), color="danger", dismissable=True)
        
        # Create input DataFrame
        input_data = pd.DataFrame({
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'sqft_living': [sqft_living],
            'floors': [floors],
            'waterfront': [waterfront],
            'view': [view],
            'house_age': [2014 - year_built],
            'is_renovated': [is_renovated],
            'condition': [condition],
            'sqft_basement': [sqft_basement],
            'sqft_living_bathrooms': [sqft_living * bathrooms],
            'city': [city],
            'zipcode': [zipcode]
        })

        # One-hot encode categorical variables
        input_data = pd.get_dummies(input_data, columns=['city', 'zipcode'], prefix=['city', 'zipcode'])

        # Align with training columns
        for col in training_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[training_columns]

        # Scale numerical features
        numerical_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'house_age', 
                         'condition', 'sqft_basement', 'sqft_living_bathrooms']
        input_data[numerical_cols] = scaler_X.transform(input_data[numerical_cols])

        # Predict price
        pred_price = model.predict(input_data)[0]
        # return f"Predicted House Price: ${pred_price:,.2f}"
        return dbc.Alert(f"Predicted House Price: ${pred_price:,.2f}", color="success", dismissable=True)
    return ""

# Run the app
if __name__ == '__main__':
    app.run(debug=True)