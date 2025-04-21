import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import joblib

# Load the saved model and preprocessors
model = joblib.load('artifacts/best_housing_model.pkl')
scaler_X = joblib.load('artifacts/scaler_X.pkl')
training_columns = joblib.load('artifacts/training_columns.pkl')

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
app = dash.Dash(__name__, external_stylesheets=['https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css'])

# Define the layout
app.layout = html.Div(className='container mx-auto p-6 bg-gray-100 min-h-screen', children=[
    html.H1('House Price Prediction', className='text-3xl font-bold text-center mb-6 text-gray-800'),
    html.Div(className='grid grid-cols-1 md:grid-cols-2 gap-4 bg-white p-6 rounded-lg shadow-md', children=[
        # Input fields
        html.Div([
            html.Label('Bedrooms', className='block text-sm font-medium text-gray-700'),
            dcc.Input(id='bedrooms', type='number', value=3, min=1, className='mt-1 block w-full border-gray-300 rounded-md shadow-sm p-2')
        ]),
        html.Div([
            html.Label('Bathrooms', className='block text-sm font-medium text-gray-700'),
            dcc.Input(id='bathrooms', type='number', value=2, min=0.5, step=0.25, className='mt-1 block w-full border-gray-300 rounded-md shadow-sm p-2')
        ]),
        html.Div([
            html.Label('Square Footage (Living)', className='block text-sm font-medium text-gray-700'),
            dcc.Input(id='sqft_living', type='number', value=2000, min=100, className='mt-1 block w-full border-gray-300 rounded-md shadow-sm p-2')
        ]),
        html.Div([
            html.Label('Floors', className='block text-sm font-medium text-gray-700'),
            dcc.Input(id='floors', type='number', value=1, min=1, step=0.5, className='mt-1 block w-full border-gray-300 rounded-md shadow-sm p-2')
        ]),
        html.Div([
            html.Label('Waterfront (0 or 1)', className='block text-sm font-medium text-gray-700'),
            dcc.Input(id='waterfront', type='number', value=0, min=0, max=1, className='mt-1 block w-full border-gray-300 rounded-md shadow-sm p-2')
        ]),
        html.Div([
            html.Label('View (0-4)', className='block text-sm font-medium text-gray-700'),
            dcc.Input(id='view', type='number', value=0, min=0, max=4, className='mt-1 block w-full border-gray-300 rounded-md shadow-sm p-2')
        ]),
        html.Div([
            html.Label('Year Built', className='block text-sm font-medium text-gray-700'),
            dcc.Input(id='year_built', type='number', value=2000, min=1900, max=2014, className='mt-1 block w-full border-gray-300 rounded-md shadow-sm p-2')
        ]),
        html.Div([
            html.Label('Renovated (0 or 1)', className='block text-sm font-medium text-gray-700'),
            dcc.Input(id='is_renovated', type='number', value=0, min=0, max=1, className='mt-1 block w-full border-gray-300 rounded-md shadow-sm p-2')
        ]),
        html.Div([
            html.Label('Condition (1-5)', className='block text-sm font-medium text-gray-700'),
            dcc.Input(id='condition', type='number', value=3, min=1, max=5, className='mt-1 block w-full border-gray-300 rounded-md shadow-sm p-2')
        ]),
        html.Div([
            html.Label('Square Footage (Basement)', className='block text-sm font-medium text-gray-700'),
            dcc.Input(id='sqft_basement', type='number', value=0, min=0, className='mt-1 block w-full border-gray-300 rounded-md shadow-sm p-2')
        ]),
        html.Div([
            html.Label('City', className='block text-sm font-medium text-gray-700'),
            dcc.Dropdown(id='city', options=[{'label': city, 'value': city} for city in cities], value='Seattle', className='mt-1 block w-full border-gray-300 rounded-md shadow-sm p-2')
        ]),
        html.Div([
            html.Label('Zipcode', className='block text-sm font-medium text-gray-700'),
            dcc.Dropdown(id='zipcode', options=[{'label': str(z), 'value': z} for z in zipcodes], value=98103, className='mt-1 block w-full border-gray-300 rounded-md shadow-sm p-2')
        ]),
    ]),
    html.Button('Predict', id='predict-button', n_clicks=0, className='mt-4 w-full bg-blue-600 text-white font-semibold py-2 rounded-md hover:bg-blue-700'),
    html.Div(id='prediction-output', className='mt-6 text-center text-lg font-medium text-gray-800')
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
        return f"Predicted House Price: ${pred_price:,.2f}"
    return ""

# Run the app
if __name__ == '__main__':
    app.run(debug=True)