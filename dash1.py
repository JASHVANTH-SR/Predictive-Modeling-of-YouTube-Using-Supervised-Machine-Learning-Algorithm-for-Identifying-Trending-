import dash
from dash import dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Initialize the Dash app
app = dash.Dash(__name__)

# Load the saved model
best_model = joblib.load('best_model.pkl')

# Load the predictions
predictions_df = pd.read_csv('predictions.csv')

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("YouTube Trend Analysis Dashboard"),
    
    # Dropdown for selecting the prediction model
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'Best Model', 'value': 'best'},
        ],
        value='best',  # Default selected model
        style={'width': '50%'}
    ),
    
    # Scatter plot for actual vs. predicted views
    dcc.Graph(id='scatter-plot'),
    
    # Display RMSE, MAE, and R-squared
    html.Div([
        html.H3("Model Evaluation Metrics"),
        html.P(id='rmse-text'),
        html.P(id='mae-text'),
        html.P(id='r2-text')
    ])
])

# Define callback to update scatter plot and metrics
@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('rmse-text', 'children'),
     Output('mae-text', 'children'),
     Output('r2-text', 'children')],
    [Input('model-dropdown', 'value')]
)
def update_dashboard(selected_model):
    # Depending on the selected model, use the best_model for predictions
    if selected_model == 'best':
        model = best_model
    
    # Make predictions using the selected model
    X_test_scaled = scaler.transform(X_test)  # You need to transform the test data
    y_pred = model.predict(X_test_scaled)
    
    # Create a scatter plot of actual vs. predicted views
    scatter_fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Views', 'y': 'Predicted Views'}, title='Actual vs. Predicted Views')
    
    # Calculate RMSE, MAE, and R-squared
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return scatter_fig, f"RMSE: {rmse:.2f}", f"MAE: {mae:.2f}", f"R-squared: {r2:.2f}"

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
