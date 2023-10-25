import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys  # Import the sys module
from sklearn.tree import export_text  # Import export_text from sklearn.tree

# Check if a command-line argument (file path) is provided
if len(sys.argv) != 2:
    print("Usage: python mlpro.py <file_path>")
    sys.exit(1)

# Get the file path from the command line
file_path = sys.argv[1]

# Load the CSV file
try:
    csv_data = pd.read_csv(file_path, encoding='latin1')
except FileNotFoundError:
    print("File not found. Please provide a valid file path.")
    sys.exit(1)

# Handling missing values
csv_data['description'].fillna('', inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
csv_data['category_id'] = label_encoder.fit_transform(csv_data['category_id'])
csv_data['channel_title'] = label_encoder.fit_transform(csv_data['channel_title'])

# Split the data into features (X) and target (y)
numeric_features = csv_data.select_dtypes(include=['number']).columns.tolist()
X = csv_data[numeric_features].drop(['views'], axis=1)
y = csv_data['views']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize regression models
linear_reg = LinearRegression()
decision_tree_reg = DecisionTreeRegressor()
random_forest_reg = RandomForestRegressor()
gradient_boost_reg = GradientBoostingRegressor()

models = [linear_reg, decision_tree_reg, random_forest_reg, gradient_boost_reg]
model_names = ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Booster']
best_model = None
best_rmse = float('inf')

# Train and evaluate models
for model, name in zip(models, model_names):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"{name} RMSE: {rmse}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model

print(f"Best Model: {type(best_model).__name__}, RMSE: {best_rmse}")

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("YouTube Trend Analysis Dashboard"),

    # Dropdown for selecting the prediction model
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'Linear Regression', 'value': 'Linear Regression'},
            {'label': 'Decision Tree', 'value': 'Decision Tree'},
            {'label': 'Random Forest', 'value': 'Random Forest'},
            {'label': 'Gradient Booster', 'value': 'Gradient Booster'}  # Added Gradient Booster option
        ],
        value='Linear Regression',  # Default selected model
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
    ]),

    # Additional visualizations
    dcc.Graph(id='histogram-actual-views'),
    dcc.Graph(id='histogram-predicted-views'),
    dcc.Graph(id='box-plot-actual-views'),
    dcc.Graph(id='box-plot-predicted-views'),

    # Content Strategy Insights
    html.H2("Content Strategy Insights"),
    html.P("Use the insights below to optimize your content strategy."),

    # Add content strategy visualizations here
    dcc.Graph(id='content-strategy-visualization'),

    # Educational Tool
    html.H2("Educational Tool"),
    html.P("This tool can be used for educational purposes in data science and machine learning courses."),
    
    # Input fields for feature values
    html.Label("Input Feature Values:"),
    dcc.Input(id='input-feature1', type='number', placeholder='Feature 1'),
    dcc.Input(id='input-feature2', type='number', placeholder='Feature 2'),
    dcc.Input(id='input-feature3', type='number', placeholder='Feature 3'),
    dcc.Input(id='input-feature4', type='number', placeholder='Feature 4'),
    dcc.Input(id='input-feature5', type='number', placeholder='Feature 5'),
    # Add more input fields for other features as needed
    
    # Button to trigger prediction
    html.Button('Predict Views', id='predict-button', n_clicks=0),
    
    # Display predicted views
    html.Div(id='predicted-views-output'),

    # How to Use the Dashboard Section
    html.H2("How to Use the Dashboard"),
    html.P("Welcome to the YouTube Trend Analysis Dashboard. This dashboard provides insights into YouTube video trends and offers tools for data analysis and prediction."),
    
    # Model Selection Instructions
    html.H3("1. Model Selection"),
    html.P("Choose a prediction model from the dropdown menu. The 'Linear Regression' model is selected by default. You can choose other models like Decision Tree, Random Forest, or Gradient Booster."),
    
    # Visualizations Instructions
    html.H3("2. Visualizations"),
    html.P("Explore the scatter plot, histograms, and box plots to understand the relationship between actual and predicted views. These visualizations will help you assess the performance of the selected model."),
    
    # Content Strategy Insights Instructions
    html.H3("3. Content Strategy Insights"),
    html.P("Gain insights into your content strategy based on the selected model. Different models provide different insights. You can analyze feature importances, decision tree visualizations, and more to optimize your content strategy."),
    
    # Educational Tool Instructions
    html.H3("4. Educational Tool"),
    html.P("Use the Educational Tool below to predict views for your videos. Enter feature values in the input fields and click 'Predict Views'. The tool will use the best-performing model to make predictions."),
])

# Define callback to update scatter plot and metrics
@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('rmse-text', 'children'),
     Output('mae-text', 'children'),
     Output('r2-text', 'children'),
     Output('histogram-actual-views', 'figure'),
     Output('histogram-predicted-views', 'figure'),
     Output('box-plot-actual-views', 'figure'),
     Output('box-plot-predicted-views', 'figure')],
    [Input('model-dropdown', 'value')]
)
def update_dashboard(selected_model):
    # Depending on the selected model, load the corresponding predictions
    if selected_model == 'Linear Regression':
        model = linear_reg
    elif selected_model == 'Decision Tree':
        model = decision_tree_reg
    elif selected_model == 'Random Forest':
        model = random_forest_reg
    elif selected_model == 'Gradient Booster':
        model = gradient_boost_reg

    # Make predictions using the selected model
    y_pred = model.predict(X_test_scaled)

    # Create a scatter plot of actual vs. predicted views
    scatter_fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Views', 'y': 'Predicted Views'},
                            title='Actual vs. Predicted Views')

    # Calculate RMSE, MAE, and R-squared
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Create histograms and box plots for actual and predicted views
    histogram_actual_views = px.histogram(x=y_test, title='Histogram of Actual Views')
    histogram_predicted_views = px.histogram(x=y_pred, title='Histogram of Predicted Views')
    box_plot_actual_views = px.box(x=y_test, title='Box Plot of Actual Views')
    box_plot_predicted_views = px.box(x=y_pred, title='Box Plot of Predicted Views')

    return scatter_fig, f"RMSE: {rmse:.2f}", f"MAE: {mae:.2f}", f"R-squared: {r2:.2f}", histogram_actual_views, histogram_predicted_views, box_plot_actual_views, box_plot_predicted_views

# Add callback for content strategy visualization
@app.callback(
    Output('content-strategy-visualization', 'figure'),
    [Input('model-dropdown', 'value')])
def update_content_strategy_visualization(selected_model):
    # Depending on the selected model, provide content strategy insights
    if selected_model == 'Linear Regression':
        # Example: Create a bar chart of feature importances for linear regression
        feature_importances = linear_reg.coef_
        feature_names = X.columns
        bar_fig = px.bar(x=feature_names, y=feature_importances, labels={'x': 'Feature', 'y': 'Importance'},
                         title='Feature Importances (Linear Regression)')
        return bar_fig
    elif selected_model == 'Decision Tree':
        # Visualize the decision tree as text
        decision_tree_text = export_text(decision_tree_reg, feature_names=X.columns.tolist())
        markdown_text = f'```{decision_tree_text}```'
        return px.scatter(title="Decision Tree Visualization", text=[markdown_text])
    elif selected_model == 'Random Forest':
        # Provide content strategy insights for Random Forest
        random_forest_feature_importance = random_forest_reg.feature_importances_
        feature_names = X.columns
        bar_fig = px.bar(x=feature_names, y=random_forest_feature_importance, labels={'x': 'Feature', 'y': 'Importance'},
                         title='Random Forest Feature Importance')
        return bar_fig
    elif selected_model == 'Gradient Booster':
        # Provide content strategy insights for Gradient Booster
        gradient_boost_feature_importance = gradient_boost_reg.feature_importances_
        feature_names = X.columns
        bar_fig = px.bar(x=feature_names, y=gradient_boost_feature_importance, labels={'x': 'Feature', 'y': 'Importance'},
                         title='Gradient Booster Feature Importance')
        return bar_fig
    else:
        return None
# Define callback to handle predictions from the educational tool
@app.callback(
    Output('predicted-views-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [dash.dependencies.State('input-feature1', 'value'),
     dash.dependencies.State('input-feature2', 'value'),
     dash.dependencies.State('input-feature3', 'value'),
     dash.dependencies.State('input-feature4', 'value'),
     dash.dependencies.State('input-feature5', 'value')]
)
def predict_views(n_clicks, feature1, feature2, feature3, feature4, feature5):
    if n_clicks > 0:
        # Use the best model for prediction
        model = best_model

        # Prepare input features for prediction
        input_features = [[feature1, feature2, feature3, feature4, feature5]]

        # Scale input features
        input_features_scaled = scaler.transform(input_features)

        # Make predictions
        predictions = model.predict(input_features_scaled)

        return html.Div([
            html.H3("Predicted Views"),
            html.P(f"Feature 1: {feature1}"),
            html.P(f"Feature 2: {feature2}"),
            html.P(f"Feature 3: {feature3}"),
            html.P(f"Feature 4: {feature4}"),
            html.P(f"Feature 5: {feature5}"),
            html.P(f"Predicted Views: {predictions[0]:,.2f}")
        ])

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
