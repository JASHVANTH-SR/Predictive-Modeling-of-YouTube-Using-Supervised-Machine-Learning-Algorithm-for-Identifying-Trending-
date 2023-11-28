import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import seaborn as sns

# Dictionary mapping country codes to country names
country_codes_to_names = {
    "CA": "Canada",
    "DE": "Germany",
    "FR": "France",
    "GB": "United Kingdom",
    "IN": "India",
    "JP": "Japan",
    "KR": "South Korea",
    "MX": "Mexico",
    "RU": "Russia",
    "US": "United States"
}

# Function to load JSON data
@st.cache_data
def load_json_data(json_file):
    json_data = json_file.read()  # Read the content of the file buffer
    data = json.loads(json_data)  # Parse the JSON data
    return data

# Function to load CSV data
@st.cache_data
def load_csv_data(csv_file):
    return pd.read_csv(csv_file, encoding='latin1')

# Function to extract categories from JSON data
def extract_categories(json_data):
    categories = {}
    for item in json_data["items"]:
        category_id = int(item["id"])
        category_title = item["snippet"]["title"]
        categories[category_id] = category_title
    return categories

# Function to display extracted video categories
def display_categories(categories):
    st.subheader("YouTube Video Categories:")
    for category_id, category_title in categories.items():
        st.write(f"Category ID: {category_id}, Title: {category_title}")
def run_dash_app(file_path):
    cmd = ["python", "mlpro.py", file_path]  # Replace with the actual filename of your Dash app
    subprocess.Popen(cmd, shell=True)

# Add an input field to provide the CSV file path

# Sidebar
st.sidebar.title("Select Country")
selected_country_code = st.sidebar.selectbox("Choose a country:", list(country_codes_to_names.keys()))
selected_country = country_codes_to_names.get(selected_country_code)

# Upload JSON file
uploaded_json = st.sidebar.file_uploader("Upload JSON file", type=["json"])

# Upload CSV file
uploaded_csv = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Check if a country is selected and both files are uploaded for the same country
if selected_country and uploaded_json and uploaded_csv:
    # Get the selected country's JSON and CSV file names
    json_file_name = f"{selected_country_code}_category_id.json"
    csv_file_name = f"{selected_country_code}videos.csv"
    
    # Check if the uploaded files have the correct names
    if (os.path.basename(uploaded_json.name) == json_file_name) and (os.path.basename(uploaded_csv.name) == csv_file_name):
        # Load JSON data
        json_data = load_json_data(uploaded_json)

        # Load CSV data
        csv_data = load_csv_data(uploaded_csv)


        # Display categories
        categories = extract_categories(json_data)
        display_categories(categories)

        # Display CSV data
        st.subheader("CSV Data")
        st.write(csv_data)

        # Display JSON data
        st.subheader("JSON Data")
        st.json(json_data)


        st.header("Exploratory Data Analysis (EDA) - CSV Data")

        # Summary statistics
        st.subheader("Summary Statistics for CSV Data")
        st.write(csv_data.describe())

        # Data types
        st.subheader("Data Types for CSV Columns")
        st.write(csv_data.dtypes)

        # Missing values
        st.subheader("Missing Values in CSV Data")
        missing_values = csv_data.isnull().sum()
        st.write(missing_values)

        # Visualize the distribution of views
        st.subheader("Distribution of Views (CSV Data)")
        fig_views = px.histogram(csv_data, x="views", nbins=50, title="Distribution of Views")
        st.plotly_chart(fig_views)

        # Visualize the distribution of likes
        st.subheader("Distribution of Likes (CSV Data)")
        fig_likes = px.histogram(csv_data, x="likes", nbins=50, title="Distribution of Likes")
        st.plotly_chart(fig_likes)

        # Visualize the distribution of dislikes
        st.subheader("Distribution of Dislikes (CSV Data)")
        fig_dislikes = px.histogram(csv_data, x="dislikes", nbins=50, title="Distribution of Dislikes")
        st.plotly_chart(fig_dislikes)
	
        st.subheader("Video Categories Distribution")
        category_distribution = csv_data['category_id'].value_counts().reset_index()
        category_distribution.columns = ['Category ID', 'Count']
        fig = px.pie(category_distribution, values='Count', names='Category ID', title='Video Categories Distribution')
        st.plotly_chart(fig)
        
	# Publish Time Analysis
        st.subheader("Publish Time Analysis")
        csv_data['publish_time'] = pd.to_datetime(csv_data['publish_time'])
        csv_data['publish_hour'] = csv_data['publish_time'].dt.hour
        publish_time_counts = csv_data['publish_hour'].value_counts().sort_index().reset_index()
        publish_time_counts.columns = ['Hour', 'Count']
        fig = px.bar(publish_time_counts, x='Hour', y='Count', title='Video Publish Hour Distribution')
        st.plotly_chart(fig)


	# Perform EDA on JSON data
        st.header("Exploratory Data Analysis (EDA) - JSON Data")

        # Visualize the distribution of video categories
        st.subheader("Distribution of Video Categories (JSON Data)")
        category_counts = csv_data["category_id"].value_counts()
        fig_categories = px.bar(x=category_counts.index, y=category_counts.values, title="Video Category Distribution")
        fig_categories.update_xaxes(title="Category ID")
        fig_categories.update_yaxes(title="Count")
        st.plotly_chart(fig_categories)

        # Visualize the top trending videos
        st.subheader("Top Trending Videos (JSON Data)")
        top_trending_videos = csv_data.nlargest(10, "views")
        st.write(top_trending_videos)

        # Visualize the correlation between views, likes, and dislikes
        st.subheader("Correlation Between Views, Likes, and Dislikes (CSV Data)")
        corr_matrix = csv_data[["views", "likes", "dislikes"]].corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="Viridis"
        ))
        st.plotly_chart(fig_corr)
	    
        url = st.text_input("Enter URL:")
        button_clicked = st.button("Run Dash App")

        # Check if the button is clicked
        if button_clicked:
                    # Check if URL is provided
                    if url:
                        # Run the Dash app with the specified URL
                        run_dash_app(url)
                    else:
                        st.warning("Please enter a valid URL.")	
	
	
        

	
    else:
        st.warning(f"Please upload the {selected_country} JSON and CSV files.")
else:
    st.warning("Please select a country from the sidebar and upload both JSON and CSV files.")

     
