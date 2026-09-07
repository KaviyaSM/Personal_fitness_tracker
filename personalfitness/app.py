import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import time

import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Personal Fitness Tracker",
    page_icon="🏃‍♂️",
    layout="wide"
)

# Custom CSS to improve UI appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2,5rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1976D2;
        margin-top: 1rem;
    }
    .info-text {
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .highlight {
        color: #1976D2;
        font-weight: bold;
    }
    .result-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #1976D2;
    }
    .sidebar-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #1565C0;
    }
    .stProgress > div > div {
        background-color: #1976D2;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 style="font-size: 3.5rem; font-weight: bold; color: #1E88E5;">Personal Fitness Tracker</h1>', unsafe_allow_html=True)
st.markdown('<p class="info-text">This application predicts calories burned based on your physical parameters and exercise metrics. Adjust the parameters in the sidebar to see how they affect your calorie burn.</p>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.markdown('<p class="sidebar-header">User Input Parameters</p>', unsafe_allow_html=True)

def user_input_features():
    age = st.sidebar.slider("Age (years)", 10, 100, 30, help="Select your age")
    bmi = st.sidebar.slider("BMI", 15.0, 40.0, 20.0, 0.1, help="Body Mass Index")
    duration = st.sidebar.slider("Exercise Duration (minutes)", 0, 35, 15, help="How long did you exercise?")
    heart_rate = st.sidebar.slider("Heart Rate (bpm)", 60, 130, 80, help="Your heart rate during exercise")
    body_temp = st.sidebar.slider("Body Temperature (°C)", 36.0, 42.0, 38.0, 0.1, help="Your body temperature during exercise")
    
    gender_options = ["Male", "Female"]
    gender_icons = ["♂️", "♀️"]
    gender_index = st.sidebar.radio(
        "Gender",
        options=range(len(gender_options)),
        format_func=lambda x: f"{gender_icons[x]} {gender_options[x]}",
    )
    gender = 1 if gender_index == 0 else 0  # Male=1, Female=0

    # Use column names to match the training data
    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender
    }

    features = pd.DataFrame(data_model, index=[0])
    return features

# Get user inputs
df = user_input_features()

# Load and preprocess data
@st.cache_data
def load_data():
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
    
    exercise_df = exercise.merge(calories, on="User_ID")
    exercise_df.drop(columns="User_ID", inplace=True)
    
    return exercise_df

exercise_df = load_data()

# Split data and train model
@st.cache_resource
def train_model(exercise_df):
    exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

    # Add BMI column to both training and test sets
    for data in [exercise_train_data, exercise_test_data]:
        data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
        data["BMI"] = round(data["BMI"], 2)

    # Prepare the training and testing sets
    exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
    exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

    # Separate features and labels
    X_train = exercise_train_data.drop("Calories", axis=1)
    y_train = exercise_train_data["Calories"]

    X_test = exercise_test_data.drop("Calories", axis=1)
    y_test = exercise_test_data["Calories"]

    # Train the model
    random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
    random_reg.fit(X_train, y_train)
    
    return random_reg, X_train

model, X_train = train_model(exercise_df)

# Layout the dashboard in columns
col1, col2 = st.columns([2, 3])

with col1:
    st.markdown('<p class="section-header">Your Parameters</p>', unsafe_allow_html=True)
    
    # Format the dataframe display
    formatted_df = df.copy()
    formatted_df.index = ["Value"]
    formatted_df.columns = [
        "Age (years)", 
        "BMI", 
        "Duration (min)", 
        "Heart Rate (bpm)", 
        "Body Temp (°C)", 
        "Gender (Male=1, Female=0)"
    ]
    
    st.dataframe(formatted_df.T, use_container_width=True)
    
    # Prediction with progress
    st.markdown('<p class="section-header">Calories Burned</p>', unsafe_allow_html=True)
    
    # Progress animation
    progress_placeholder = st.empty()
    with progress_placeholder.container():
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(0.01)
    
    # Align prediction data columns with training data
    prediction_df = df.reindex(columns=X_train.columns, fill_value=0)
    
    # Make prediction
    prediction = model.predict(prediction_df)
    
    # Format and display the prediction
    st.markdown(
        f"""
        <div class="result-box">
            <h2 style="text-align: center; margin: 0;color:black;">{round(prediction[0], 2)}</h2>
            <p style="text-align: center; margin: 0; font-size: 1.2rem;color:black;">kilocalories</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

with col2:
    st.markdown('<p class="section-header">Comparison with Others</p>', unsafe_allow_html=True)
    
    # Create metrics for comparison
    col_a, col_b = st.columns(2)
    
    # Boolean logic for age, duration, etc., compared to the user's input
    boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
    boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
    boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
    boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()
    
    age_percentile = round(sum(boolean_age) / len(boolean_age) * 100, 1)
    duration_percentile = round(sum(boolean_duration) / len(boolean_duration) * 100, 1)
    hr_percentile = round(sum(boolean_heart_rate) / len(boolean_heart_rate) * 100, 1)
    temp_percentile = round(sum(boolean_body_temp) / len(boolean_body_temp) * 100, 1)
    
    with col_a:
        st.metric("Age Percentile", f"{age_percentile}%", 
                 f"Older than {age_percentile}% of others")
        st.metric("Heart Rate Percentile", f"{hr_percentile}%", 
                 f"Higher than {hr_percentile}% of others")
    
    with col_b:
        st.metric("Duration Percentile", f"{duration_percentile}%", 
                 f"Longer than {duration_percentile}% of others")
        st.metric("Body Temp Percentile", f"{temp_percentile}%", 
                 f"Higher than {temp_percentile}% of others")
    
    # Find and display similar results
    st.markdown('<p class="section-header">Similar Profiles</p>', unsafe_allow_html=True)
    
    # Progress animation
    similar_progress = st.empty()
    with similar_progress.container():
        similar_bar = st.progress(0)
        for i in range(100):
            similar_bar.progress(i + 1)
            time.sleep(0.01)
    
    # Find similar results based on predicted calories
    calorie_range = [prediction[0] - 10, prediction[0] + 10]
    similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & 
                              (exercise_df["Calories"] <= calorie_range[1])]
    
    # Convert gender to readable format
    display_data = similar_data.copy()
    display_data["Gender"] = display_data["Gender"].map({1: "Male", 0: "Female"})
    
    # Select columns to display
    cols_to_display = ["Gender", "Age", "Weight", "Height", "Duration", 
                       "Heart_Rate", "Body_Temp", "Calories"]
    
    st.dataframe(
        display_data[cols_to_display].sample(5).reset_index(drop=True),
        use_container_width=True,
        column_config={
            "Gender": st.column_config.TextColumn("Gender"),
            "Age": st.column_config.NumberColumn("Age (years)"),
            "Weight": st.column_config.NumberColumn("Weight (kg)"),
            "Height": st.column_config.NumberColumn("Height (cm)"),
            "Duration": st.column_config.NumberColumn("Duration (min)"),
            "Heart_Rate": st.column_config.NumberColumn("Heart Rate (bpm)"),
            "Body_Temp": st.column_config.NumberColumn("Body Temp (°C)"),
            "Calories": st.column_config.NumberColumn("Calories", format="%.1f")
        }
    )

# Add footer with information
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        <p>This fitness tracker uses a Random Forest model trained on exercise and calorie data.</p>
        <p>Data is for informational purposes only. Consult with a healthcare professional for personalized advice.</p>
    </div>
    """, 
    unsafe_allow_html=True
)