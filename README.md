# Personal Fitness Tracker

## Overview
This project is a **Personal Fitness Tracker** application built using **Streamlit** and **Machine Learning**. The application predicts the number of calories burned based on user input parameters such as age, BMI, duration, heart rate, body temperature, and gender.

![Image](https://github.com/user-attachments/assets/0a159189-c592-4ccf-9973-e5124aebdd76)

## Features
- Interactive **Streamlit UI** for real-time calorie burn predictions
- **Machine Learning Model** using Random Forest Regressor
- **Data visualization** and analytical insights
- **Comparison metrics** to evaluate user performance
- **Optimized performance** using caching techniques

## Tech Stack
### Programming Language & Framework:
- Python
- Streamlit

### Libraries Used:
- `numpy`, `pandas` – Data processing
- `matplotlib`, `seaborn` – Data visualization
- `sklearn` – Machine Learning model implementation

## Data Sources
The application uses two datasets:
- `calories.csv` – Contains calorie burn data
- `exercise.csv` – Contains exercise metrics

## Installation & Setup
### Prerequisites
Ensure you have **Python 3.x** installed.

### Steps to Run the Project
1. **Clone the repository:**
   ```sh
   git clone https://github.com/KaviyaSM/Personal_fitness_tracker.git
   cd Personal_fitness_tracker
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```sh
   streamlit run app.py
   ```

## Usage
1. Open the web application in your browser.
2. Use the **sidebar sliders** to input personal fitness parameters.
3. View predicted **calories burned** and analytical insights.
4. Compare your fitness metrics with dataset trends.

## Future Improvements
- **Enhance the model accuracy** with deep learning approaches.
- **Incorporate additional features** like step count & sleep tracking.
- **Add user authentication** to store and track fitness history.

## Author
**Kaviya S.M.**


