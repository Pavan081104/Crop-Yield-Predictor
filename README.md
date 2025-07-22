![image](https://github.com/user-attachments/assets/a8a396d5-1d67-4e52-a38f-0816b22f8303)

# Overview

The Crop Yield Prediction System is a machine learning-powered web application designed to predict crop yields based on environmental, soil, and crop parameters. It helps farmers, researchers, and policymakers with accurate predictions, enabling better agricultural planning and productivity.

# Project Structure

├── app.py 

├── templates/

│   └── index.html

├── static/

│   ├── css/

│   └── js/

├── models/

│   └── trained_model.pkl

├── data/

│   └── mongo_ingested_data.csv

├── src/

│   ├── data_ingestion.py

│   ├── data_transformation.py

│   ├── model_trainer.py

│   ├── evaluation.py

│   └── prediction_pipeline.py

└── requirements.txt

# Installation
Clone the repository and install the dependencies:

git clone https://github.com/KVHarsha2611/crop_yield_predictor.git

cd crop-yield-predictor

pip install -r requirements.txt

Start the Flask application:

python app.py
Access the app by visiting: http://127.0.0.1:5000/

# Pipeline Overview
The project is modularly designed with independent components for ingestion, transformation, model training, evaluation, and prediction. Scikit-learn pipelines ensure a smooth flow from raw data to final predictions.

# Data Ingestion
The data_ingestion.py module handles data fetching from a MongoDB collection and saves it as a CSV file locally. This ensures the model always uses updated and clean data.

# Data Transformation
The data_transformation.py module processes the data by handling missing values, scaling numerical features, and encoding categorical variables using scikit-learn pipelines to ensure consistency between training and prediction stages.

# Model Training
The model_trainer.py module trains multiple regression models including Linear Regression, Ridge, Lasso, ElasticNet, Decision Tree, and Random Forest. After hyperparameter tuning, Random Forest achieved the best results.

# Evaluation
The evaluation.py module compares the model performances using R² Score and Root Mean Squared Error (RMSE). Random Forest achieved an R² score of 0.93+, making it the selected model for deployment.

# Prediction Pipeline
The prediction_pipeline.py connects the trained ML model to the Flask app. It takes user input from the frontend, processes it through the same transformation steps, and predicts the crop yield.

# How to Run
Ensure MongoDB is installed and connected properly if data ingestion is required.

Train the model using scripts inside src/ if not already trained.

Start the Flask server:

python app.py

Open http://127.0.0.1:5000/ in your browser.

Fill out the form with soil and climate parameters to get the predicted crop yield.

# Dataset
The dataset was sourced from MongoDB and contains:

Soil features: pH, Nitrogen (N), Phosphorus (P), Potassium (K)

Climate features: Temperature, Humidity, Rainfall

Crop name and State name

Historical crop yield values as the target

The dataset underwent preprocessing steps such as outlier removal, scaling, and encoding to ensure the highest model performance.
