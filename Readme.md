# Predicting Bankruptcy for a Company

## Project overview:

The goal is to use a supervised classification model to check whether an organization is bankrupt or not, through leveraging various financial ratios with respect to the companys macro/microecenomic factors. 

## Tools used: 

>Languages: Python

>Libraries: Pandas, Matplotlib, Seaborn, NumPy, Scikitlearn

>IDE: Jupyter Notebook

## Data Source: 

The dataset was obtained from Kaggle (https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction?resource=download)
The Dataset is 100% cleaned, with no null values and consistent formatting.

## Information about the dataset:

The Dataset Contains Various Financial Ratios, as follows:  

![image](https://github.com/user-attachments/assets/23d6ec36-aefd-4401-8718-926ce03c5e5d)

 
## Analysis:

>Exploratory Data Analysis (EDA)

Calculated and visualized descriptive statistics of the organizations data.
Plotted the ratios individually to determine the impacting features.
Visualized the calculated technical ratios to gain further insights.

>Modeling and Prediction

Split the data into training and testing sets.
Trained and evaluated a Random Forest Classifier model to predict bankruptcy for other organizations with similar data.

>Evaluation

Models were evaluated based on their precision, recall, F1 scores, yielding these results:  
![image](https://github.com/user-attachments/assets/4cb38994-a298-4852-a1d7-2d6dc692c807)


>Future Work

Experiment with other models: Explore and evaluate the performance of other time series models (SARIMA) and machine learning models (LSTM, Random Forest, XGBoost).
Hyperparameter tuning: Fine-tune the hyperparameters of the models to potentially improve their performance.
Incorporate additional features: Consider including other potentially relevant features such as macroeconomic indicators or sentiment analysis data.
Ensemble models: Combine the predictions of multiple models to potentially achieve better accuracy.
Deploy and monitor: Implement a system to make real-time predictions and monitor the model's performance over time, updating it as needed.
