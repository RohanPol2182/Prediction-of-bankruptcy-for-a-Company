# Predicting Bankruptcy for a Company

## Project overview:

The goal is to use a supervised classification model to check whether an organization is bankrupt or not, through leveraging various financial ratios with respect to the companys macro/microecenomic factors. 
![image](https://github.com/user-attachments/assets/391fd427-185e-48f5-9677-2a2aec072878)


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

![image](https://github.com/user-attachments/assets/a33347bb-cf76-4212-889c-5197cb1076e4)   
Frequency of Bankrupt companies  
  
![image](https://github.com/user-attachments/assets/7fa9678d-3668-485c-bad3-33e40ad03028)  
Net income to Total Assets vs Bankrupt classes (0,1)  
  
![image](https://github.com/user-attachments/assets/fc47a200-95f0-48f0-ac8b-f6ce10cca9ec)  
Distribution of Net income to total assets  

![image](https://github.com/user-attachments/assets/c39bc5bb-7155-4432-821d-18373a28a910)  
Distribution of Borrowing dependency  

![image](https://github.com/user-attachments/assets/0f8db311-924a-40b6-b44d-418f04d2600d)
Dustribution of Total Assets to GNP


>Modeling and Prediction

Split the data into training and testing sets.
Trained and evaluated a Random Forest Classifier model to predict bankruptcy for other organizations with similar data.

>Evaluation

Models were evaluated based on their precision, recall, F1 scores, yielding these results:  
![image](https://github.com/user-attachments/assets/4cb38994-a298-4852-a1d7-2d6dc692c807)


>Future Work
Evaluate the performance using other classification algorithms (XGBCV, RF)
Hyperparameter tuning: Fine-tune the hyperparameters of the models to potentially improve their performance.
Improving the feature engineering by considering other parameters to build the model.
Ensemble models: Combine the predictions of multiple models to potentially achieve better accuracy.
Deploy and monitor: Implement a system to make real-time predictions and monitor the model's performance over time, updating it as needed.
