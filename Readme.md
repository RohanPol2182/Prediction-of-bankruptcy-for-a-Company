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
Bankrupt?
ROA(C) before interest and depreciation before interest
ROA(A) before interest and % after tax
ROA(B) before interest and depreciation after tax
Operating Gross Margin
Realized Sales Gross Margin
Operating Profit Rate
Pre-tax net Interest Rate
After-tax net Interest Rate
Non-industry income and expenditure/revenue
Continuous interest rate (after tax)
Operating Expense Rate
Research and development expense rate
Cash flow rate
Interest-bearing debt interest rate
Tax rate (A)
Net Value Per Share (B)
Net Value Per Share (A)
Net Value Per Share (C)
Persistent EPS in the Last Four Seasons
Cash Flow Per Share
Revenue Per Share (Yuan ¥)
Operating Profit Per Share (Yuan ¥)
Per Share Net profit before tax (Yuan ¥)
Realized Sales Gross Profit Growth Rate
Operating Profit Growth Rate
After-tax Net Profit Growth Rate
Regular Net Profit Growth Rate
Continuous Net Profit Growth Rate
Total Asset Growth Rate
Net Value Growth Rate
Total Asset Return Growth Rate Ratio
Cash Reinvestment %
Current Ratio
Quick Ratio
Interest Expense Ratio
Total debt/Total net worth
Debt ratio %
Net worth/Assets
Long-term fund suitability ratio (A)
Borrowing dependency
Contingent liabilities/Net worth
Operating profit/Paid-in capital
Net profit before tax/Paid-in capital
Inventory and accounts receivable/Net value
Total Asset Turnover
Accounts Receivable Turnover
Average Collection Days
Inventory Turnover Rate (times)
Fixed Assets Turnover Frequency
Net Worth Turnover Rate (times)
Revenue per person
Operating profit per person
Allocation rate per person
Working Capital to Total Assets
Quick Assets/Total Assets
Current Assets/Total Assets
Cash/Total Assets
Quick Assets/Current Liability
Cash/Current Liability
Current Liability to Assets
Operating Funds to Liability
Inventory/Working Capital
Inventory/Current Liability
Current Liabilities/Liability
Working Capital/Equity
Current Liabilities/Equity
Long-term Liability to Current Assets
Retained Earnings to Total Assets
Total income/Total expense
Total expense/Assets
Current Asset Turnover Rate
Quick Asset Turnover Rate
Working capitcal Turnover Rate
Cash Turnover Rate
Cash Flow to Sales
Fixed Assets to Assets
Current Liability to Liability
Current Liability to Equity
Equity to Long-term Liability
Cash Flow to Total Assets
Cash Flow to Liability
CFO to Assets
Cash Flow to Equity
Current Liability to Current Assets
Liability-Assets Flag
Net Income to Total Assets
Total assets to GNP price
No-credit Interval
Gross Profit to Sales
Net Income to Stockholder's Equity
Liability to Equity
Degree of Financial Leverage (DFL)
Interest Coverage Ratio (Interest expense to EBIT)
Net Income Flag
Equity to Liability
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


>Future Work

Experiment with other models: Explore and evaluate the performance of other time series models (SARIMA) and machine learning models (LSTM, Random Forest, XGBoost).
Hyperparameter tuning: Fine-tune the hyperparameters of the models to potentially improve their performance.
Incorporate additional features: Consider including other potentially relevant features such as macroeconomic indicators or sentiment analysis data.
Ensemble models: Combine the predictions of multiple models to potentially achieve better accuracy.
Deploy and monitor: Implement a system to make real-time predictions and monitor the model's performance over time, updating it as needed.
