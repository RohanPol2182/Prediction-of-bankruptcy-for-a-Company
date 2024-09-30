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
0   Bankrupt?                                                 6819 non-null   int64  
 1    ROA(C) before interest and depreciation before interest  6819 non-null   float64
 2    ROA(A) before interest and % after tax                   6819 non-null   float64
 3    ROA(B) before interest and depreciation after tax        6819 non-null   float64
 4    Operating Gross Margin                                   6819 non-null   float64
 5    Realized Sales Gross Margin                              6819 non-null   float64
 6    Operating Profit Rate                                    6819 non-null   float64
 7    Pre-tax net Interest Rate                                6819 non-null   float64
 8    After-tax net Interest Rate                              6819 non-null   float64
 9    Non-industry income and expenditure/revenue              6819 non-null   float64
 10   Continuous interest rate (after tax)                     6819 non-null   float64
 11   Operating Expense Rate                                   6819 non-null   float64
 12   Research and development expense rate                    6819 non-null   float64
 13   Cash flow rate                                           6819 non-null   float64
 14   Interest-bearing debt interest rate                      6819 non-null   float64
 15   Tax rate (A)                                             6819 non-null   float64
 16   Net Value Per Share (B)                                  6819 non-null   float64
 17   Net Value Per Share (A)                                  6819 non-null   float64
 18   Net Value Per Share (C)                                  6819 non-null   float64
 19   Persistent EPS in the Last Four Seasons                  6819 non-null   float64
 20   Cash Flow Per Share                                      6819 non-null   float64
 21   Revenue Per Share (Yuan ¥)                               6819 non-null   float64
 22   Operating Profit Per Share (Yuan ¥)                      6819 non-null   float64
 23   Per Share Net profit before tax (Yuan ¥)                 6819 non-null   float64
 24   Realized Sales Gross Profit Growth Rate                  6819 non-null   float64
 25   Operating Profit Growth Rate                             6819 non-null   float64
 26   After-tax Net Profit Growth Rate                         6819 non-null   float64
 27   Regular Net Profit Growth Rate                           6819 non-null   float64
 28   Continuous Net Profit Growth Rate                        6819 non-null   float64
 29   Total Asset Growth Rate                                  6819 non-null   float64
 30   Net Value Growth Rate                                    6819 non-null   float64
 31   Total Asset Return Growth Rate Ratio                     6819 non-null   float64
 32   Cash Reinvestment %                                      6819 non-null   float64
 33   Current Ratio                                            6819 non-null   float64
 34   Quick Ratio                                              6819 non-null   float64
 35   Interest Expense Ratio                                   6819 non-null   float64
 36   Total debt/Total net worth                               6819 non-null   float64
 37   Debt ratio %                                             6819 non-null   float64
 38   Net worth/Assets                                         6819 non-null   float64
 39   Long-term fund suitability ratio (A)                     6819 non-null   float64
 40   Borrowing dependency                                     6819 non-null   float64
 41   Contingent liabilities/Net worth                         6819 non-null   float64
 42   Operating profit/Paid-in capital                         6819 non-null   float64
 43   Net profit before tax/Paid-in capital                    6819 non-null   float64
 44   Inventory and accounts receivable/Net value              6819 non-null   float64
 45   Total Asset Turnover                                     6819 non-null   float64
 46   Accounts Receivable Turnover                             6819 non-null   float64
 47   Average Collection Days                                  6819 non-null   float64
 48   Inventory Turnover Rate (times)                          6819 non-null   float64
 49   Fixed Assets Turnover Frequency                          6819 non-null   float64
 50   Net Worth Turnover Rate (times)                          6819 non-null   float64
 51   Revenue per person                                       6819 non-null   float64
 52   Operating profit per person                              6819 non-null   float64
 53   Allocation rate per person                               6819 non-null   float64
 54   Working Capital to Total Assets                          6819 non-null   float64
 55   Quick Assets/Total Assets                                6819 non-null   float64
 56   Current Assets/Total Assets                              6819 non-null   float64
 57   Cash/Total Assets                                        6819 non-null   float64
 58   Quick Assets/Current Liability                           6819 non-null   float64
 59   Cash/Current Liability                                   6819 non-null   float64
 60   Current Liability to Assets                              6819 non-null   float64
 61   Operating Funds to Liability                             6819 non-null   float64
 62   Inventory/Working Capital                                6819 non-null   float64
 63   Inventory/Current Liability                              6819 non-null   float64
 64   Current Liabilities/Liability                            6819 non-null   float64
 65   Working Capital/Equity                                   6819 non-null   float64
 66   Current Liabilities/Equity                               6819 non-null   float64
 67   Long-term Liability to Current Assets                    6819 non-null   float64
 68   Retained Earnings to Total Assets                        6819 non-null   float64
 69   Total income/Total expense                               6819 non-null   float64
 70   Total expense/Assets                                     6819 non-null   float64
 71   Current Asset Turnover Rate                              6819 non-null   float64
 72   Quick Asset Turnover Rate                                6819 non-null   float64
 73   Working capitcal Turnover Rate                           6819 non-null   float64
 74   Cash Turnover Rate                                       6819 non-null   float64
 75   Cash Flow to Sales                                       6819 non-null   float64
 76   Fixed Assets to Assets                                   6819 non-null   float64
 77   Current Liability to Liability                           6819 non-null   float64
 78   Current Liability to Equity                              6819 non-null   float64
 79   Equity to Long-term Liability                            6819 non-null   float64
 80   Cash Flow to Total Assets                                6819 non-null   float64
 81   Cash Flow to Liability                                   6819 non-null   float64
 82   CFO to Assets                                            6819 non-null   float64
 83   Cash Flow to Equity                                      6819 non-null   float64
 84   Current Liability to Current Assets                      6819 non-null   float64
 85   Liability-Assets Flag                                    6819 non-null   int64  
 86   Net Income to Total Assets                               6819 non-null   float64
 87   Total assets to GNP price                                6819 non-null   float64
 88   No-credit Interval                                       6819 non-null   float64
 89   Gross Profit to Sales                                    6819 non-null   float64
 90   Net Income to Stockholder's Equity                       6819 non-null   float64
 91   Liability to Equity                                      6819 non-null   float64
 92   Degree of Financial Leverage (DFL)                       6819 non-null   float64
 93   Interest Coverage Ratio (Interest expense to EBIT)       6819 non-null   float64
 94   Net Income Flag                                          6819 non-null   int64  
 95   Equity to Liability                                      6819 non-null   float64
 
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
