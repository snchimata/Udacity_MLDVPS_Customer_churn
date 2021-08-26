# Predict Customer Churn

- Project **Predict Customer Churn** of [ML DevOps Engineer Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821) by Udacity.


## Project Description

Customer attrition, also known as customer churn, is the loss of clients or customers. Businesses measure and track churn as a percentage of lost customers compared to total number of customers over a given time period. Identifying and handling the customers about to churn can improve overall outcomes of the business.

This project seeks to answer the customer churn that is happening in the banking industry using clean code principles

The project is divided into the following sections:

- Import Data
- Exploratory data analysis and visualizations
- Feature engineering to transform data
- Building Models and evaulating performance

## Project Components

There are two components in this project:

### 1. Churn Library

File _churn_library.py_:

- Loads the `bank_data` dataset
- Performs EDA and Visualizations
- Feature Engineering
- Model and Evaluation


### 2. Churn Script Logging and Testing

File _churn_script_logging_and_testing.py_:

- Tests churn library
- Logs activity


## Running

### Dependencies

List of libraries used for this project:

```
autopep8==1.5.7
joblib==0.11
matplotlib==2.1.0
numpy==1.12.1
pandas==0.23.3
pylint==2.9.6
scikit-learn==0.22
seaborn==0.8.1
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the dependencies from the `requirements.txt`

```
pip install -r requirements.txt
```
### Modeling

Run the following command to execute the main script
```
python churn_library.py
``` 
script execution generates
- EDA plots are available in directory ```./images/eda/```
- Model metrics are available in directory ```./images/results/```
- Model pickle files are available in directory ```./models/```
- Log files are available in directory ```./logs/results.log``` 

### Testing and Logging

Run the following command to run the tests script 
```
python churn_script_logging_and_tests.py
```

script execution generates
- Log file ```./logs/results.log```

## License
Distributed under the [MIT](https://choosealicense.com/licenses/mit/) License. See ```LICENSE``` for more information.