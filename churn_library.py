# library doc string
'''
Collection of functions to do Customer Churn analysis and Modeling
Author: Sai Chimata
Date: August 2021
'''

# import libraries

import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from constants import IMAGES_DIR, RESULTS_DIR, CATEGORY_COLUMNS, KEEP_COLUMNS
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()

#os.system('pip install --user pandas_profiling')
#from pandas_profiling import ProfileReport
logging.basicConfig(
    filename=os.path.join(os.path.dirname(__file__), 'logs/results.log'),
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            dataframe: pandas dataframe
    '''
    try:
        dataframe = pd.DataFrame(pd.read_csv(pth))
        logging.info(
            "SUCCESS: Read file at %s with %s rows",
            pth,
            dataframe.shape[0])
    except FileNotFoundError as err:
        logging.error("ERROR: Failed to read file at %s", pth)
        raise err

    try:
        assert 'Attrition_Flag' in dataframe.columns
        assert dataframe["Attrition_Flag"].shape[0] > 0

        dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        return dataframe

    except AssertionError as err:
        logging.error("Creation of Churn Column Failed")
        raise err


def perform_eda(dataframe):
    '''
    perform eda on dataframe and save figures to images folder
    input:
            dataframe: pandas dataframe

    output:
            None
    '''

    logging.info('Performing exploratory data analysis...')

    try:
        plot_columns = [
            "Churn",
            "Customer_Age",
            "Heat_Map",
            "Marital_Status",
            "Total_Trans_Ct"
        ]

        # Create plots for features
        for column in plot_columns:

            # Ignore Assertion for Heat Map
            if column != "Heat_Map":
                assert column in dataframe.columns
                assert dataframe[column].shape[0] > 0

            logging.info("Creating plot for %s", column)

            plt.figure(figsize=(20, 10))
            if column in CATEGORY_COLUMNS:
                dataframe[column].value_counts('normalize').plot(kind='bar')
            elif column == "Total_Trans_Ct":
                sns.distplot(dataframe['Total_Trans_Ct'])
            elif column == "Heat_Map":
                sns.heatmap(
                    dataframe.corr(),
                    annot=False,
                    cmap='Dark2_r',
                    linewidths=2)
            else:
                dataframe[column].hist()

            # Save plot
            plot_saved_dir = f'{IMAGES_DIR}{column}_plot.png'
            logging.info(
                "Saving plot for %s in %s",
                column,
                plot_saved_dir)
            plt.savefig(plot_saved_dir)
            logging.info("SUCCESS: EDA plots successfully generated")
    except AssertionError as err:
        logging.error("ERROR: Creation of EDA plots failed for %s", column)
        raise err


def encoder_helper(dataframe, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name

    output:
            dataframe: pandas dataframe with new columns for
    '''
    for category in category_lst:
        try:
            logging.info(
                "Encoding categorical variable %s",
                category)
            category_group = dataframe.groupby(category).mean()[response]
            dataframe[f'{category}_{response}'] = dataframe[category].apply(
                lambda x: category_group.loc[x])
        except KeyError as err:
            logging.error(
                "ERROR: Encoding of categorical column %s failed",
                category)
            raise err
    logging.info("SUCCESS: Encoding of categorical data complete")
    return dataframe


def perform_feature_engineering(dataframe, response, test_size=0.3):
    '''
    input:
              dataframe: pandas dataframe
              response: string of response name

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    try:
        assert response in dataframe.columns
        assert dataframe[response].shape[0] > 0
        logging.info(
            'Performing feature engineering...')
        logging.info('Target feature is: %s',response)
        y_df = dataframe[response]
        x_df = dataframe[KEEP_COLUMNS]
        x_train, x_test, y_train, y_test = train_test_split(
            x_df, y_df, test_size=test_size, random_state=42)
        return x_train, x_test, y_train, y_test
        logging.info("SUCCESS: Feature engineering complete")
    except AssertionError as err:
        logging.error("ERROR: Feature engineering failed")
        raise err


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                output_pth):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
            output_pth: path to store the figure

    output:
             None
    '''
    # Random Forest model classification report
    rfr_filename = 'random_forest_report.jpg'
    plt.figure(figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_test, y_test_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(f'{output_pth}/{rfr_filename}', bbox_inches='tight')
    logging.info('SUCCESS: Saved Random Forest Classification Report')

    # plots Logistic Regression model classification Report
    lrr_filename = 'logistic_regression_report.jpg'
    plt.figure(figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_test, y_test_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(f'{output_pth}/{lrr_filename}', bbox_inches='tight')
    logging.info(
        'SUCCESS: Saved Logistic Regression Classification Report')


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    feature_importance_filename = 'feature_importance_plot.jpg'

    file_path = f'{output_pth}/{feature_importance_filename}'
    plt.savefig(file_path, bbox_inches='tight')
    logging.info("SUCCESS: Saving plot for feature importance in %s",
                 file_path)


def roc_plot(lr_model, rfc_model, x_test, y_test, output_pth):
    '''
        produces and Saves the ROC results for models
        input:
            lr_model: Logistic Regression model
            rfc_model:  Random Forest model
            x_test: Test data
            y_test: Test data response values
            output_pth: path to store the figure
        output:
            None
    '''
    roc_filename = 'roc_plot.jpg'
    lrc_plot = plot_roc_curve(lr_model, x_test, y_test)
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    rfc_plot = plot_roc_curve(rfc_model, x_test, y_test, ax=axis, alpha=0.8)
    lrc_plot.plot(ax=axis, alpha=0.8)
    file_path = f'{output_pth}/{roc_filename}'
    plt.savefig(file_path)
    logging.info(
        "SUCCESS: Saving plot for ROC in %s",
        file_path)


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    logging.info('Initialising Random Forest classifier...')
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    logging.info('Training RFC. Please wait...')
    cv_rfc.fit(x_train, y_train)

    logging.info('Initialising Logistic Regression...')
    lrc = LogisticRegression(solver='liblinear')
    logging.info('Training LR. Please wait...')
    lrc.fit(x_train, y_train)

    # store models
    try:
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        joblib.dump(lrc, './models/logistic_model.pkl')
        logging.info('Models saved successfully.')
    except Exception as err:
        logging.error('Error saving models')
        logging.exception('churn_library')
        raise err

    # run predictions
    train_preds_lr = lrc.predict(x_train)
    test_preds_lr = lrc.predict(x_test)

    train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    # reporting and plotting results
    classification_report_image(y_train, y_test, train_preds_lr,
                                train_preds_rf, test_preds_lr,
                                test_preds_rf, output_pth=RESULTS_DIR)

    feature_importance_plot(
        cv_rfc.best_estimator_,
        x_test,
        output_pth=RESULTS_DIR)
    roc_plot(
        lrc,
        cv_rfc.best_estimator_,
        x_test,
        y_test,
        output_pth=RESULTS_DIR)


if __name__ == '__main__':
    try:

        # Load Data
        data = import_data('data/bank_data.csv')

        # Perform EDA on the Dataframe
        perform_eda(data)

        # Encode Categorical Data
        encoder_helper(data, CATEGORY_COLUMNS, "Churn")

        # Perform Feature engineering to split data
        X_train_df, X_test_df, y_train_df, y_test_df = perform_feature_engineering(
            data, "Churn")

        # Model and Scoring
        train_models(X_train_df, X_test_df, y_train_df, y_test_df)
        logging.info('SUCCESS - Model training and scoring complete')
    except BaseException:
        logging.error("Model Training Failed")
        raise
