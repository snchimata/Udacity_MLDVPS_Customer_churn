"""
Module to run unit tests on churn_library.py
Author: Sai Chimata
Date: August 2021
"""
import os
import logging
import churn_library as cls
from constants import CATEGORY_COLUMNS

logging.basicConfig(
    filename=os.path.join(os.path.dirname(__file__),'logs/churn_library.log'),
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions

    input:
            import_data: import_data function from churn_library
    output:
            dataframe (pandas dataframe): data
    '''
    try:
        dataframe = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
        return dataframe
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function

    input:
            perform_eda: perform_eda function from churn_library
    output:
            None
    '''
    try:
        dataframe = test_import(cls.import_data)
        perform_eda(dataframe)
        logging.info("Testing perform_eda: SUCCESS")
    except AttributeError as err:
        logging.error("Testing perform_eda: Input should be a dataframe")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper

    input:
            encoder_helper: encoder_helper function from churn_library
    output:
            dataframe (pandas dataframe): encoded data
    '''
    try:
        dataframe = test_import(cls.import_data)
        encoder_helper(dataframe, CATEGORY_COLUMNS, "Churn")
        logging.info("Testing encoder_helper: SUCCESS")
        return dataframe
    except BaseException:
        logging.error("Testing encoder_helper: encoder_helper failed")
        raise


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering

    input:
            perform_feature_engineering: perform_feature_engineering function from churn library
    output:
            x_train (pandas dataframe): X training data
            x_test  (pandas dataframe): X testing data
            y_train (pandas dataframe): y training data
            y_test  (pandas dataframe): y testing data
    '''
    try:
        dataframe = test_encoder_helper(cls.encoder_helper)
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            dataframe, "Churn")
        assert len(x_train) > 0
        assert len(x_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info("Testing test_perform_feature_engineering: SUCCESS")
        return x_train, x_test, y_train, y_test
    except AssertionError as err:
        logging.error(
            "Testing test_perform_feature_engineering: test_perform_feature_engineering failed")
        raise err


def test_train_models(train_models):
    '''
    test train_models

    input:
            train_models: train_models function from churn_library
    output:
            None
    '''
    try:
        x_train, x_test, y_train, y_test = test_perform_feature_engineering(
            cls.perform_feature_engineering)
        train_models(x_train, x_test, y_train, y_test)
        assert os.path.isfile('./models/rfc_model.pkl')
        assert os.path.isfile('./models/logistic_model.pkl')
        logging.info("Testing test_train_models: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing test_train_models: test_train_models failed")
        raise


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
