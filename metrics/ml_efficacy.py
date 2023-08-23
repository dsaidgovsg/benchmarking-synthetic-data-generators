"""Get Machine learing efficacy"""
from sdmetrics.single_table import (BinaryAdaBoostClassifier,
                                    BinaryDecisionTreeClassifier,
                                    BinaryLogisticRegression,
                                    BinaryMLPClassifier, LinearRegression,
                                    MLPRegressor)


def compute_ml_classification(test_data, train_data, target_column, metadata, ml_model):
    """
    Get the classification score using a specified machine learning model.

    Parameters:
        test_data (pandas.DataFrame): Test dataset.
        train_data (pandas.DataFrame): Train dataset.
        target_column (str): Name of the target column to predict.
        metadata (dict): Additional metadata for the model.
        ml_model (str): Machine learning model identifier.

    Returns:
        classification_score (float): Classification score for the specified model.
    """
    # Map ml_model identifier to corresponding classifier
    if ml_model == "adaboost":
        classifier = BinaryAdaBoostClassifier
    elif ml_model == "decision_tree":
        classifier = BinaryDecisionTreeClassifier
    elif ml_model == "logistic":
        classifier = BinaryLogisticRegression
    else:
        classifier = BinaryMLPClassifier

    # Compute classification score using the selected classifier
    classification_score = classifier.compute(
        test_data=test_data,
        train_data=train_data,
        target=target_column,
        metadata=metadata
    )

    return classification_score


def compute_ml_regression(test_data, train_data, target_column, metadata, ml_model):
    """
    Get the R2 regression score using a specified machine learning model.

    Parameters:
        test_data (pandas.DataFrame): Test dataset.
        train_data (pandas.DataFrame): Train dataset.
        target_column (str): Name of the target column to predict.
        metadata (dict): Additional metadata for the model.
        ml_model (str): Machine learning model identifier.

    Returns:
        regression_score: R2 regression score from the selected model.
    """
    # Map ml_model identifier to corresponding regressor
    if ml_model == "linear":
        regressor = LinearRegression
    else:
        regressor = MLPRegressor

    # Compute R2 regression score using the selected regressor
    regression_score = regressor.compute(
        test_data=test_data,
        train_data=train_data,
        target=target_column,
        metadata=metadata
    )

    return regression_score


# from sdmetrics.single_table import MulticlassDecisionTreeClassifier, MulticlassMLPClassifier
# MulticlassMLPClassifier.compute(
#     test_data=real_data,
#     train_data=synthetic_data,
#     target='categorical_column_name',
#     metadata=metadata
# )
