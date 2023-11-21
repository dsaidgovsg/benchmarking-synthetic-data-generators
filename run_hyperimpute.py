"Apply imputation on a DataFrame"

import time
import json
from pandas import DataFrame
from typing import Any, Dict

from hyperimpute.plugins.imputers import Imputers

# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import SimpleImputer, IterativeImputer
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def is_mcar(dataframe):
    # TODO (futuristic)
    """
    Heuristic to check if the dataset is likely MCAR (Missing Completely At Random).
    This is a simplified check and might not be accurate for complex datasets.

    Args:
    dataframe (pd.DataFrame): The dataset with missing values.

    Returns:
    bool: True if the dataset is likely MCAR, False otherwise.
    """
    # Implement logic to check for MCAR. This can be a statistical test or heuristic.
    # For simplicity, let's assume a dataset is MCAR if missingness is low and does not correlate with other variables.
    # This is a very simplified assumption.
    missing_percent = dataframe.isnull().mean()

    return missing_percent.max() < 5 and dataframe.corrwith(missing_percent).abs().max() < 0.1


def apply_imputation(dataframe: DataFrame,
                     method: str,
                     dataset_name: str,
                     output_path: str) -> DataFrame:
    """
    Apply the specified imputation method to the given DataFrame.

    Parameters:
    - dataframe (DataFrame): The DataFrame on which to apply the imputation.
    - method (str): The imputation method to use. Valid options are 'simple', 'hyperimpute', 'mice', 'ice', 'missforest'.
    - dataset_name (str): The name of the dataset, used for saving the output files.
    - output_path (str): The path where the output files (imputed data and imputer details) will be saved.

    Returns:
    - DataFrame: The DataFrame after imputation.

    Raises:
    - ValueError: If an invalid imputation method is specified.
    """

    # Check if the dataframe has any missing values
    if not dataframe.isna().any().any():
        print("No missing values in the dataset. Returning the original dataframe.")
        return dataframe

    begin_impute_time = time.time()

    try:
        if method == "simple":
            for column in dataframe.columns:
                if dataframe[column].dtype.kind in 'bifc':  # Check for numeric types
                    print("Numerical: ", column)
                    strategy = 'mean' if - \
                        0.5 < dataframe[column].skew() < 0.5 else 'median'
                    imputer = Imputers().get(strategy)
                    dataframe[column] = imputer.fit_transform(
                        dataframe[[column]])
                else:
                    print("Categorical: ", column)
                    imputer = Imputers().get("most_frequent")
                    dataframe[column] = imputer.fit_transform(
                        dataframe[[column]])

        elif method == "hyperimpute":
            imputer = Imputers().get(
                "hyperimpute",
                # optimizer: str. The optimizer to use: simple, hyperband, bayesian
                optimizer="hyperband",
                # classifier_seed: list. Model search pool for categorical columns.
                classifier_seed=["logistic_regression",
                                 "catboost", "xgboost", "random_forest"],
                # regression_seed: list. Model search pool for continuous columns.
                regression_seed=["linear_regression", "catboost_regressor",
                                 "xgboost_regressor", "random_forest_regressor"],
                # class_threshold: int. how many max unique items must be in the column to be is associated with categorical
                class_threshold=5,
                # imputation_order: int. 0 - ascending, 1 - descending, 2 - random
                imputation_order=2,
                # n_inner_iter: int. number of imputation iterations
                n_inner_iter=10,
                # select_model_by_column: bool. If true, select a different model for each column.
                # Else, it reuses the model chosen for the first column.
                select_model_by_column=True,
                # select_model_by_iteration: bool. If true, selects new models for each iteration.
                # Else, it reuses the models chosen in the first iteration.
                select_model_by_iteration=True,
                # select_lazy: bool. If false, starts the optimizer on every column unless other restrictions apply.
                # Else, if for the current iteration there is a trend(at least to columns of the same type got the same model from the optimizer),
                # it reuses the same model class for all the columns without starting the optimizer.
                select_lazy=True,
                # select_patience: int. How many iterations without objective function improvement to wait.
                select_patience=5
            )
            dataframe[:] = imputer.fit_transform(dataframe)

        elif method in ["ice", "missforest"]:  # "mice"
            imputer = Imputers().get(method)
            dataframe[:] = imputer.fit_transform(dataframe)

        else:
            raise ValueError("Not a valid imputation method.")

    except Exception as e:
        print(f"Error using {method} imputation method: {e}")

    end_impute_time = time.time()
    imputer_info: Dict[str, Any] = {
        "lib": "hyperimpute==0.1.17",
        "imputation_method": method,
        "impute_time_sec": end_impute_time - begin_impute_time
    }

    with open(f"{output_path}{dataset_name}_imputer.json", "w", encoding="utf-8") as json_file:
        json.dump(imputer_info, json_file)
    dataframe.to_csv(
        f"{output_path}{dataset_name}_imputed_data.csv", index=False)

    return dataframe

# NOTES: will remove later
# 1. methods for mixed type
# --> hyperimpute, ice, missforest
# 2. methods for numeric type:
# --> median, mean, sklearn_ice, sklearn_missforest, gain, sinkhorn, miwae, mice
# 3. what does this one do?
# --> 'nop'.
# TypeError: ufunc 'isnan' not supported for the input types, and the inputs could
# 4. not be safely coerced to any supported types according to the casting rule ''safe''
# --> 'miracle', 'EM', 'softimpute'

# more
# can use the benchmarking functioanlity of synthcity
