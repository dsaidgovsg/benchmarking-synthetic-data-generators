import pandas as pd
from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def is_mcar(dataframe):
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


def auto_impute(dataframe):
    """
    Automatically applies suitable imputation methods based on the dataset.

    Args:
    dataframe (pd.DataFrame): The dataset with missing values.

    Returns:
    pd.DataFrame: Dataset with imputed values.
    """

    # Check if the dataset is likely MCAR

    # Initialize a copy of the dataframe to avoid modifying the original one
    imputed_dataframe = dataframe.copy()

    # Iterate over each column in the DataFrame
    for column in imputed_dataframe.columns:
        # Select appropriate imputer based on the data type of the column
        if imputed_dataframe[column].dtype.kind in 'bifc':  # Numeric columns
            """
            is used to determine if a column contains numeric data (either integer, boolean, or floating-point).
            If this condition is true, it implies that the column is numeric and suitable for numerical imputation methods like IterativeImputer.
            If the condition is false, it suggests that the column is non-numeric (likely categorical or string data), 
            and a different imputation method (like using the 'most_frequent' strategy with SimpleImputer) would be more appropriate.
            """
            # Use IterativeImputer for numeric columns
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(), max_iter=10, random_state=0)
        else:
            # Use SimpleImputer with 'most_frequent' for non-numeric columns
            imputer = SimpleImputer(strategy='most_frequent')

        # Apply the imputation
        imputed_data = imputer.fit_transform(imputed_dataframe[[column]])
        imputed_dataframe[column] = imputed_data

    # print("------>", is_mcar(dataframe))
    # if is_mcar(dataframe):
    #     # Simple imputation for MCAR data
    #     imputer = SimpleImputer(strategy='mean')
    # else:
    #     # The dataset is likely MAR or more complex
    #     categorical = any(dtype.name == 'category' or dtype.name == 'object' for dtype in dataframe.dtypes)
    #     if categorical:
    #         # MICE or similar for mixed data types
    #         imputer = IterativeImputer(estimator=RandomForestRegressor() if dataframe.dtypes[0] != 'object' else RandomForestClassifier(),
    #                                    max_iter=10, random_state=0)
    #     else:
    #         # MissForest for numerical datasets with complex relationships
    #         imputer = IterativeImputer(estimator=RandomForestRegressor(), max_iter=10, random_state=0)

    # Apply the imputation
    # imputed_data = imputer.fit_transform(dataframe)

    print("SUCCESS")
    print(imputed_dataframe.head)
    print(
        f"after imputation: total missing values in the real DataFrame: {imputed_dataframe.isna().sum().sum()}")
    # return pd.DataFrame(imputed_data, columns=dataframe.columns)


# Usage example
df = pd.read_csv(
    '/Users/anshusingh/DPPCC/whitespace/benchmarking-synthetic-data-generators/sample_datasets/accidential_drug_deaths.csv')
print(
    f"before imputation: total missing values in the real DataFrame: {df.isna().sum().sum()}")
print(df.head)
print("*"*20)
imputed_df = auto_impute(df)
