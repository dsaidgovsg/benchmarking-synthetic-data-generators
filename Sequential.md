
# Synthetic Time Series Data Generation Parameters

For generating synthetic time series data using models like PAR (Probabilistic Auto-Regressive) or DoppleGANger, it's crucial to set the right parameters for the entity, temporal column, static columns, and dynamic columns. Given the taxi trip dataset, here's how these parameters can be configured:

## Parameters Configuration

### 1. Entity Column
The entity column is the unique identifier for each entity in your dataset.
- **Entity Column**: `taxi_id`

### 2. Temporal Column
This column indicates the time aspect of your data.
- **Temporal Column**: `trip_start_timestamp`

### 3. Static Columns
These columns have values that remain constant for each entity over time.
- **Static Columns**: 
  - `company` 
  > Assuming a taxi remains with the same company throughout the dataset.

### 4. Dynamic Columns
These columns contain values that change over time for each entity.
- **Dynamic Columns**: 
  - `trip_end_timestamp`
  - `trip_seconds`
  - `trip_miles`
  - `pickup_census_tract`
  - `dropoff_census_tract`
  - `pickup_community_area`
  - `dropoff_community_area`
  - `fare`
  - `tips`
  - `tolls`
  - `extras`
  - `trip_total`
  - `payment_type`
  - `pickup_latitude`
  - `pickup_longitude`
  - `dropoff_latitude`
  - `dropoff_longitude`

## Implementation
When inputting these parameters into the PAR or DoppleGANger model, ensure that your dataset structure aligns correctly with the defined parameters. The choice of static and dynamic attributes may vary based on specific contexts or assumptions about the dataset.


# Process Groups Function

## Overview
The `process_groups` function is designed to process groups within a Pandas DataFrame based on a specified operation. It allows for flexible data manipulation, especially useful in scenarios where groups of data need to be standardized in size by either padding with additional data or truncating excess data.

## Function Definition
```python
process_groups(df, required_group_size, group_by_column, operation='padding_and_truncate')
```

## Parameters

| Parameter           | Type      | Description                                                                                                                   |
|---------------------|-----------|-------------------------------------------------------------------------------------------------------------------------------|
| `df`                | DataFrame | The DataFrame to be processed. This is the input DataFrame that contains the data to be grouped and modified.                 |
| `required_group_size` | int        | The target size for each group. This parameter defines the desired number of entries in each group after processing.          |
| `group_by_column`   | str       | The name of the column in the DataFrame to group by. This parameter determines how the DataFrame will be split into groups.  |
| `operation`         | str       | The operation to be applied to the groups. Options: `'padding'`, `'truncate'`, `'padding_and_truncate'`, `'drop_and_truncate'`. |

## Returns
- **pd.DataFrame**: A new DataFrame with groups processed according to the specified operation.

## Example Usage
```python
data = {'Category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'C'],
        'Value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
df = pd.DataFrame(data)
processed_df = process_groups(df, 3, 'Category', operation='drop_and_truncate')
print(processed_df)
```
```

This README is formatted for GitHub and includes a table detailing each parameter of the `process_groups` function for clarity and ease of understanding. Users can refer to this documentation to better understand how to use the function in their projects.
