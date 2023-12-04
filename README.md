# benchmarking-synthetic-data-generators

| Argument | Type | Default Value | Description | Possible Values |
|----------|------|---------------|-------------|-----------------|
| --library, --l | str | sdv | Enter library | {sdv, gretel, synthcity} |
| --modality, --m | str | tabular | Enter dataset modality | {tabular, sequential, text} |
| --synthesizer, --s | str | ctgan | Enter synthesizer name | {ctgan, tvae, gaussian_copula, par, dgan, actgan, goggle} |
| --data | str | adult | Enter dataset name | {adult, census, child, covtype, credit, insurance, intrusion, health_insurance, drugs, loan, nasdaq, taxi, pums} |
| --imputer, --i | str | hyperimpute | Enter hyperimputer plugin name | {'simple', 'mice', 'missforest', 'hyperimpute'} |
| --optimizer_trials, --trials | int | 25 | | |
| --num_epochs, --e | int | 0 | Default epoch is set as 0 as statistical models do not need epochs | |
| --data_folder, --d | str | data | | |
| --output_folder, --o | str | outputs | | |
| --train_test_data, --tt | bool | False | Whether to return train and test data | |
| --get_quality_report, --qr | bool | False | Whether to generate SDV quality report | |
| --get_diagnostic_report, --dr | bool | False | Whether to generate SDV diagnostic report | |
| --run_optimizer, --ro | bool | False | Whether to run hyperparameter optimizer | |
| --run_hyperimpute, --ri | bool | False | Whether to run hyperimpute | |
| --run_model_training, --rt | bool | False | Whether to train a model | |
| --use_gpu, --cuda | bool | False | Whether to use GPU device(s) | |


```bash

python3 run_model.py --m tabular --l sdv --s gaussian_copula --data loan --o outputs --rt
python3 run_model.py --m tabular --l sdv --s ctgan --data loan --o outputs --ri --rt --tt
python3 run_model.py --m tabular --l gretel --s actgan --data loan --o outputs --ri --rt --tt

health_insurance
```


## Key Packages Used

- **ydata-profiling**: Provides comprehensive data profiling for understanding and assessing data quality.
- **hyperimpute**: Advanced tool for imputing missing data in datasets using various techniques.
- **optuna**: A hyperparameter optimization framework designed for machine learning models.
- **synthcity**: Generates realistic synthetic datasets while preserving the statistical properties of original data.
- **sdv**: A library for modeling and generating synthetic datasets.
- **rdt**: Transforms data for compatibility with synthetic data generation and modeling.
- **metrics**: Offers metrics for evaluating the quality of synthetic data models.
- **ctgan**: Implements CTGAN, a model for generating synthetic tabular data.
- **gretel-synthetics**: Creates synthetic data that maintains the privacy and utility of original datasets.
- **ydata-synthetics**: Framework focused on generating statistically consistent synthetic data.
