# Imputation Methods for Synthetic Data Generation

## Overview
This document covers various imputation methods for handling missing values in datasets, particularly focusing on generating high-quality synthetic data.

## Imputation Methods Overview

The following table summarizes each imputation method along with its explanation, supported data types, and suitability for different datasets:

| Method          | Explanation                                                                                     | Supported Data Types       | Suitable Datasets                                     |
|-----------------|-------------------------------------------------------------------------------------------------|----------------------------|-------------------------------------------------------|
| *HyperImpute     | Uses regression/classification with various models for iterative imputation.                    | Categorical & Continuous   | Complex datasets with non-linear variable relationships|
| Mean            | Replaces missing values with column mean.                                                       | Numerical                  | Datasets with relatively normal distribution          |
| Median          | Replaces missing values with column median.                                                     | Numerical                  | Skewed numerical datasets                            |
| Most-frequent   | Uses the most frequent value for imputation.                                                    | Categorical & Discrete Num.| Categorical or discrete numerical data               |
| *MissForest      | Based on Random Forests for iterative imputation.                                               | Mixed                      | Complex datasets with mixed data types               |
| ICE             | Regularized linear regression for iterative imputation.                                         | Categorical & Continuous   | Linear relationship datasets                         |
| *MICE            | Multiple imputations based on ICE.                                                              | Categorical & Continuous   | For statistical analysis with uncertainty in imputation|
| SoftImpute      | Low-rank matrix approximation approach.                                                         | Numerical                  | Large datasets with low-rank structure                |
| EM              | Iterative method using other variables to impute missing values.                                | Varied                     | Datasets where data is MAR                           |
| Sinkhorn        | Imputation using Optimal Transport theory.                                                      | Varied                     | Datasets with systematic missingness pattern          |
| GAIN            | GAN framework for imputation in complex data.                                                   | High-dimensional           | High-dimensional, complex datasets                   |
| MIRACLE         | Focuses on learning the mechanism behind missing data.                                          | Varied                     | Datasets where missingness is influenced by factors   |
| MIWAE           | Combines deep generative models with imputation.                                                | High-dimensional, Complex  | High-dimensional datasets with non-linear relationships|

## Suitable Approaches for High-Quality Synthetic Data

When applying imputation methods to real data for synthetic data generation, certain approaches are more likely to yield high-quality synthetic datasets:

1. **HyperImpute**: Versatile for complex datasets with various models, capturing intricate patterns effectively.
2. **MissForest**: Powerful in handling complex, non-linear relationships and mixed-type data.
3. **MICE**: Handles uncertainty in imputation by creating multiple imputations, beneficial for robust synthetic data.
4. **SoftImpute**: Ideal for large datasets, uses matrix completion to preserve underlying relationships.
5. **GAIN (Generative Adversarial Nets)**: Suited for generating synthetic data, especially in high-dimensional spaces.
6. **MIWAE**: Utilizes deep generative models, capturing complex relationships in high-dimensional data.
7. **MIRACLE**: Focuses on the mechanisms of missing data, aligning imputations closely with real data's distribution.

Each method's effectiveness can vary based on the dataset characteristics. The chosen method should align with these characteristics to ensure accurate reflection in the synthetic data. Combining multiple methods or refining the process iteratively can further enhance quality.

# Key Insights from "HyperImpute: Generalized Iterative Imputation with Automatic Model Selection"

## Overview
This document summarizes the key insights from the paper, which presents the HyperImpute method - a comprehensive approach for imputing missing values in datasets.

## Imputation Problem and Existing Approaches
- **Challenge of Missing Data**: The paper acknowledges the ubiquity of missing data in real-life datasets.
- **Conventional Methods**: Discusses conventional iterative imputation and deep generative models. However, each has limitations: iterative imputation requires specific model specification, and generative models are often difficult to optimize and rely on strong data assumptions.

## HyperImpute Framework
- **Combining Advantages**: HyperImpute aims to combine the advantages of iterative imputation and deep generative models.
- **Framework Goals**: Designed to be flexible, easily optimized, and not reliant on the assumption that missingness is completely random.

## Focus on MCAR and MAR Data
- **Data Types**: Focuses on Missing Completely At Random (MCAR) and Missing At Random (MAR) data, common in practice.

## Definitions of MCAR and MAR with Examples

### Missing Completely At Random (MCAR)
- **Definition**: The missingness of data is completely independent of any observed or unobserved data.
- **Example**: In a medical dataset, blood pressure values missing due to random equipment failures, irrespective of patients' age, weight, or blood pressure readings, represent MCAR.

### Missing At Random (MAR)
- **Definition**: The missingness is systematically related only to the observed data, not the missing data.
- **Example**: In the same medical dataset, if older patients are more likely to miss their follow-up appointments, leading to missing blood pressure measurements, this missingness (related to the observed data - age) is MAR.

## Generalized Iterative Imputation and Automatic Model Selection
- **Iterative Imputation**: Employs generalized iterative imputation, allowing for different models for each feature.
- **Automatic Model Selection**: Integrates automatic model selection based on feature characteristics.

## HyperImpute Algorithm and Practical Implementation
- **Algorithm Details**: The paper details the HyperImpute algorithm, including baseline imputation and iterative refinement.
- **Practical Implementation**: Includes a range of learners, optimizers, and imputers, designed for integration into existing pipelines.

## Empirical Evaluation
- **Effectiveness**: Demonstrates the effectiveness of HyperImpute over various benchmarks.
- **Performance Analysis**: Investigates the sources of HyperImpute's gains, model selections, and its convergence.

