# BetterData Synthetic Data Report Metrics Overview

This document provides an overview of the key metrics used in the synthetic data report, including their purpose, preferable values (maximum or minimum), and typical ranges.

## Fidelity Metrics
- **Purpose**: Assess the closeness of synthetic data to the original data in terms of statistical properties and patterns.
- **Specific Metrics**: The report might use statistical measures like correlation coefficients, distribution comparisons (e.g., Kolmogorov-Smirnov test), or mean squared error between real and synthetic data distributions. Unfortunately, the exact metrics used for fidelity in your report are not specified, but they are generally intended to ensure that synthetic data maintains the structural and statistical integrity of the original data.
- **Preferable Value**: Higher fidelity is better.
- **Typical Range**: Varies, may be qualitative or quantitative.

## Distance-Based Metrics
### DCR (Distance to Closest Record)
- **Purpose**: Measures proximity of synthetic records to the closest real records.
- **Preferable Value**: Lower is better.
- **Typical Range**: Usually from 0 upwards.

### NNDR (Nearest Neighbor Distance Ratio)
- **Purpose**: Compares distances of nearest neighbors between synthetic and real datasets.
- **Preferable Value**: Lower is better.
- **Typical Range**: Usually from 0 upwards.

## Privacy Metrics
### Inference Attack Risk
- **Purpose**: Assesses vulnerability of dataset attributes to being inferred using other attributes.
- **Preferable Value**: Lower is better.
- **Typical Range**: Expressed as a probability or percentage.

### Linkability Attack Risk
- **Purpose**: Quantifies the risk of linking synthetic records to real individuals.
- **Preferable Value**: Lower is better.
- **Typical Range**: Expressed as a probability or percentage.

### Singling Out Multivariate Attack Risk
- **Purpose**: Measures the probability of isolating records that can identify individuals.
- **Preferable Value**: Lower is better.
- **Typical Range**: Expressed as a probability or percentage.

## ML Classification Metrics
- **Purpose**: Evaluates the performance of ML models trained on synthetic versus real data.
- **Preferable Value**: Higher performance of synthetic data-trained model is better.
- **Typical Range**: Performance comparison, often in percentage.

## Privacy Rank and Score
### Privacy Score
- **Purpose**: Composite score derived from various privacy metrics.
- **Preferable Value**: Higher is better.
- **Typical Range**: Typically on a scale up to 100.

This README aims to provide a clear understanding of the various metrics used to evaluate the quality and privacy of synthetic data, as well as its utility in machine learning applications.
