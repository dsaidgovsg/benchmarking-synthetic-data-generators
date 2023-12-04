## Rough Dumps/Notes

#### List of resources for metrics 

- [ydata](https://6266199.fs1.hubspotusercontent-na1.net/hubfs/6266199/Whitepapers/YData_Whitepaper_SyntheticDataMetrics.pdf)
- [ydata-profiling](https://ydata-profiling.ydata.ai/docs/master/pages/getting_started/overview.html)
- [Betterdata](https://betterdata.gitbook.io/data-metrics-guide/)
- [Gretel](https://docs.gretel.ai/reference/evaluate)
- [SDMetrics](https://docs.sdv.dev/sdmetrics/)
- [BulianAI](https://docs.bulian.ai/bulianai-overview/api-docs/getting-started-with-bulian-ai/synthetic-data-quality-report)

- [List of metrics from Datomize](https://www.datomize.com/generative-models-benchmark/)

The Overall Score depends upon the task at hand and is user defined, for example, if the user asks for the ML Efficacy metric, then only the Efficacy Score is taken into account to calculate the Overall Score.  

#### dython library 
**random_forest_feature_importance¶**
- random_forest_feature_importance(forest, features, precision=4)

- Given a trained sklearn.ensemble.RandomForestClassifier, plot the different features based on their importance according to the classifier, from the most important to the least.

**liklihood metrics** 
https://docs.sdv.dev/sdmetrics/metrics/metrics-in-beta/data-likelihood 
- Bayesian Network Metrics 
- Gaussian Mixture Metrics

**Discriminator**
Metrics that train a Machine Learning Classifier to distinguish between the real and the synthetic data. 
- LogisticDetection
- SVCDetection

**Outliers Coverage**
**Boundaries/Range Coverage**
**Cardinality Coverage/ CategoryCoverage**


**Privacy Metrics**
#### CategoricalCAP
- CategoricalCAP (Correct Attribution Probability) measures the risk of disclosing sensitive information through an inference attack. 
- https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/categoricalcap


**Mutual Information**


The difference between Pearson and Spearman is in whether we assume the real data has a linear trend. Use a coefficient based on what you expect the real data to have and what you hope the synthetic data will be able to effectively capture.
Note that the Spearman coefficient may be slower to compute.


---------------------
KSComplement (Kolmogorov-Smirnov statistic) -- Numeric and datetime
https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/kscomplement

TVComplement (Total Variation Distance (TVD) -- Categorical and Boolean
https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/tvcomplement

** CSTest **
This test normalizes the real and synthetic data in order to compute the category frequencies. Then, it applies the Chi-squared test [1] to test the null hypothesis that the synthetic data comes from the same distribution as the real data.
The test returns the p-value [2], where a smaller p-value indicates that the synthetic data is significantly different from the real data, rejecting the null hypothesis and leading to a worse overall score.

https://docs.sdv.dev/sdmetrics/metrics/metrics-in-beta/cstest 


## Sequential 
**Detection**
https://docs.sdv.dev/sdmetrics/metrics/metrics-in-beta/detection-sequential


**Jensen Shannon Distance**
# https://towardsdatascience.com/how-to-understand-and-use-jensen-shannon-divergence-b10e11b03fd6
