from ydata_synthetic.synthesizers.regular import RegularSynthesizer 
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters 

import pandas as pd

# Refer exmaples here: https://github.com/ydataai/ydata-synthetic/tree/dev/examples/regular/models
# https://github.com/ydataai/ydata-synthetic/blob/dev/integrations/expectations_to_SyntheticData/1-Expectations%20%26%20Profiling.ipynb
#-----------------------------------------
#  CTGAN
#-----------------------------------------
# ID,Age,Experience,Income,ZIP Code,Family,CCAvg,Education,Mortgage,Personal Loan,Securities Account,CD Account,Online,CreditCard

# data = pd.read_csv("/Users/anshusingh/DPPCC/whitespace/benchmarking-synthetic-data-generators/sample_datasets/loan.csv")
# num_cols = ["ID", "Age", "Experience", "Income", "ZIP Code", "Family", "CCAvg", "Mortgage", "Personal Loan", "Securities Account", "CD Account", "Online", "CreditCard"]
# cat_cols = ["Education"]

print("@"*10)
dp = "sample_datasets/adult.csv"
data = pd.read_csv(dp)
num_cols = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
cat_cols = ['workclass','education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'native-country', "label"]


# Defining the training parameters
batch_size = 500
epochs = 2 #500+1
learning_rate = 2e-4
beta_1 = 0.5
beta_2 = 0.9

ctgan_args = ModelParameters(batch_size=500,
                             lr=learning_rate,
                             betas=(beta_1, beta_2))

# ctgan_args = ModelParameters()

train_args = TrainParameters(epochs=epochs)
synth = RegularSynthesizer(modelname='ctgan', model_parameters=ctgan_args) #ctgan 
# cgan
print("traning starts")
synth.fit(data=data, train_arguments=train_args, num_cols=num_cols, cat_cols=cat_cols)
print("traning ends")
# synth.save("model.pkl")


# synth = RegularSynthesizer.load('model.pkl')
synth_data = synth.sample(10)
print(synth_data)

print("SUCCESS train and generation!!! :D")
# breakpoint()

#-------------------
# TimeGAN
# conflciting python version possibly
#-------------------


# from os import path
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# from ydata_synthetic.synthesizers import ModelParameters
# from ydata_synthetic.preprocessing.timeseries import processed_stock
# from ydata_synthetic.synthesizers.timeseries import TimeGAN

# breakpoint()
# #Specific to TimeGANs
# seq_len=24
# n_seq = 6
# hidden_dim=24
# gamma=1

# noise_dim = 32
# dim = 128
# batch_size = 128

# log_step = 100
# learning_rate = 5e-4

# gan_args = ModelParameters(batch_size=batch_size,
#                            lr=learning_rate,
#                            noise_dim=noise_dim,
#                            layers_dim=dim)

# stock_data = processed_stock(path='g.csv', seq_len=seq_len)
# print(len(stock_data),stock_data[0].shape)

# synth = TimeGAN(model_parameters=gan_args, hidden_dim=24, seq_len=seq_len, n_seq=n_seq, gamma=1)
# synth.train(stock_data, train_steps=50000)
# synth.save('synthesizer_stock.pkl')

# synth_data = synth.sample(len(stock_data))
# print(synth_data.shape)

# breakpoint()

#-----------------------------------------
#  CramerGAN Example
#-----------------------------------------
#Install ydata-synthetic lib
# pip install ydata-synthetic
# import sklearn.cluster as cluster
# import numpy as np
# import pandas as pd

# from ydata_synthetic.utils.cache import cache_file
# from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
# from ydata_synthetic.synthesizers.regular import RegularSynthesizer

# #Read the original data and have it preprocessed
# # data_path = cache_file('creditcard.csv', 'https://datahub.io/machine-learning/creditcard/r/creditcard.csv')
# data_path = "/Users/anshusingh/DPPCC/whitespace/benchmarking-synthetic-data-generators/data/tabular/credit.csv"

# #cache_file('creditcard.csv', 'https://datahub.io/machine-learning/creditcard/r/creditcard.csv')
# data = pd.read_csv(data_path, index_col=[0])

# #Data processing and analysis
# num_cols = list(data.columns[ data.columns != 'label' ])
# cat_cols = ['label']

# print('Dataset columns: {}'.format(num_cols))
# sorted_cols = ['V14', 'V0', 'V4', 'V10', 'V17', 'V12', 'V26', 'Amount', 'V21', 'V8', 'V11', 'V7', 'V19', 'V3', 'V22', 'V6', 'V20', 'V27', 'V16', 'V13', 'V25', 'V24', 'V18', 'V2', 'V1', 'V5', 'V15', 'V9', 'V23', 'label']
# processed_data = data[ sorted_cols ].copy()

# #For the purpose of this example we will only synthesize the minority class
# train_data = processed_data.loc[processed_data['label'] == 1].copy()

# #Create a new class column using KMeans - This will mainly be useful if we want to leverage conditional GAN
# print("Dataset info: Number of records - {} Number of variables - {}".format(train_data.shape[0], train_data.shape[1]))
# algorithm = cluster.KMeans
# args, kwds = (), {'n_clusters':2, 'random_state':0}
# labels = algorithm(*args, **kwds).fit_predict(train_data[ num_cols ])

# print( pd.DataFrame( [ [np.sum(labels==i)] for i in np.unique(labels) ], columns=['count'], index=np.unique(labels) ) )

# fraud_w_classes = train_data.copy()
# fraud_w_classes['label'] = labels

# # GAN training
# #Define the GAN and training parameters
# noise_dim = 32
# dim = 128
# batch_size = 128

# log_step = 100
# epochs = 2#500+1
# learning_rate = 5e-4
# beta_1 = 0.5
# beta_2 = 0.9
# models_dir = '../cache'

# model_parameters = ModelParameters(batch_size=batch_size,
#                            lr=learning_rate,
#                            betas=(beta_1, beta_2),
#                            noise_dim=noise_dim,
#                            layers_dim=dim)

# train_args = TrainParameters(epochs=epochs,
#                              sample_interval=log_step)

# #Training the CRAMERGAN model
# synth = RegularSynthesizer(modelname='cramer', model_parameters=model_parameters)
# synth.fit(data=train_data, train_arguments = train_args, num_cols = num_cols, cat_cols = cat_cols)

# #Saving the synthesizer to later generate new events
# synth.save(path='creditcard_cramergan_model.pkl')

# #########################################################
# #    Loading and sampling from a trained synthesizer    #
# #########################################################
# synth = RegularSynthesizer.load(path='creditcard_cramergan_model.pkl')
# #Sampling the data
# #Note that the data returned it is not inverse processed.
# data_sample = synth.sample(100000)
