from sklearn.linear_model import Ridge
import numpy as np

########################
### Main for task 1a ###
########################

# Regularisation parameter
lambda_reg = [0.1,1,10,100,200]

## Load Data from csv files
train_data = np.genfromtxt('train.csv', delimiter=',')[1:,:]


def split_dataset(training_data):
    length = np.shape(training_data)[0]
    split_set = np.split(training_data,10,axis=0)


