from sklearn.linear_model import Ridge
import numpy as np
import math

def split_dataset(training_data):
    length = np.shape(training_data)[0]
    split_set = np.split(training_data,10,axis=0)
    return split_set

########################
### Main for task 1a ###
########################

# Regularisation parameter
lambda_reg = [0.1,1,10,100,200]
lambda_score = np.zeros(len(lambda_reg))

## Load Data from csv files
train_data = np.genfromtxt('Project_1/train.csv', delimiter=',')[1:,:]
data = split_dataset(train_data)

## Validate model for every alpha
j = 0
for alpha in lambda_reg:
    ## Get validation 10 times
    ridge_regressor = Ridge(alpha)
    score = np.zeros((1,len(data)))
    i = 0
    for cv_set in data:
        ## Train with remaining data
        aux_data = data[:]
        aux_data.pop(i)
        train_tuple = tuple(aux_data)
        train_set = np.vstack(train_tuple)
        ridge_regressor.fit(train_set[:,1:],train_set[:,0:1])

        ## Evaluation
        cv_predictions = ridge_regressor.predict(cv_set[:,1:])
        score[0,i] = math.sqrt(np.mean((cv_set[:,0:1] - cv_predictions)**2))
        i += 1
    
    # Compute average score
    avg_score = np.mean(score)

    # Append to lambda_score
    lambda_score[j] = avg_score
    j += 1

## Print test outputs to csv files
np.savetxt('Project_1/scores.csv', lambda_score, fmt=['%.15f'], delimiter='\n', comments='')
print("Task completed")





