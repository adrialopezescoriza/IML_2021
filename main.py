import numpy as np
from sklearn.metrics import mean_squared_error

## Load Data from csv files
train_data = np.genfromtxt('train.csv', delimiter=',')
test_data  = np.genfromtxt('test.csv', delimiter=',')

## Check Training Set
train_predictions = np.mean(train_data[1:,2:], axis=1)
RMSE_train = mean_squared_error(train_data[1:,1],train_predictions)
print('Training Data RMSE: ', RMSE_train)

## Testing Set
test_predictions = np.mean(test_data[1:,1:], axis=1)
test_id = test_data[1:,0]
results = np.vstack((test_id,test_predictions)).T

## Print test outputs to csv files
header = np.array(["Id, y\n"])
#np.savetxt('results.csv', header, fmt="%s", delimiter=',')
np.savetxt('results.csv', results, fmt=['%.0f','%.1f'], delimiter=',', header="Id, y", comments='')