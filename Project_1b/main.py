import numpy as np
from sklearn.linear_model import LinearRegression

def feature_transformation(x):
    # Input in format  [x1,...,x5]
    # Output in format [x1,...,x21]

    # Linear
    lin_x = x
    # Quadratic
    quad_x = x**2
    # Exponential 
    exp_x = np.exp(x)
    # Coseine
    cos_x = np.cos(x)
    # Constant
    const_x = np.array([1])
    x_transform = np.hstack((lin_x,quad_x,exp_x,cos_x,const_x))

    return x_transform

########################
### Main for task 1b ###
########################

## Load Data from csv files
train_data = np.genfromtxt('Project_1b/train.csv', delimiter=',')[1:,1:]

## Transform training data to feature space
for i in range(np.shape(train_data)[0]):
    phi_x = feature_transformation(train_data[i,1:])
    phi_xy = np.hstack((train_data[i,0:1],phi_x))
    if i==0:
        phi_train = phi_xy
    else:
        phi_train = np.vstack((phi_train,phi_xy))

## Linear regression
X, y = phi_train[:,1:], phi_train[:,0:1]
reg = LinearRegression().fit(X, y)
print("Regression score = ", reg.score(X,y))

# Coefficients
w = reg.coef_

## Save parameters to csv file
np.savetxt('Project_1b/results.csv', w, fmt='%.18f', delimiter='\n', comments='')
print("Task completed")