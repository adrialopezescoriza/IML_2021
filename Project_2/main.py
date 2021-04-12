import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import datasets, svm
from sklearn.kernel_approximation import Nystroem
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_multilabel_classification

# Function for imputation
def imputation(data):
    imp = IterativeImputer(max_iter=3, random_state=0)
    return imp.fit_transform(data)

# Function for dimensionality reduction
def feature_extraction():
    pass

# Load dateset
X = np.genfromtxt('Project_2/train_features.csv', delimiter=',')[1:,1:]
y1 = np.genfromtxt('Project_2/train_labels.csv', delimiter=',')[1:,1:11].astype(int)
y2 = np.genfromtxt('Project_2/train_labels.csv', delimiter=',')[1:,12].astype(int)
y3 = np.genfromtxt('Project_2/train_labels.csv', delimiter=',')[1:,13:].astype(int)

# Data Imputation
np.nan_to_num(X,copy=False,nan=0.0)
#X = imputation(X)

# Time series concatenation
patients_size = int(X.shape[0]/12)
X_stacked = np.zeros((patients_size,X.shape[1]*12))
for i in range(patients_size):
    X_stacked[i,:] = X[i*12:(i+1)*12,:].reshape(-1)
print(X_stacked.shape)

########################
######## Task 1 ########
########################
clf_1 = MultiOutputClassifier(KNeighborsClassifier(),n_jobs=10)
feature_map_nystroem = Nystroem(gamma=.2, random_state=1,n_components=6)
data_transformed = feature_map_nystroem.fit_transform(X_stacked)
clf_1.fit(data_transformed, y1)
print(clf_1.score(data_transformed, y1))

########################
######## Task 2 ########
########################
clf_2 = svm.SVC()
clf_2.fit(data_transformed, y2)
print(clf_2.score(data_transformed, y2))

########################
######## Task 3 ########
########################
