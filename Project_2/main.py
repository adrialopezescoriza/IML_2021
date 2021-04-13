import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import datasets, svm
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_ridge import KernelRidge
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, r2_score

# Function for imputation
def imputation(data):
    imp = IterativeImputer(max_iter=3, random_state=0)
    return imp.fit_transform(data)

# Function for dimensionality reduction
def feature_extraction(X):
    feature_map_nystroem = Nystroem(gamma=.2, random_state=1,n_components=1)
    return feature_map_nystroem.fit_transform(X)

def time_series_conc(X):
    patients_size = int(X.shape[0]/12)
    X_stacked = np.zeros((patients_size,X.shape[1]*12))
    for i in range(patients_size):
        X_stacked[i,:] = X[i*12:(i+1)*12,:].reshape(-1)
    return X_stacked

# Load dateset
X = np.genfromtxt('Project_2/train_features.csv', delimiter=',')[1:,1:]
y1 = np.genfromtxt('Project_2/train_labels.csv', delimiter=',')[1:,1:12].astype(int)
y2 = np.genfromtxt('Project_2/train_labels.csv', delimiter=',')[1:,12].astype(int)
y3 = np.genfromtxt('Project_2/train_labels.csv', delimiter=',')[1:,13:].astype(int)

#######################
#### Preprocessing ####
#######################

# Data Imputation
#np.nan_to_num(X,copy=False,nan=0.0)
X = imputation(X)

# Time series concatenation
X_stacked = time_series_conc(X)

# Dimensionality reduction
data_transformed = feature_extraction(X_stacked)

########################
######## Task 1 ########
########################
clf_1 = MultiOutputClassifier(KNeighborsClassifier(),n_jobs=10)
clf_1.fit(data_transformed, y1)
scr1 = roc_auc_score(y1, 1/(1+np.exp(clf_1.predict(data_transformed))))
print("Score Task 1:",scr1)

########################
######## Task 2 ########
########################
clf_2 = svm.SVC()
clf_2.fit(data_transformed, y2)
#scr2 = roc_auc_score(y2, 1/(1+np.exp(clf_2.predict(data_transformed))), multi_class='ovo')
scr2 = 0.5
print("Score Task 2:",scr2)

########################
######## Task 3 ########
########################
clf_3 = KernelRidge(alpha=1.0)
clf_3.fit(data_transformed, y3)
scr3 = r2_score(y3, clf_3.predict(data_transformed))
print("Score Task 3:",scr3)

avg_scr = (scr1 + scr2 + scr3) / 3
print("Average score: ", avg_scr)


########################
## Results Generation ##
########################
X_test = np.genfromtxt('Project_2/test_features.csv', delimiter=',')[1:,1:]
patient_id = np.atleast_2d(np.genfromtxt('Project_2/test_features.csv', delimiter=',')[1::12,0:1])

# Preprocessing
X_test = imputation(X_test)
X_test = time_series_conc(X_test)
X_test = feature_extraction(X_test)

# Predictions
y_pred1 = 1/(1+np.exp(-clf_1.predict(X_test)))
y_pred2 = np.atleast_2d(1/(1+np.exp(-clf_2.predict(X_test)))).T
y_pred3 = clf_3.predict(X_test)
y_pred = np.hstack((patient_id,y_pred1,y_pred2,y_pred3))

# Save results
fmt = ['%.0f','%.3f','%.3f','%.3f','%.3f','%.3f','%.3f','%.3f','%.3f','%.3f','%.3f','%.3f','%.3f','%.3f','%.3f','%.3f']
header = 'pid,LABEL_BaseExcess,LABEL_Fibrinogen,LABEL_AST,LABEL_Alkalinephos,LABEL_Bilirubin_total,LABEL_Lactate,LABEL_TroponinI,LABEL_SaO2,LABEL_Bilirubin_direct,LABEL_EtCO2,LABEL_Sepsis,LABEL_RRate,LABEL_ABPm,LABEL_SpO2,LABEL_Heartrate'
np.savetxt('Project_2/results.csv', y_pred, fmt=fmt, delimiter=',', comments='', header = header)
print("Task completed")
