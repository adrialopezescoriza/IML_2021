import numpy as np
from pandas import DataFrame
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn import datasets, svm
from sklearn.svm import SVR
from sklearn.kernel_approximation import Nystroem
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.decomposition import PCA
import pandas as pd

# Function for imputation
def imputation(data,type):
    
    #imp = IterativeImputer(max_iter=3, random_state=0)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean',fill_value=0)
    X_imp = imp.fit_transform(data)
    return X_imp
    '''
    altered_data = np.empty([1,np.shape(data)[1]])
    for i in range(int(np.shape(data)[0]/12)): # 227940 entries = 18995 patients
        if type == 'mean':
            patient = np.nanmean(data[(12*i):(12*(i+1)),], axis=0)
            if i == 0:
                altered_data = patient
            else:
                altered_data = np.vstack([altered_data, patient])
        elif type == 'interpolate':
            patient = DataFrame(data[(12*i):(12*(i+1)),2:])
            data[(12*i):(12*(i+1)),2:] = patient.interpolate(axis=0, limit=11, limit_direction='both')
            altered_data = data

    df = DataFrame(altered_data) #including pid, Time, age
    coverage = df.count()/df.shape[0]*100
    df = df.drop(columns=coverage[coverage < 30].index) # remove features with data for less than 25% of patients

    return df.to_numpy()
    '''
    

# Function for dimensionality reduction
def feature_extraction(X,nc):
    feature_map = PCA(n_components=nc)
    return feature_map.fit_transform(X)

def time_series_conc(X):
    patients_size = int(X.shape[0]/12)
    X_stacked = np.zeros((patients_size,X.shape[1]))
    for i in range(patients_size):
        X_stacked[i,:] = np.mean(X[i*12:(i+1)*12,:],axis=0)
    return X_stacked

def arrange_probs(probs_array):
    y_probs = np.zeros((len(probs_array[0]),0))
    for array in probs_array:
        # Return probability of class 1
        y_probs = np.hstack((y_probs,array[:,1:2]))
    return y_probs

nc1 = 15
nc2 = 15
nc3 = 15
# Load dateset
X = np.genfromtxt('Project_2/train_features.csv', delimiter=',')[1:,1:]
y1 = np.genfromtxt('Project_2/train_labels.csv', delimiter=',')[1:,1:11].astype(int)
y2 = np.genfromtxt('Project_2/train_labels.csv', delimiter=',')[1:,11].astype(int)
y3 = np.genfromtxt('Project_2/train_labels.csv', delimiter=',')[1:,12:].astype(float)

#######################
#### Preprocessing ####
#######################

# Data Imputation
X = imputation(X,type="mean")

# Time series concatenation
X_stacked = time_series_conc(X)

########################
######## Task 1 ########
########################
data_transformed_1 = feature_extraction(X_stacked,nc=nc1)
clf_1 = MultiOutputClassifier(svm.SVC(probability=True, C=1),n_jobs=10)
clf_1.fit(data_transformed_1, y1)
probs_predicted = clf_1.predict_proba(data_transformed_1)
y1_probs = arrange_probs(probs_predicted)
scr1 = roc_auc_score(y1, y1_probs)
print("Score Task 1:",scr1)

########################
######## Task 2 ########
########################
# PCA analysis
# Symptoms for Sepsis (Heartrate: 31, Temp: 6, Lactate:5, Age:1, RRate:10, EtCO2:2
clf_2 = svm.SVC(probability=True, C=1)
#clf_2.fit(X_stacked[:,[1,2,6,10,31]], y2)
#probs_predicted = clf_2.predict_proba(X_stacked[:,[2,5,6,10,31]])
# Feature extraction
data_transformed_2 = feature_extraction(X_stacked,nc=nc2)
clf_2.fit(data_transformed_2, y2)
probs_predicted = clf_2.predict_proba(data_transformed_2)
y2_probs = probs_predicted[:,1:2]
scr2 = roc_auc_score(y2, y2_probs)
print("Score Task 2:",scr2)

########################
######## Task 3 ########
########################
# Symptoms (Heartrate: 31, Temp: 6, Lactate:5, Age:1, RRate:10, EtCO2:2, ABPm: 21
data_transformed_3 = feature_extraction(X_stacked,nc=nc3)
clf_3 = MultiOutputRegressor(SVR(kernel='poly', C=1, gamma=0.1, epsilon=.1))
clf_3.fit(data_transformed_3, y3)
#clf_3.fit(X_stacked[:,[10,21,27,31]], y3)
scr3 = r2_score(y3, clf_3.predict(data_transformed_3))
print("Score Task 3:",scr3)

avg_scr = (scr1 + scr2 + scr3) / 3
print("Average score: ", avg_scr)


########################
## Results Generation ##
########################
X_test = np.genfromtxt('Project_2/test_features.csv', delimiter=',')[1:,1:]
patient_id = np.atleast_2d(np.genfromtxt('Project_2/test_features.csv', delimiter=',')[1::12,0:1])

# Preprocessing
X_test = imputation(X_test,type="mean")
X_test = time_series_conc(X_test)

# Predictions
X_reduced_1 = feature_extraction(X_test,nc=nc1)
probs_predicted_1 = clf_1.predict_proba(X_reduced_1)
y_pred1 = arrange_probs(probs_predicted_1)

#y_pred2 = clf_2.predict_proba(X_test[:,[2,5,6,10,31]])[:,1:2]
X_reduced_2 = feature_extraction(X_test,nc=nc2)
y_pred2 = clf_2.predict_proba(X_reduced_2)[:,1:2]
X_reduced_3 = feature_extraction(X_test,nc=nc3)
y_pred3 = clf_3.predict(X_reduced_3)
y_pred = np.hstack((patient_id,y_pred1,y_pred2,y_pred3))

# Save results
df = pd.DataFrame(y_pred, columns = ['pid','LABEL_BaseExcess','LABEL_Fibrinogen','LABEL_AST','LABEL_Alkalinephos','LABEL_Bilirubin_total','LABEL_Lactate','LABEL_TroponinI','LABEL_SaO2','LABEL_Bilirubin_direct','LABEL_EtCO2','LABEL_Sepsis','LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate'])
df.to_csv('Project_2/prediction.zip', index=False, float_format='%.3f', compression='zip')
df.to_csv('Project_2/prediction.csv', index=False, float_format='%.3f')
print("Task completed")
