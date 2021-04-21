import numpy as np
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn import svm
from sklearn.svm import SVR
from sklearn.linear_model import RidgeCV
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.decomposition import PCA
import pandas as pd

from score_submission import get_score

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
y2 = np.genfromtxt('Project_2/train_labels.csv', delimiter=',')[1:,11:12].astype(int)
y3 = np.genfromtxt('Project_2/train_labels.csv', delimiter=',')[1:,12:].astype(float)
pat_id = np.genfromtxt('Project_2/train_labels.csv', delimiter=',')[1:,0:1].astype(int)

#######################
#### Preprocessing ####
#######################

# Data Imputation
X = imputation(X,type="mean")
# Time series concatenation
X_stacked = time_series_conc(X)
# Split Dataset
train_size = int(len(X_stacked)*0.8)
X_train = X_stacked[:train_size,:]
print(np.shape(X_train))
y1_train = y1[:train_size,:]
y2_train = y2[:train_size,:]
y3_train = y3[:train_size,:]
pat_id_train = pat_id[:train_size,:]

X_val = X_stacked[train_size:,:]
print(np.shape(X_val))
y1_val = y1[train_size:,:]
y2_val = y2[train_size:,:]
y3_val = y3[train_size:,:]
pat_id_val = pat_id[train_size:,:]

########################
######## Task 3 ########
########################
# Symptoms (Heartrate: 31, Temp: 6, Lactate:5, Age:1, RRate:10, EtCO2:2, ABPm: 21
X_reduced_3 = feature_extraction(X_train,nc=nc3)
#clf_3 = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=5))
clf_3 = MultiOutputRegressor(SVR(kernel='rbf', C=10, gamma=0.1, epsilon=.1))
clf_3.fit(X_reduced_3, y3_train)
print("HEY")
#clf_3.fit(X_stacked[:,[10,21,27,31]], y3)

X_val_reduced_3 = feature_extraction(X_val,nc=nc3)
y_predict_val_3 = clf_3.predict(X_val_reduced_3)
scr3 = np.mean([0.5 + 0.5 * np.maximum(0, r2_score(y3_val, y_predict_val_3))])
print("Score Task 3:",scr3)

########################
######## Task 1 ########
########################
X_reduced_1 = feature_extraction(X_train,nc=nc1)
clf_1 = MultiOutputClassifier(svm.SVC(probability=True, C=20),n_jobs=10)
clf_1.fit(X_reduced_1, y1_train)

X_val_reduced_1 = feature_extraction(X_val,nc=nc1)
y_predict_val_1 = arrange_probs(clf_1.predict_proba(X_val_reduced_1))
scr1 = roc_auc_score(y1_val, y_predict_val_1)
print("Score Task 1:",scr1)

########################
######## Task 2 ########
########################
# PCA analysis
# Symptoms for Sepsis (Heartrate: 31, Temp: 6, Lactate:5, Age:1, RRate:10, EtCO2:2
#clf_2.fit(X_stacked[:,[1,2,6,10,31]], y2)
#probs_predicted = clf_2.predict_proba(X_stacked[:,[2,5,6,10,31]])
X_reduced_2 = feature_extraction(X_train,nc=nc2)
clf_2 = svm.SVC(probability=True, C=50)
clf_2.fit(X_reduced_2, y2_train)

X_val_reduced_2 = feature_extraction(X_val,nc=nc2)
y_predict_val_2 = clf_2.predict_proba(X_val_reduced_2)[:,1:2]
scr2 = roc_auc_score(y2_val, y_predict_val_2)
print("Score Task 2:",scr2)

## Score of all tasks in validation set
avg_scr = (scr1 + scr2 + scr3) / 3
print("Average score: ", avg_scr)

y_pred_val = np.hstack((pat_id_val,y_predict_val_1,y_predict_val_2,y_predict_val_3))
y_true_val = np.hstack((pat_id_val,y1_val,y2_val,y3_val))
df_pred_val = pd.DataFrame(y_pred_val, columns = ['pid','LABEL_BaseExcess','LABEL_Fibrinogen','LABEL_AST','LABEL_Alkalinephos','LABEL_Bilirubin_total','LABEL_Lactate','LABEL_TroponinI','LABEL_SaO2','LABEL_Bilirubin_direct','LABEL_EtCO2','LABEL_Sepsis','LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate'])
df_true_val = pd.DataFrame(y_true_val, columns = ['pid','LABEL_BaseExcess','LABEL_Fibrinogen','LABEL_AST','LABEL_Alkalinephos','LABEL_Bilirubin_total','LABEL_Lactate','LABEL_TroponinI','LABEL_SaO2','LABEL_Bilirubin_direct','LABEL_EtCO2','LABEL_Sepsis','LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate'])

# Score from score_submission.py
scroe = get_score(df_true_val, df_pred_val)

df_pred_val.to_csv('Project_2/prediction_val.csv', index=False, float_format='%.3f')
df_true_val.to_csv('Project_2/labels_val.csv', index=False, float_format='%.3f')

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
