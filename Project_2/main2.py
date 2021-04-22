import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn import svm
from sklearn.decomposition import PCA

from score_submission import get_score

## Parameters
val_percentage = 0.2
col_labels = ['pid','LABEL_BaseExcess','LABEL_Fibrinogen','LABEL_AST','LABEL_Alkalinephos','LABEL_Bilirubin_total','LABEL_Lactate','LABEL_TroponinI','LABEL_SaO2','LABEL_Bilirubin_direct','LABEL_EtCO2','LABEL_Sepsis','LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate']
nc1 = 12
nc2 = 12
nc3 = 12

def arrange_probs(probs_array):
    y_probs = np.zeros((len(probs_array[0]),0))
    for array in probs_array:
        # Return probability of class 1
        y_probs = np.hstack((y_probs,array[:,1:2]))
    return y_probs

def imputation(X):
    #imp = IterativeImputer(max_iter=3, random_state=0)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean',fill_value=0)
    X_imp = imp.fit_transform(X)
    return X_imp

def task1_model(X,y):
    # Deal with time series
    patients_size = int(len(X)/12)
    X_stacked = np.zeros((patients_size,X.shape[1]))
    for i in range(patients_size):
        # Mean of the symptoms
        X_stacked[i,:] = np.mean(X[i*12:(i+1)*12,:],axis=0)

    # Extract significant features
    feature_map = PCA(n_components=nc1)
    return feature_map.fit_transform(X_stacked), y[:,:-5]

def task2_model(X,y):
    # Symptoms for Sepsis (Age:1, EtCO2:2, Lactate:5, Temp: 6, RRate:10, Heartrate: 31)
    patients_size = int(X.shape[0]/12)
    X_stacked = np.zeros((patients_size,X.shape[1]))
    for i in range(patients_size):
        # Mean evolution of the symptoms
        X_stacked[i,:] = np.mean(X[i*12:(i+1)*12-1,:]-X[i*12+1:(i+1)*12],axis=0)
    feature_map = PCA(n_components=nc1)
    return feature_map.fit_transform(X_stacked), y[:,-5:-4]

def task3_model(X,y):
    # Deal with time series
    patients_size = int(len(X)/12)
    X_stacked = np.zeros((patients_size,X.shape[1]))
    for i in range(patients_size):
        # Mean of the symptoms
        X_stacked[i,:] = np.mean(X[i*12:(i+1)*12,:],axis=0)

    # Extract significant features
    feature_map = PCA(n_components=nc3)
    return feature_map.fit_transform(X_stacked), y[:,-4:]

## Load dateset
print("Loading Datasets...")
X = np.genfromtxt('Project_2/train_features.csv', delimiter=',')[1:,1:]
y = np.genfromtxt('Project_2/train_labels.csv', delimiter=',')[1:,1:]
pat_id = np.genfromtxt('Project_2/train_labels.csv', delimiter=',')[1::12,0:1].astype(int)

## Split in training and validation
print("Spliting Datasets...")
cv_size = int(len(y)*val_percentage)
X_train = X[:-(12*cv_size),:]
y_train = y[:-cv_size,:]
pat_id_train = pat_id[:-cv_size,:]

X_val = X[-(12*cv_size):,:]
y_val = y[-cv_size:,:]
pat_id_val = pat_id[-cv_size:,:]

## Model training data for each task
print("Preprocsessing data...")
X_train = imputation(X_train)
X1_train,y1_train = task1_model(X_train,y_train)
X2_train,y2_train = task2_model(X_train,y_train)
X3_train,y3_train = task3_model(X_train,y_train)

X_val = imputation(X_val)
X1_val,y1_val = task1_model(X_val,y_val)
X2_val,y2_val = task2_model(X_val,y_val)
X3_val,y3_val = task3_model(X_val,y_val)

## Machine learning models
model1 = MultiOutputClassifier(svm.SVC(probability=True, C=10),n_jobs=10)
model2 = svm.SVC(probability=True, C=10)
model3 = MultiOutputRegressor(svm.SVR(kernel='poly',degree=3, C=10, gamma='scale', epsilon=.1))

## Fit models
print("Fiting to model 1...")
model1.fit(X1_train,y1_train)
print("Fiting to model 2...")
model2.fit(X2_train,y2_train)
print("Fiting to model 3...")
model3.fit(X3_train,y3_train)

## Predict validation set
print("Predicting...")
y1_pred = arrange_probs(model1.predict_proba(X1_val))
y2_pred = model2.predict_proba(X2_val)[:,1:2]
y3_pred = model3.predict(X1_val)

## Validation score
print("Scores:")
y_pred_val = np.hstack((pat_id,y1_pred,y2_pred,y3_pred))
y_true_val = np.hstack((pat_id,y1_val,y2_val,y3_val))
df_pred_val = pd.DataFrame(y_pred_val, columns = col_labels)
df_true_val = pd.DataFrame(y_true_val, columns = col_labels)
score = get_score(df_true_val, df_pred_val)
print("Average score:", score)

# Save validation results
print("Saving data...")
df_pred_val.to_csv('Project_2/prediction_val.csv', index=False, float_format='%.3f')
df_true_val.to_csv('Project_2/labels_val.csv', index=False, float_format='%.3f')

print("End")