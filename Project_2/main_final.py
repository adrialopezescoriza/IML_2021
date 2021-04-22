import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from score_submission import get_score

## Parameters
val_percentage = 0
col_labels = ['pid','LABEL_BaseExcess','LABEL_Fibrinogen','LABEL_AST','LABEL_Alkalinephos','LABEL_Bilirubin_total','LABEL_Lactate','LABEL_TroponinI','LABEL_SaO2','LABEL_Bilirubin_direct','LABEL_EtCO2','LABEL_Sepsis','LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate']

def time_series_conc(X):
    patients_size = int(X.shape[0]/12)
    
    X_stacked = np.zeros((patients_size,X.shape[1]))
    for i in range(patients_size):
        # Mean evolution of the symptoms
        X_stacked[i,:] = np.nanmean(X[i*12:(i+1)*12,:],axis=0)
    
    return X_stacked

def imputation(X, type):
    if type == 'mean':
        imp = SimpleImputer(missing_values=np.nan, strategy='mean',fill_value=0)
    if type == 'zero':
        imp = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value=0.0)
    X_imp = imp.fit_transform(X)

    return X_imp

def standardization(X):
    return StandardScaler().fit_transform(X)

def task1_model(X,y=None):
    X = time_series_conc(X)
    X = imputation(X, 'zero') #zero
    X = standardization(X)
    return X, y[:,:-5]

def task2_model(X,y=None):
    X = time_series_conc(X)
    X = imputation(X, 'zero') #zero
    X = standardization(X)
    return X, y[:,-5:-4]

def task3_model(X,y=None):
    X = time_series_conc(X)
    X = imputation(X, 'mean') #mean
    X = standardization(X)
    return X, y[:,-4:]

## Load dataset
print("Loading Datasets...")
X = np.genfromtxt('Project_2/train_features.csv', delimiter=',')[1:,1:]
y = np.genfromtxt('Project_2/train_labels.csv', delimiter=',')[1:,1:]
pat_id = np.genfromtxt('Project_2/train_labels.csv', delimiter=',')[1:,0:1].astype(int)
X_test = np.genfromtxt('Project_2/test_features.csv', delimiter=',')[1:,1:]
pat_id_test = np.genfromtxt('Project_2/test_features.csv', delimiter=',')[1::12,0:1].astype(int)

if val_percentage == 0: #predicting for submission, no split needed
    X_train = X
    y_train = y
    X_val = X_test

    ## Model training data for each task
    print("Preprocessing data...")
    X1_train,y1_train = task1_model(X_train,y_train)
    X2_train,y2_train = task2_model(X_train,y_train)
    X3_train,y3_train = task3_model(X_train,y_train)

    X1_val,y_dummy = task1_model(X_val,np.zeros((10,10)))
    X2_val,y_dummy = task2_model(X_val,np.zeros((10,10)))
    X3_val,y_dummy = task3_model(X_val,np.zeros((10,10)))

else: # Split in training and validation
    print("Spliting Datasets...")
    cv_size = int(len(y)*val_percentage)
    X_train = X[:-(12*cv_size),:]
    y_train = y[:-cv_size,:]
    pat_id_train = pat_id[:-cv_size,:]

    X_val = X[-(12*cv_size):,:]
    y_val = y[-cv_size:,:]
    pat_id_val = pat_id[-cv_size:,:]

    ## Model training data for each task
    print("Preprocessing data...")
    X1_train,y1_train = task1_model(X_train,y_train)
    X2_train,y2_train = task2_model(X_train,y_train)
    X3_train,y3_train = task3_model(X_train,y_train)

    X1_val,y1_val = task1_model(X_val,y_val)
    X2_val,y2_val = task2_model(X_val,y_val)
    X3_val,y3_val = task3_model(X_val,y_val)

def subtask1(X_train, y_train, X_test):
    y_pred = []
    for i in range(np.shape(y_train)[1]):
        y = y_train[:,i]
        # Create linear regressor and train it
        #model = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X_train.values, y.values)
        model = LogisticRegression(random_state=0).fit(X_train, y)
        # Make predicitons 
        #y_pred.append(1/(1+np.exp(model.decision_function(X_test.values))))
        y_pred.append(model.predict_proba(X_test)[:,1])
    return np.transpose(y_pred)

def subtask2(X_train, y_train, X_test):
    model = LogisticRegression(random_state=0)
    #model = MLPClassifier(solver='lbfgs', alpha=1e-3,hidden_layer_sizes=(10, 10), random_state=1, max_iter=2000)
    model.fit(X_train,y_train)
    y_pred = model.predict_proba(X_test)[:,1:2]
    return y_pred

def subtask3(X_train, y_train, X_test):
    y_pred = []
    for i in range(np.shape(y_train)[1]):
        y = y_train[:,i]
        # Create linear regressor and train it
        #model  = LinearRegression().fit(X_train, y)
        model  = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X_train, y)
        # Make predicitons 
        y_pred.append(model.predict(X_test))
    return np.transpose(y_pred)

y1_pred = subtask1(X1_train, y1_train, X1_val)
y2_pred = subtask2(X2_train, y2_train, X2_val)
y3_pred = subtask3(X3_train, y3_train, X3_val)

if val_percentage == 0: #predicting based on test set
    print("Saving test data...")
    y_pred_test = np.hstack((pat_id_test,y1_pred,y2_pred,y3_pred))
    df_pred_test = pd.DataFrame(y_pred_test, columns = col_labels)
    df_pred_test.to_csv('Project_2/submission.csv', index=False, float_format='%.3f')
    df_pred_test.to_csv('Project_2/submission.zip', index=False, float_format='%.3f', compression='zip')

else: #predicting based on validation
    print("Validation scores:")
    y_pred_val = np.hstack((pat_id_val,y1_pred,y2_pred,y3_pred))
    y_true_val = np.hstack((pat_id_val,y1_val,y2_val,y3_val))
    df_pred_val = pd.DataFrame(y_pred_val, columns = col_labels)
    df_true_val = pd.DataFrame(y_true_val, columns = col_labels)
    score = get_score(df_true_val, df_pred_val)
    print("Average score:", score)
    # Save validation results
    print("Saving validation data...")
    df_pred_val.to_csv('Project_2/prediction_val.csv', index=False, float_format='%.3f')
    df_true_val.to_csv('Project_2/labels_val.csv', index=False, float_format='%.3f')

print("End")