import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.neural_network import MLPClassifier
from score_submission import get_score

# labels
VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2']

# Data imputation
def impute_mean(train_features, test_features):
    X_train = (train_features.groupby('pid').mean()).fillna(train_features.mean()).drop('Time', axis=1).sort_values(by='pid')
    X_test = (test_features.groupby('pid').mean()).fillna(test_features.mean()).drop('Time', axis=1).sort_values(by='pid')
    return X_train, X_test

def impute_median(train_features, test_features):
    X_train = (train_features.groupby('pid').mean()).fillna(train_features.median()).drop('Time', axis=1).sort_values(by='pid')
    X_test = (test_features.groupby('pid').mean()).fillna(test_features.median()).drop('Time', axis=1).sort_values(by='pid')
    return X_train, X_test

def impute_zero(train_features, test_features):
    X_train = (train_features.groupby('pid').mean()).fillna(0.0).drop('Time', axis=1).sort_values(by='pid')
    X_test = (test_features.groupby('pid').mean()).fillna(0.0).drop('Time', axis=1).sort_values(by='pid')
    return X_train, X_test

def standardize_data(X_train, X_test):
    # Standardize data
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index)
    
    return X_train, X_test

def task1(X_train, y_train, X_test):
    y_pred = []
    for test in TESTS:
        y = y_train[test]
        # Create linear regressor and train it
        model = LogisticRegression(random_state=0).fit(X_train.values, y.values)
        # Make predicitons 
        y_pred.append(model.predict_proba(X_test.values)[:,1])
    return pd.DataFrame(np.transpose(y_pred), columns=TESTS, index=X_test.index)
    

def task2(X_train, y_train, X_test):
    model = MLPClassifier(solver='lbfgs', alpha=1e-3,hidden_layer_sizes=(5, 10), random_state=1, max_iter=20000)
    model.fit(X_train.values,y_train['LABEL_Sepsis'].values)
    y_pred = model.predict_proba(X_test.values)[:,1:2]
    return pd.DataFrame(y_pred, columns=['LABEL_Sepsis'], index=X_test.index)

def task3(X_train, y_train, X_test):
    y_pred = []
    for vital in VITALS:
        y = y_train[vital]
        # Create linear regressor and train it
        model  = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X_train, y)
        # Make predicitons 
        y_pred.append(model.predict(X_test))
    return pd.DataFrame(np.transpose(y_pred), columns=VITALS, index=X_test.index)

# Loading data
train_features = pd.read_csv('Project_2/train_features.csv')
val_features = pd.read_csv('Project_2/val_features.csv')
test_features = pd.read_csv('Project_2/test_features.csv')
train_labels = pd.read_csv('Project_2/train_labels.csv', index_col = 'pid').sort_values(by='pid')
val_labels = pd.read_csv('Project_2/val_labels.csv', index_col = 'pid').sort_values(by='pid')

y_train = train_labels
y_val = val_labels

# patient IDs
pids = train_features['pid'].drop_duplicates().sort_values().reset_index(drop=True)

X_train, X_test = impute_zero(train_features, test_features)
X_train, X_test = standardize_data(X_train, X_test)
X_val, _ = impute_zero(val_features, test_features)
X_val, _ = standardize_data(X_val, X_test)

labels_tests_test = task1(X_train, y_train, X_test)
labels_tests_val = task1(X_train, y_train, X_val)

X_train, X_test = impute_zero(train_features, test_features)
X_train, X_test = standardize_data(X_train, X_test)
X_val, _ = impute_zero(val_features, test_features)
X_val, _ = standardize_data(X_val, X_test)

label_sepsis_test = task2(X_train, y_train, X_test)
label_sepsis_val = task2(X_train, y_train, X_val)

X_train, X_test = impute_mean(train_features, test_features)
X_train, X_test = standardize_data(X_train, X_test)
X_val, _ = impute_mean(val_features, test_features)
X_val, _ = standardize_data(X_val, X_test)

labels_vitals_test = task3(X_train, y_train, X_test)
labels_vitals_val = task3(X_train, y_train, X_val)

result_val = pd.concat([labels_tests_val, label_sepsis_val, labels_vitals_val], axis=1)
result_test = pd.concat([labels_tests_test, label_sepsis_test, labels_vitals_test], axis=1)

print("Avg score for validation set:", get_score(y_val,result_val))

result_val.to_csv('Project_2/prediction_val.csv', float_format='%.3f')
result_test.to_csv('Project_2/prediction.zip', float_format='%.3f', compression='zip')

