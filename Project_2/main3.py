import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, RidgeClassifierCV, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.decomposition import PCA
from sklearn import svm
import matplotlib.pyplot as plt

from score_submission import get_score

# parameters:
l1_out = 100 # output size of first linear layer
l2_out = 120 # output size of second linear layer
l3_out = 140
n_epochs = 30

# labels
VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
         
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

# Define the architecture of my classifier network
class Net(nn.Module):
    def __init__(self, size_input, size_output):
        super().__init__()
        self.fc1 = nn.Linear(size_input, l1_out, bias= True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(l1_out, l2_out, bias=True)
        self.fc3 = nn.Linear(l2_out, l3_out, bias=True)
        self.fc4 = nn.Linear(l3_out, size_output, bias=True)
        self.sig= nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sig(self.fc4(x))
        return x

# Define How the classifier is going to be trained

class Classifier():
    def __init__(self, size_input, size_output, plot_loss=False):
        super().__init__()
        # Asign it the NN
        self.net = Net(size_input, size_output)
        # Since it's a binary classification problem, we use Binary cross entropy loss 
        self.criterion = nn.BCELoss()
        # Use stochastic gradient descent for training
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.95)
        self.plot_loss = plot_loss
    
    def train(self, X, Y, batch_size=50):
        
        epoch_losses=[]
        for epoch in range(n_epochs):  # loop over the dataset multiple times

            # X is a torch Variable
            permutation = torch.randperm(X.size()[0])
            losses = []

            for i in range(0,X.size()[0], batch_size):
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Split the data in mini batches
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X[indices], Y[indices]

                # forwards
                outputs = self.net.forward(batch_x)
                loss = self.criterion(outputs,batch_y)

                # backwards
                loss.backward()

                # optimize
                self.optimizer.step()
                
                losses.append(loss.data.numpy())
            
            epoch_losses += losses 

        if self.plot_loss:
            plt.plot(epoch_losses)
            plt.show()
        
        print("Finished training")

    def predict(self, X):
        self.net.eval()
        return self.net(X)

def subtask1(X_train, y_train, X_test):
    '''
    X = torch.Tensor(X_train.values)
    y = torch.Tensor(y_train[TESTS].values)
    X_pred = torch.Tensor(X_test.values)
    model = Classifier(35, 10)
    model.train(X, y)
    y_pred = model.predict(X_pred)
    return pd.DataFrame(y_pred.detach().numpy(), columns=TESTS, index=X_test.index)
    '''
    # Extract significant features
    feature_map = PCA(n_components=12)
    #feature_map.fit_transform(X_train.values)
    y_pred = []
    for test in TESTS:
        y = y_train[test]

        # Create linear regressor and train it
        #model = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X_train.values, y.values)
        model = LogisticRegression(random_state=0).fit(X_train.values, y.values)
        # Make predicitons 
        #y_pred.append(1/(1+np.exp(model.decision_function(X_test.values))))
        y_pred.append(model.predict_proba(X_test.values)[:,1])
    '''    
    #model = MultiOutputClassifier(svm.SVC(probability=True, C=10),n_jobs=10)
    model = MultiOutputClassifier(RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1]))
    model.fit(X_train.values, y_train[TESTS].values)
    y_pred = model.decision_function(X_test.values)
    y_probs = np.zeros((len(y_pred[0]),0))
    
    for array in y_pred:
        # Return probability of class 1
        y_probs = np.hstack((y_probs,array[:,1:2]))
    '''
    return pd.DataFrame(np.transpose(y_pred), columns=TESTS, index=X_test.index)
    

def subtask2(X_train, y_train, X_test):
    X = torch.Tensor(X_train.values)
    y = torch.Tensor([y_train['LABEL_Sepsis'].values]).transpose(0, 1)
    X_pred = torch.Tensor(X_test.values)
    #model = Classifier(35, 1)
    #model.train(X, y)
    #y_pred = model.predict_proba(X_pred)
    #return pd.DataFrame(y_pred.detach().numpy(), columns=['LABEL_Sepsis'], index=X_test.index)

    # Extract significant features
    #feature_map = PCA(n_components=12)
    #feature_map.fit_transform(X_train.values)
    #model = svm.SVC(probability=True, C=0.1)
    train_labels = ['Age','EtCO2','Temp','RRate','Heartrate','ABPs']
    model = LogisticRegression(random_state=0)
    model.fit(X_train['Age','EtCO2','Temp','RRate','Heartrate','ABPs'].values,y_train['LABEL_Sepsis'].values)
    y_pred = model.predict_proba(X_test['Age','EtCO2','Temp','RRate','Heartrate','ABPs'].values)[:,1:2]
    return pd.DataFrame(y_pred, columns=['LABEL_Sepsis'], index=X_test.index)

def subtask3(X_train, y_train, X_test):
    y_pred = []

    for vital in VITALS:
        y = y_train[vital]

        # Create linear regressor and train it
        model  = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X_train, y)
        # model = LassoCV(random_state=42).fit(X_train, y)

        # Make predicitons 
        y_pred.append(model.predict(X_test))

    return pd.DataFrame(np.transpose(y_pred), columns=VITALS, index=X_test.index)

def make_submission(train_features, y_train, val_features, y_val, test_features):
    X_train, X_test = impute_zero(train_features, test_features)
    X_train, X_test = standardize_data(X_train, X_test)
    X_val, _ = impute_zero(val_features, test_features)
    X_val, _ = standardize_data(X_val, X_test)
    
    labels_tests_test = subtask1(X_train, y_train, X_test)
    label_sepsis_test = subtask2(X_train, y_train, X_test)

    labels_tests_val = subtask1(X_train, y_train, X_val)
    label_sepsis_val = subtask2(X_train, y_train, X_val)

    X_train, X_test = impute_mean(train_features, test_features)
    X_train, X_test = standardize_data(X_train, X_test)
    X_val, _ = impute_mean(val_features, test_features)
    X_val, _ = standardize_data(X_val, X_test)

    labels_vitals_test = subtask3(X_train, y_train, X_test)
    labels_vitals_val = subtask3(X_train, y_train, X_val)

    result_val = pd.concat([labels_tests_val, label_sepsis_val, labels_vitals_val], axis=1)
    result_test = pd.concat([labels_tests_test, label_sepsis_test, labels_vitals_test], axis=1)

    print("Avg score for validation set:", get_score(y_val,result_val))

    result_val.to_csv('Project_2/prediction_val.csv', float_format='%.3f')
    result_test.to_csv('Project_2/prediction.zip', float_format='%.3f', compression='zip')

make_submission(train_features, y_train, val_features, y_val, test_features)