########################
### Import libraries ###
########################

# Sklearn imports
from numpy.core.fromnumeric import mean
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

# Torch imports
import torch.nn as nn
import torch.optim as optim
import torch

# Other imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

########################
### Custom functions ###
########################
def code_dataframe(dataframe,test=False):
    converted_sequence = []
    for sequence in dataframe['Sequence']:
        coded_seq = np.zeros((1,4))
        i = 0
        for char in sequence:
            # Conver to ASCII code
            coded_seq[0,i] = ord(char)-64
            i += 1
        converted_sequence.append(coded_seq) 
    
    if(test):
        coded_data = {'Sequence': converted_sequence}
        return pd.DataFrame(coded_data)
    
    coded_data = {'Sequence': converted_sequence, 'Active':dataframe['Active']}
    return pd.DataFrame(coded_data)

def f1_loss(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=True) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
        
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    #if (f1==0):
    #    f1 += 1e-1
    #f1.requires_grad = is_training
    return 1-f1

# Prediction
def predict_test(model):
    df_str_test = pd.read_csv('Project_3/test.csv')
    df_int_test = code_dataframe(df_str_test,test=True)
    X_test = torch.from_numpy(np.vstack(df_int_test['Sequence'].values)).float()
    y_predict_test = torch.round(model.predict(X_test))

    # Convert to csv
    y_np = y_predict_test.detach().numpy()
    #print(y_np)
    #y_df = pd.DataFrame(y_np)
    np.savetxt("Project_3/results.csv",y_np,fmt='%i', delimiter="\n")

########################
### Code starts here ###
########################

# Tunable parameters
l1_out = 40
l2_out = 60
l3_out = 80
l3_out = 100
l4_out = 120
l5_out = 140
l6_out = 120
l7_out = 100
l8_out = 80
l9_out = 60
l10_out = 40
n_epochs = 500
lr = 1e-3
wd = 0

train_frac = 0.8
only_test = False

# Neural network
# Define the architecture of my classifier network
class Net(nn.Module):
    def __init__(self, size_input, size_output):
        super().__init__()
        self.fc1 = nn.Linear(size_input, l1_out, bias= True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(l1_out, l2_out, bias=True)
        self.fc3 = nn.Linear(l2_out, l3_out, bias=True)
        self.fc4 = nn.Linear(l3_out, l4_out, bias=True)
        self.fc5 = nn.Linear(l4_out, l5_out, bias=True)
        self.fc6 = nn.Linear(l5_out, l6_out, bias=True)
        self.fc7 = nn.Linear(l6_out, l7_out, bias=True)
        self.fc8 = nn.Linear(l7_out, l8_out, bias=True)
        self.fc9 = nn.Linear(l8_out, l9_out, bias=True)
        self.fc10 = nn.Linear(l9_out, l10_out, bias=True)
        self.fc11 = nn.Linear(l10_out, size_output, bias=True)
        self.sig= nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.relu(self.fc8(x))
        x = self.relu(self.fc9(x))
        x = self.relu(self.fc10(x))
        x = self.sig(self.fc11(x))
        return x

# Define How the classifier is going to be trained
class Classifier():
    def __init__(self, size_input, size_output, plot_loss=True):
        super().__init__()
        # Asign it the NN
        self.net = Net(size_input, size_output)
        # Since it's a binary classification problem, we use Binary cross entropy loss 
        #self.criterion = nn.BCELoss()
        self.criterion = f1_loss
        # Use stochastic gradient descent for training
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.95, weight_decay=wd)
        self.plot_loss = plot_loss
    
    def train(self, X, Y, X_cv, Y_cv, batch_size=50):
        best_score_cv = 0
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
                loss = self.criterion(torch.squeeze(outputs),torch.squeeze(batch_y))

                # backwards
                loss.backward()

                # optimize
                self.optimizer.step()
                losses.append(loss.data.numpy())
            
            epoch_losses += losses

            if((epoch%1) == 0):
                outputs_cv = self.net.forward(X_cv)
                loss_cv = self.criterion(torch.squeeze(outputs_cv),torch.squeeze(Y_cv))

                BCE_loss_train_dB = 10*np.log10(np.mean(losses))
                BCE_loss_cv_dB    = 10*np.log10(loss_cv.item())

                #f1_score_train = f1_score(Y_)
                f1_score_cv    = f1_score(y_cv.detach(),torch.round(outputs_cv.detach()))

                if (f1_score_cv > best_score_cv):
                    torch.save(self,'Project_3/best-model.pt')
                    best_score_cv = f1_score_cv

                print('\nEpoch',epoch,'training loss: ', 1-np.mean(losses))#BCE_loss_train_dB,'[dB]')
                print('Epoch',epoch,'validation loss: ', 1-loss_cv.item())#BCE_loss_cv_dB,'[dB]') 
                print('Epoch',epoch,'validation F1 Score: ', f1_score_cv) 

        if self.plot_loss:
            plt.plot(epoch_losses)
            plt.show()
        
        print("Finished training")

    def predict(self, X):
        self.net.eval()
        return self.net(X)

if (not only_test):
    # Load datasets
    df_str = pd.read_csv('Project_3/train.csv')
    df_int = code_dataframe(df_str)

    # Split into validation and training set
    df_shuffled = shuffle(df_int)
    df_size = df_shuffled['Sequence'].size
    df_train = df_shuffled.iloc[:round(df_size*train_frac),:]
    df_cv    = df_shuffled.iloc[round(df_size*train_frac):,:]

    # Classifier model and training
    protein_Net = Classifier(4,1,plot_loss=False)
    X,y = torch.from_numpy(np.vstack(df_train['Sequence'].values)).float(), torch.from_numpy(df_train['Active'].values).float()
    X_cv,y_cv = torch.from_numpy(np.vstack(df_cv['Sequence'].values)).float(), torch.from_numpy(df_cv['Active'].values).float() 
    protein_Net.train(X,y,X_cv,y_cv)

else:
    # Save results
    predict_test(torch.load('Project_3/best-model.pt'))









