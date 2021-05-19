#############################################
#############################################
################# Project 4 #################
#############################################
#############################################

## Procedure:
# 1. Randomly invert half of the rows in the train_triplets for training
# 2. Save inverted array in RAM or file
# 3. Initialize network with 303 inputs for one hot encoding
# 4. For each training example take the one hot encoder and train
# 5. Save model and run in test set

################ Imports ####################
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

import pandas as pd

import matplotlib.pyplot as plt

################ Config params ##################
# Network parameters
l1_out = 500
l2_out = 600
l3_out = 700
l3_out = 800
l4_out = 700
l5_out = 600
l6_out = 500
l7_out = 400
l8_out = 200
l9_out = 50
l10_out = 5
n_epochs = 500
lr = 1e-3
wd = 0

# Training params
train_frac = 0.9
b_size = 50
only_test = False

################ Functions ##################
# Prediction
def predict_test(model):
    # Load datasets
    X_test = torch.from_numpy(np.genfromtxt('Project_4/test.csv',delimiter=',').astype('int'))
    y_predict_test = torch.round(model.predict(X_test.float()))

    # Convert to csv
    y_np = y_predict_test.detach().numpy()
    #print(y_np)
    #y_df = pd.DataFrame(y_np)
    np.savetxt("Project_4/results.csv",y_np,fmt='%i', delimiter="\n")

# Define network
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
        self.criterion = nn.BCELoss()
        # Use stochastic gradient descent for training
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.95, weight_decay=wd)
        self.plot_loss = plot_loss
    
    def train(self, X, Y, X_cv, Y_cv, batch_size=b_size):
        best_score_cv = 1
        best_epoch = 0
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

                if (loss_cv.item() < best_score_cv):
                    torch.save(self,'Project_4/best-model.pt')
                    best_score_cv = loss_cv.item()
                    best_epoch = epoch

                print('\nEpoch',epoch,'training loss: ', np.mean(losses))#BCE_loss_train_dB,'[dB]')
                print('Epoch',epoch,'validation loss: ', loss_cv.item())#BCE_loss_cv_dB,'[dB]') 
                print('Best loss: ',best_score_cv, '; epoch:', best_epoch)

        if self.plot_loss:
            plt.plot(epoch_losses)
            plt.show()
        
        print("Finished training")

    def predict(self, X):
        self.net.eval()
        return self.net(X)


################ Code ##################
if (not only_test):
    # Load datasets
    dataset = torch.from_numpy(np.genfromtxt('Project_4/train.csv',delimiter=',').astype('int'))

    # Split into validation and training set
    df_size = dataset.size(0)
    df_train = dataset[0:int(df_size*(train_frac)),:]
    df_val = dataset[0:int(df_size*(1-train_frac)),:]

    X_train = df_train[:,0:303]
    X_val = df_val[:,0:303]

    Y_train = df_train[:,303]
    Y_val = df_val[:,303]

    print("Dataset loaded...")
    # Classifier model and training
    protein_Net = Classifier(303,1,plot_loss=False)
    protein_Net.train(X_train.float(),Y_train.float(),X_val.float(),Y_val.float())

else:
    # Save results
    predict_test(torch.load('Project_4/best-model.pt'))