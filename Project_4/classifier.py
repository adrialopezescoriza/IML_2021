#############################################
#############################################
################# Project 4 #################
#############################################
#############################################

################ Imports ####################
import torch
import torch.nn as nn
import torchvision as tv
from torchvision import datasets, transforms
import numpy as np
import torch.nn.functional as F

import helper
import matplotlib.pyplot as plt

################ Config params ##################
batch_size = 50
epochs = int(10000/(batch_size*10))
one_hot = False

################ Functions ##################

# Load image classification model
classifier_model = torch.load("Project_4/resnet50model.pth", map_location='cpu')['model']
classifier_model.eval()

# Predict labels and save image classification
transform = transforms.Compose([transforms.Resize(256),
                                transforms.TenCrop(224),
                                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(crop) for crop in crops]))])

dataset = datasets.ImageFolder('Project_4/', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

file_csv = open('Project_4/classification_softmax.csv', 'ab')
softmax = nn.Softmax(dim=1)

for epoch in range(0,epochs):
    images, labels = next(iter(dataloader))
    #plt.imshow(np.transpose(images[0,0], (1, 2, 0)))
    #plt.show()
    for idx in range(0,batch_size):
        output = softmax(classifier_model(images[idx]))

        # One hot encoding
        if(one_hot):
            for i in range(0,output.size(0)):
                mask = output[i,:] >= torch.max(output[i,:])
                output[i,mask] = 1
                output[i,~mask] = 0        

        print("Epoch:",epoch,"; Idx:",idx)
        np.savetxt(file_csv,output.detach().numpy())
