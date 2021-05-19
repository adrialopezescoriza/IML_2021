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
from torch.utils.data.sampler import SequentialSampler

import helper
import matplotlib.pyplot as plt

################ Config params ##################
batch_size = 50
one_hot = True

image_type = np.loadtxt("Project_4/food/classes.txt",dtype='str')
################ Functions ##################
def progressBar(current, total, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))
    print('Analyzing shitty food images: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')

# Load image classification model
classifier_model = torch.load("Project_4/resnet50model.pth", map_location='cpu')['model']
classifier_model.eval()

# Predict labels and save image classification
transform = transforms.Compose([transforms.Resize(256),
                                transforms.TenCrop(224),
                                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(crop) for crop in crops]))])

transform_show = transforms.Compose([transforms.Resize(256),
                                transforms.TenCrop(224),
                                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))])
                                #transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(crop) for crop in crops]))])

dataset = datasets.ImageFolder('Project_4/', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

dataset_show = datasets.ImageFolder('Project_4/', transform=transform_show)
dataloader_show = torch.utils.data.DataLoader(dataset_show, batch_size=batch_size, shuffle=False)

file_csv = open('Project_4/classification_one_hot.csv', 'ab')
softmax = nn.Softmax(dim=0)
sampler = SequentialSampler(dataset)

epoch = 0
i = 0
for images,labels in dataloader:
    epoch = epoch + 1
    for idx in range(0,batch_size):
        output = softmax(torch.mean(classifier_model(images[idx]),dim=0))
        # One hot encoding
        if(one_hot):
            mask = output >= torch.max(output)
            output[mask] = 1
            output[~mask] = 0 
            fmt = '%i'
        else:
            fmt='%.4f'
        i = i+1        
        progressBar(i,10000)
        np.savetxt(file_csv,output.detach().numpy()[None],fmt=fmt)

        #print(image_type[torch.argmax(output)])
        #plt.imshow(np.transpose(images[idx,0], (1, 2, 0)))
        #plt.show()
    
file_csv.close()
