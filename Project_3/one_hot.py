import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

train_features = pd.read_csv('Project_3/train.csv')
test_features = pd.read_csv('Project_3/test.csv')

dum = train_features['Sequence']
dummer = test_features['Sequence']

def one_hot(data_train, data_test):
    split_seq_train = np.empty((dum.shape[0],4), dtype=str)
    i=0
    for sequence in dum:
        split_seq_train[i,:] = np.array(list(sequence))
        i=i+1

    split_seq_test = np.empty((dummer.shape[0],4), dtype=str)
    i=0
    for sequence in dummer:
        split_seq_test[i,:] = np.array(list(sequence))
        i=i+1

    cat = list(sorted(np.unique(split_seq_train[:,0])))

    ohc = OneHotEncoder(sparse=False, categories=[cat,cat,cat,cat])
    hi=ohc.fit_transform(split_seq_train)
    feat=ohc.transform(split_seq_test)

    return hi, feat

hi, feat = one_hot(dum, dummer)