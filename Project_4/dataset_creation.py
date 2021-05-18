import numpy as np

def progressBar(current, total, type, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))
    print('Progress',type,': [%s%s] %d %%' % (arrow, spaces, percent), end='\r')

idx = 0
one_hot = True
if(one_hot):
    food_file = 'Project_4/classification_one_hot.csv'
else:
    food_file = 'Project_4/classification_softmax.csv'

#################### Create training set ##########################
# Load
train_triplets = np.loadtxt('Project_4/train_triplets.txt').astype('int')
foods_array = np.genfromtxt(food_file,delimiter=' ').astype('int')

# Foods array concatenation
train_csv = open('Project_4/train.csv', 'ab')
train_set = np.empty((0,304))
for triplet in train_triplets:
    # Random correctness
    correctness = np.random.randint(0,2)
    if(correctness == 0):
        # Change triplet order
        triplet = triplet[[0,2,1]]

    # Dataset creation [0:303] is X and [304] is label
    np.savetxt(train_csv,np.hstack((np.hstack(foods_array[triplet]),correctness)))
    progressBar(idx,59515, 'train set')
    idx += 1
print("Train set generation completed")

#################### Create test set ##########################
idx = 0
# Load
test_triplets = np.loadtxt('Project_4/test_triplets.txt').astype('int')
foods_array = np.genfromtxt(food_file,delimiter=' ').astype('int')

# Foods array concatenation
test_csv = open('Project_4/test.csv', 'ab')
test_set = np.empty((0,303))
for triplet in test_triplets:
    # Test set creation [0:303] is X
    np.savetxt(test_csv,np.hstack(foods_array[triplet]))
    progressBar(idx,59544, 'test set')
    idx += 1
print("Test set generation completed")