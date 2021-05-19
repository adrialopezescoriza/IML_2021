import numpy as np

def progressBar(current, total, type, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))
    print('Progress',type,': [%s%s] %d %%' % (arrow, spaces, percent), end='\r')

idx = 0
one_hot = False
if(one_hot):
    food_file = 'Project_4/classification_one_hot.csv'
    fmt = '%i'
    type = 'int'
else:
    food_file = 'Project_4/classification_softmax.csv'
    fmt = '%.3f'
    type = 'float'

#################### Create training set ##########################
# Load
train_triplets = np.loadtxt('Project_4/train_triplets.txt').astype('int')
foods_array = np.genfromtxt(food_file,delimiter=' ').astype(type)

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
    array = np.hstack((np.hstack(foods_array[triplet]),correctness))
    np.savetxt(train_csv,array[None],fmt=fmt,delimiter=',')
    progressBar(idx,59515, 'train set')
    idx += 1
train_csv.close()
#################### Create test set ##########################
idx = 0
# Load
test_triplets = np.loadtxt('Project_4/test_triplets.txt').astype('int')
foods_array = np.genfromtxt(food_file,delimiter=' ').astype(type)

# Foods array concatenation
test_csv = open('Project_4/test.csv', 'ab')
image_type = np.loadtxt("Project_4/food/classes.txt",dtype='str')
test_set = np.empty((0,303))
for triplet in test_triplets:
    # Test set creation [0:303] is X
    array = np.hstack(foods_array[triplet])
    '''
    image_A = image_type[np.argmax(array[0:101])]
    image_B = image_type[np.argmax(array[101:202])]
    image_C = image_type[np.argmax(array[202:303])]
    print(image_A,image_B,image_C)
    '''

    np.savetxt(test_csv,array[None],fmt=fmt,delimiter=',')
    progressBar(idx,59544, 'test set')
    idx += 1
test_csv.close()