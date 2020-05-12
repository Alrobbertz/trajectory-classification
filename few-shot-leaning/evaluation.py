import pickle
import numpy as np

from tensorflow.keras.models import load_model as load
from tensorflow import convert_to_tensor

# Local Imports
from data import *
from siamese import initialize_bias

validate = pickle.load(open("validate_set.pkl",'rb'))
label = pickle.load(open("validate_label.pkl",'rb'))

def load_model():
    '''
    Load your model
    '''
    model = load('best_model.h5', custom_objects={'initialize_bias':initialize_bias})

    return model


def process_data(traj_1, traj_2):
    """
    Input:
        Traj: a list of list, contains one trajectory for one driver 
        example:[[114.10437, 22.573433, '2016-07-02 00:08:45', 1],
            [114.179665, 22.558701, '2016-07-02 00:08:52', 1]]
    Output:
        Data: any format that can be consumed by your model.

    """
    
    [data1, outside] = process_trajectory(traj_1)
    [data2, outside] = process_trajectory(traj_2)
    
    data = [data1, data2]
    return np.array(data)

def run(data, model):
    """
    
    Input:
        Data: the output of process_data function.
        Model: your model.
    Output:
        prediction: the predicted label(plate) of the data, an int value.
    
    """
    [input_a, input_b] = data
    input_a = np.array(input_a)
    input_b = np.array(input_b)
    result = model.predict(x=[[input_a], [input_b]])

    return 1 if result >= 0.5 else 0


model = load_model()
s = 0
for d, l in zip(validate,label):
    data = process_data(d[0],d[1])
    prd = run(data,model)
    if l==prd:
        s+=1
print(s/len(validate))