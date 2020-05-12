import numpy as np
import pickle

from evaluation import load_model, process_data, run
from data import load_validation_data, pair_superset


# Load Model and Training Data
model = load_model()

(x_val, y_val) = load_validation_data()

# Save Predictions
predictions = []

# create test positive and negative pairs
te_pairs, te_y = pair_superset(x_val, y_val)


for i in range(len(te_pairs[:1000])):
    print(f'Running Test {i}')

    data = process_data(te_pairs[i][0], te_pairs[i][1])
 
    # print(f'data Shape : {data.shape}')

    result = run(data, model)
    predictions.append(result)
    

    
correct = [1 if predictions[i] == te_y[i] else 0 for i in range(len(predictions))]
print(f'Test Accuracy: {sum(correct)/len(correct)}')