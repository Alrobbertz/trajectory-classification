import pickle
import numpy as np
import numpy.random as rng
import random
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Flatten, Dropout, Lambda
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow import convert_to_tensor
from tensorflow.keras.utils import to_categorical, plot_model
# Local Import 
from data import create_pairs, pair_superset

def initialize_bias(shape, name=None, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() >= 0.5
    return np.mean(pred == y_true)


def evaluate_accuracy(pairs, y, step='train'):
    # compute final accuracy on given sets
    pos = []
    neg =  []
    for i in range(len(y)):
        if y[i] > 0 :
            pos.append(pairs[i])
        else:
            neg.append(pairs[i])
    pos = np.array(pos)
    neg = np.array(neg)

    print(f'{step} Pos : {pos.shape}')
    print(f'{step} NEG : {neg.shape}')

    y_pred_pos = model.predict([pos[:, 0], pos[:, 1]])
    acc_pos = compute_accuracy(np.ones(len(pos)), y_pred_pos)

    y_pred_neg = model.predict([neg[:, 0], neg[:, 1]])
    acc_neg = compute_accuracy(np.zeros(len(neg)), y_pred_neg)
    return len(pos), acc_pos, len(neg), acc_neg


def create_encoder():
    # input mmodel
    visible = Input(shape=(410))
    l1 = Dense(512, activation='relu')(visible)
    # d1 = Dropout(0.1)(l1)
    l2 = Dense(256, activation='relu')(l1)
    # d2 = Dropout(0.1)(l2)
    l3 = Dense(128, activation='relu')(l2)
    encoding = Dense(64, activation='relu')(l3)
    base_encoder = Model(inputs=[visible], outputs=encoding)
    return base_encoder


def large_encoder():
    # input mmodel
    visible = Input(shape=(410))
    l1 = Dense(1024, activation='relu')(visible)
    # d1 = Dropout(0.1)(l1)
    l2 = Dense(512, activation='relu')(l1)
    # d2 = Dropout(0.1)(l2)
    l3 = Dense(256, activation='relu')(l2)
    encoding = Dense(128, activation='relu')(l3)
    base_encoder = Model(inputs=[visible], outputs=encoding)
    return base_encoder

def create_model():
    # # Create the Model
    input_a = Input(shape=410)
    input_b = Input(shape=410)

    # Load the Shared Encoder 
    base_encoder = create_encoder()
    # base_encoder = large_encoder()

    # Add the Comparisson Model
    encoded_a = base_encoder(input_a)
    encoded_b = base_encoder(input_b)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_distance = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))([encoded_a, encoded_b])
    # L2_distance = Lambda(lambda tensors:K.square(tensors[0] - tensors[1]))([encoded_a, encoded_b])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid',bias_initializer=initialize_bias)(L1_distance)

    model = Model(inputs=[input_a, input_b], outputs=prediction)
    return model


"""

TRAINING

"""

# Set File Save Locations
base_fig = '.\images\_base.png'
model_fig = '.\images\large-squared.png'
tensor_dir = '.\logs\large-squared'
mc_dir = '.\models\large-squared.h5'

# Set Training Parameters
num_classes = 500

validation_split = 0.2
batch_size = 64
steps_per_epoch = 10
epochs = 200

# Load and Process the Training Data
(x_train, y_train) = pickle.load(open('.\data\_train_data_simple.pkl', 'rb'))

# Load and Process the Testing Data
(x_val, y_val) = pickle.load(open('.\data\_val_data_simple.pkl', 'rb'))

# create training positive and negative pairs
digit_indices = np.array([np.where(y_train == i)[0] for i in range(num_classes)])
tr_pairs, tr_y = create_pairs(x_train, digit_indices)

# create test positive and negative pairs
te_pairs, te_y = pair_superset(x_val, y_val)


print(f'Training Generated {len(tr_pairs)} Pairs of trajectories')
print(f'Training Data Shape : {tr_pairs.shape}')
print(f'Training LSTM A1 Shape : {tr_pairs[:, 0].shape}')

print(f'Testing Generated {len(te_pairs)} Pairs of trajectories')
print(f'Testing Data Shape : {te_pairs.shape}')
print(f'Testing LSTM A1 Shape : {te_pairs[:, 0].shape}')


# Create the Siamese Network Modee
model = create_model()

# plot graph
# plot_model(base_encoder, to_file=base_fig, show_shapes=True, show_layer_names=True, expand_nested=True)
plot_model(model, to_file=model_fig, show_shapes=True, show_layer_names=True, expand_nested=True)
 
# Compile the Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Set log data to feed to Tensor Board for visual analysis
tensor_board = TensorBoard(tensor_dir)

# Simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc = ModelCheckpoint(mc_dir, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

# Train the model
start_time = time.time()

model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, 
                    use_multiprocessing = True,
                    validation_split=validation_split,
                    epochs=epochs, batch_size=batch_size,
                    # steps_per_epoch=steps_per_epoch,
                    shuffle=True, verbose=1,
                    callbacks=[tensor_board, es, mc])

print('Training took {} seconds'.format(time.time() - start_time))


# Evaluate final Training and Test Accuracy
tr_num_pos, tr_acc_pos, tr_num_neg, tr_acc_neg = evaluate_accuracy(tr_pairs, tr_y, step='Training')
te_num_pos, te_acc_pos, te_num_neg, te_acc_neg = evaluate_accuracy(te_pairs, te_y, step='Testing')

print('* Accuracy on POS training set: %0.2f%%' % (100 * tr_acc_pos))
print('* Accuracy on NEG training set: %0.2f%%' % (100 * tr_acc_neg))
print('* Accuracy on POS test set: %0.2f%%' % (100 * te_acc_pos))
print('* Accuracy on NEG test set: %0.2f%%' % (100 * te_acc_neg))
print(f'* Overall Training Accuracy {(tr_acc_pos*tr_num_pos + tr_acc_neg*tr_num_neg)/(len(tr_pairs))}')
print(f'* Overall Testing Accuracy {(te_acc_pos*te_num_pos + te_acc_neg*te_num_neg)/(len(te_pairs))}')
