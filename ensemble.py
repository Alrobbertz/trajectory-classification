import pickle
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import convert_to_tensor
from tensorflow.keras.utils import to_categorical, plot_model


# Set Training Parameters
max_traj_length = 2000

validation_split = 0.2

batch_size = 64
steps_per_epoch = 10
epochs = 100

# Load and Process the Training Data
(x_train_p, x_train_np, x_train_gen, y_train) = pickle.load(open('train_data_ensemble.pkl', 'rb'))

print(f'Passenger Series Shape: {x_train_p.shape}')
print(f'No Passenger Series Shape: {x_train_np.shape}')
print(f'Label Shape: {y_train.shape}')

x_train_p = convert_to_tensor(x_train_p, dtype='float')
x_train_np = convert_to_tensor(x_train_np, dtype='float')
x_train_gen = convert_to_tensor(x_train_gen, dtype='float')

y_train = to_categorical(y_train, num_classes=5)

print(f'Passenger Series Shape: {x_train_p.shape}')
print(f'No Passenger Series Shape: {x_train_np.shape}')
print(f'Label Shape: {y_train.shape}')

# Create the Model
# first input model
visible1 = Input(shape=(max_traj_length, 3))
lstm11 = LSTM(64, return_sequences=True, dropout=.1)(visible1)
lstm12 = LSTM(64, return_sequences=True, dropout=.1)(lstm11)
flat1 = LSTM(32)(lstm12)
# second input model
visible2 = Input(shape=(max_traj_length, 3))
lstm21 = LSTM(64, return_sequences=True, dropout=.1)(visible2)
lstm22 = LSTM(64, return_sequences=True, dropout=.1)(lstm21)
flat2 = LSTM(32)(lstm22)
# third input mmodel
visible3 = Input(shape=(410))
d1 = Dropout(0.1)(visible3)
l1 = Dense(64, activation='relu')(d1)
d2 = Dropout(0.1)(l1)
l2 = Dense(64, activation='relu')(d2)
flat3 = Dense(32, activation='relu')(l2)
# merge input models
merge = concatenate([flat1, flat2, flat3])
# interpretation model
hidden1 = Dense(64, activation='relu')(merge)
hidden2 = Dense(32, activation='relu')(hidden1)
output = Dense(5, activation='softmax')(hidden2)
model = Model(inputs=[visible1, visible2, visible3], outputs=output)
# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file='multiple_inputs.png', show_shapes=True)

# Compile the Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Set log data to feed to Tensor Board for visual analysis
tensor_board = TensorBoard('.\logs\early-stop-drpoout3')

# Simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc = ModelCheckpoint('best_dropout3.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# Train the model
import time

start_time = time.time()

# model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=64, verbose=1, callbacks=[tensor_board])
model.fit([x_train_p, x_train_np, x_train_gen], y_train, validation_split=validation_split,
                    epochs=epochs, batch_size=batch_size,
                    shuffle=True, steps_per_epoch=steps_per_epoch,
                    verbose=1, callbacks=[tensor_board, es, mc])

print('Training took {} seconds'.format(time.time() - start_time))

# Save the Model
model.save('model.h5')
print('Saved Model to file')