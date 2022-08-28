import pickle
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Flatten, Dropout, concatenate
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow import convert_to_tensor
from tensorflow.keras.utils import to_categorical, plot_model


# Set Training Parameters
max_traj_length = 2000
validation_split = 0.2
batch_size = 32
steps_per_epoch = 20
epochs = 20



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
lstm1 = LSTM(64, dropout=.1)(visible1)
flat1 = Dense(32, activation='relu')(lstm1)
# second input model
visible2 = Input(shape=(max_traj_length, 3))
lstm2 = LSTM(64, dropout=.1)(visible2)
flat2 = Dense(32, activation='relu')(lstm2)
# third input mmodel
visible3 = Input(shape=(410))
l1 = Dense(128, activation='relu')(visible3)
l2 = Dense(64, activation='relu')(l1)
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