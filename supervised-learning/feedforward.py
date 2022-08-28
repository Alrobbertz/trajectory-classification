import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow import convert_to_tensor

# Set Random Seed
np.random.seed(7)

# Set Training Parameters
validation_split = 0.1
batch_size = 64
steps_per_epoch = 10
epochs = 20



# Load and Process the Training Data
x_train, y_train = pickle.load(open('train_data_ff.pkl', 'rb'))

# Clean & Pre-Process Data
x_train = convert_to_tensor(x_train, dtype='float')
y_train = to_categorical(y_train, num_classes=5)

print(f'Features Shape: {x_train.shape}')
print(f'Label Shape: {y_train.shape}')


# Create the Model
model = Sequential()
model.add(Input(shape=(414)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(units=5, activation='softmax'))

# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file='feedforward.png', show_shapes=True)

# Compile the Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Set log data to feed to Tensor Board for visual analysis
tensor_board = TensorBoard('.\logs\_feed_forward2')

# Simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc = ModelCheckpoint('best_ff.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# Train the model
import time

start_time = time.time()

# model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=64, verbose=1, callbacks=[tensor_board])
model.fit(x_train, y_train, validation_split=validation_split,
                    epochs=epochs, batch_size=batch_size,
                    shuffle=True, steps_per_epoch=steps_per_epoch,
                    workers=6, use_multiprocessing=True,
                    verbose=1, callbacks=[tensor_board, es, mc])

print('Training took {} seconds'.format(time.time() - start_time))

# Save the Model
model.save('model.h5')
print('Saved Model to file')