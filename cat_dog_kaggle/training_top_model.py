# coding=utf-8
import numpy as np
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Flatten, Dense

# Get the bottleneck features
with np.load('model/bottleneck_features.npz') as data:
    training_data = data['arr_0.npy']

# Get the label
with np.load('model/training_label.npz') as data:
    training_label = data['arr_0.npy']

model = Sequential()
model.add(Flatten(input_shape=training_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Checkpoint
filepath = 'model/weights-top_model-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(
    filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(
    training_data, training_label,
    epochs=50,
    validation_split=0.3,
    callbacks=callbacks_list,
    verbose=True)
model.save_weights('model/weights-top_model_last.hdf5')

