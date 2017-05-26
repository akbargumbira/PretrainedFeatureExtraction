# coding=utf-8
import numpy as np
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import RMSprop


def train(previous_model=None, lr=0.001, epochs=50):
    model = Sequential()
    model.add(Flatten(input_shape=training_data.shape[1:]))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    if previous_model:
        model.load_weights(previous_model)

    model.compile(optimizer=RMSprop(lr=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # # Checkpoint
    filepath = 'model/smile/weights-top_model-improvement-{epoch:02d}-{' \
               'val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit(
        training_data,
        training_label,
        epochs=epochs,
        validation_data=(val_data, val_label),
        callbacks=callbacks_list,
        verbose=True)
    model.save_weights('model/smile/weights-top_model_last.hdf5')


# Load training data and label
with np.load('model/training_bottleneck_features.npz') as data:
    training_data = data['arr_0.npy']
with np.load('model/training_smile_label.npz') as data:
    training_label = data['arr_0.npy']

# Load validation data
with np.load('model/val_bottleneck_features.npz') as data:
    val_data = data['arr_0.npy']
with np.load('model/val_smile_label.npz') as data:
    val_label = data['arr_0.npy']

# # First training
# train()

# 2nd training with smaller lr
train(
    previous_model='model/smile/top_model_1/weights-top_model_last.hdf5',
    lr=1e-4
)

# # With smaller lr
# train(
#     previous_model='model/gender/top_model_2/weights-top_model_last.hdf5',
#     lr=1e-5,
#     epochs=100,
# )
