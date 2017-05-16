# coding=utf-8
from keras.models import Sequential
from keras.applications import VGG16
from keras.optimizers import SGD
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from utilities import load_dataset

TOP_MODEL_WEIGHTS_PATH = 'model/weights-top_model-improvement-47-0.97.hdf5'
TRAINING_ID, TRAINING_DATA, TRAINING_LABEL = load_dataset()

# Build the VGG16 network and get only the bottleneck features
model = VGG16(weights='imagenet', include_top=False)
# Add the top model layer that we have already trained
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output.shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))
top_model.load_weights(TOP_MODEL_WEIGHTS_PATH)

# Fine tune the base model by training the CONV BLOCK #5 only
for layer in model.layers[:25]:
    layer.trainable = False

# Compile the model with small learning rate
model.compile(
    loss='binary_crossentropy',
    optimizer=SGD(lr=1e-4),
    metrics=['accuracy']
)

# Checkpoint
filepath = 'model/weights-final_model-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(
    filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Train the whole model now
model.fit(
    TRAINING_DATA,
    TRAINING_LABEL,
    epochs=50,
    validation_split=0.3,
    callbacks=callbacks_list,
    verbose=True)
model.save_weights('model/weights-final_model_last.hdf5')
