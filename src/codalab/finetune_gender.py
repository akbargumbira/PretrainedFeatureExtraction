# coding=utf-8
from keras.models import Sequential
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras.callbacks import ModelCheckpoint
from utilities import load_dataset


TOP_MODEL_WEIGHTS_PATH = 'model/gender/weights-top_model_last.hdf5'
BATCH_SIZE = 128

print 'Compiling the VGG network...'
# Build the VGG16 network and get only the bottleneck features
input_tensor = Input(shape=(150, 150, 3))
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_tensor=input_tensor)

# Add the top model layer that we have already trained
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))
top_model.load_weights(TOP_MODEL_WEIGHTS_PATH)

# add the model on top of the convolutional base
model = Model(
    input=base_model.input,
    output=top_model(base_model.output))

# Fine tune the base model by training the CONV BLOCK #5 and the seq only
for layer in model.layers[:15]:
    layer.trainable = False

# Compile the model with small learning rate
model.compile(
    loss='binary_crossentropy',
    optimizer=SGD(lr=1e-4, momentum=0.9),
    metrics=['accuracy']
)

print 'Loading training dataset...'
t_id, t_data, t_gender_label, _ = load_dataset()

print 'Preprocessing training dataset...'
t_data = preprocess_input(t_data)

print 'Loading validation dataset...'
val_id, val_data, val_gender_label, _ = load_dataset(prefix='val')

print 'Preprocessing validation dataset...'
val_data = preprocess_input(val_data)

print 'Training...'
# Checkpoint
filepath = 'model/gender/weights-final_model-improvement-{epoch:02d}-{' \
           'val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(
    filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Train the whole model now
model.fit(
    t_data,
    t_gender_label,
    epochs=5,
    batch_size=BATCH_SIZE,
    validation_data=(val_data, val_gender_label),
    callbacks=callbacks_list,
    verbose=True)
model.save_weights('model/gender/weights-final_model_last.hdf5')
