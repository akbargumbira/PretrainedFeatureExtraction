# coding=utf-8
from keras.models import Sequential
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras.callbacks import ModelCheckpoint
from src.utilities import load_dataset, get_full_model



TOP_MODEL_WEIGHTS_PATH = 'codalab/224_224/model/gender/vgg16/weights' \
                         '-last-200-0.80.hdf5'
IMAGE_SIZE = (224, 224)
BASE_ARCHITECTURE = 'inceptionv3'
n_classes = 3

full_model = get_full_model(
    IMAGE_SIZE, BASE_ARCHITECTURE, n_target_classes=n_classes,
    top_model_weights_path=TOP_MODEL_WEIGHTS_PATH)

n_last = {
    1: 17, #trainable starting from 17
    2: 16,
    3: 15,
    4: 13,
    5: 12,
    6: 11
}
# Fine tune the base model by training the CONV BLOCK #5 and the seq only
# for layer in full_model.layers[:15]:
#     layer.trainable = False

loss = 'categorical_crossentropy' if n_classes > 2 else \
        'binary_crossentropy'
full_model.compile(
    optimizer='rmsprop',
    loss=loss,
    metrics=['accuracy'])

print(full_model.summary())

# t_id, t_data, t_gender_label, _ = load_dataset()
# t_data = preprocess_input(t_data)
# val_id, val_data, val_gender_label, _ = load_dataset(prefix='val')
# val_data = preprocess_input(val_data)
#
# # Checkpoint
# filepath = 'model/gender/weights-final_model-improvement-{epoch:02d}-{' \
#            'val_acc:.2f}.hdf5'
# checkpoint = ModelCheckpoint(
#     filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]
#
# # Train the whole model now
# model.fit(
#     t_data,
#     t_gender_label,
#     epochs=5,
#     batch_size=BATCH_SIZE,
#     validation_data=(val_data, val_gender_label),
#     callbacks=callbacks_list,
#     verbose=True)
# model.save_weights('model/gender/weights-final_model_last.hdf5')
