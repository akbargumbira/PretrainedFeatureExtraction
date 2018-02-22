# coding=utf-8
import os
import time

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

from src.utilities import data_path, serialize_object, get_top_model, \
    load_bottleneck_features


def train(training_features_path, training_label_path, output_dir,
          previous_model=None, epochs=100, initial_epoch=0, checkpoint=False,
          val_features_path=None, val_label_path=None):
    start_time = time.time()
    abs_output_path = data_path(output_dir)
    os.makedirs(abs_output_path, exist_ok=True)

    training_data, training_label = load_bottleneck_features(
        training_features_path, training_label_path)

    n_classes = len(np.unique(training_label))
    assert n_classes >= 2, 'n_classes should be >= 2, got %s' % n_classes
    if n_classes >= 2:
        # OHE training label
        training_label = to_categorical(training_label, n_classes)

    if val_features_path:
        val_data, val_label = load_bottleneck_features(
            val_features_path, val_label_path)
        train_data, train_label = training_data, training_label
        if n_classes >= 2:
            val_label = to_categorical(val_label, n_classes)
    else:
        # Split into train and validation set
        train_data, val_data, train_label, val_label = train_test_split(
            training_data, training_label, random_state=42, stratify=training_label)

    # Prepare the model
    model = get_top_model(train_data.shape[1:], n_classes)
    # Use saved model weights if specified
    if previous_model:
        model.load_weights(previous_model)

    loss = 'categorical_crossentropy' if n_classes >= 2 else \
        'binary_crossentropy'
    model.compile(optimizer='adadelta',
                  loss=loss,
                  metrics=['accuracy'])

    # Prepare callbacks
    callbacks_list = []

    # 1. ModelCheckpoint
    if checkpoint:
        checkpoint_basedir = os.path.join(abs_output_path, 'checkpoint')
        os.makedirs(checkpoint_basedir, exist_ok=True)
        checkpoint_filepath = os.path.join(
            checkpoint_basedir, 'improved-{epoch:03d}-{val_acc:.2f}.hdf5')
        checkpoint = ModelCheckpoint(
            checkpoint_filepath,
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            mode='max')
        callbacks_list.append(checkpoint)

    history = model.fit(
        train_data,
        train_label,
        epochs=epochs,
        initial_epoch=initial_epoch,
        validation_data=(val_data, val_label),
        callbacks=callbacks_list,
        verbose=2)

    # Dump the run time
    run_time = int(time.time() - start_time)
    serialize_object(run_time, os.path.join(abs_output_path, 'time.pkl'))

    # Dump the history
    serialize_object(
        history.history,
        os.path.join(abs_output_path, 'hist.pkl'))
    # Dump the last weights
    last_filepath = os.path.join(
        abs_output_path,
        'weights-last-{epoch:03d}-{val_acc:.2f}.hdf5'.format(
            epoch=initial_epoch+epochs,
            val_acc=history.history['val_acc'][-1]))
    model.save_weights(last_filepath)


# --------------------------------------------------------------------
# # Codalab Gender
# models = ['vgg16', 'inceptionv3', 'resnet50']
# for model in models:
#     print('Training with %s....' % model)
#     train(
#         training_features_path='codalab/224_224/features_training_%s.npz' %
#                                model,
#         training_label_path='codalab/224_224/training_gender_label.npz',
#         output_dir='codalab/224_224/model/gender/%s/' % model,
#         checkpoint=True,
#         epochs=200,
#         val_features_path='codalab/224_224/features_val_%s.npz' % model,
#         val_label_path='codalab/224_224/val_gender_label.npz'
#     )

# # Codalab Smile
# models = ['vgg16', 'inceptionv3', 'resnet50']
# for model in models:
#     print('Training with %s....' % model)
#     train(
#         training_features_path='codalab/224_224/features_training_%s.npz' %
#                                model,
#         training_label_path='codalab/224_224/training_smile_label.npz',
#         output_dir='codalab/224_224/model/smile/%s/' % model,
#         epochs=200,
#         checkpoint=True,
#         val_features_path='codalab/224_224/features_val_%s.npz' % model,
#         val_label_path='codalab/224_224/val_smile_label.npz'
#     )
# --------------------------------------------------------------------
# # Kaggle Dog and Cat
# models = ['vgg16', 'inceptionv3', 'resnet50']
# for model in models:
#     print('Training with %s....' % model)
#     train(
#         training_features_path='kaggle_dog_cat/224_224/features_training_%s'
#                                '.npz' % model,
#         training_label_path='kaggle_dog_cat/224_224/training_label.npz',
#         output_dir='kaggle_dog_cat/224_224/model/%s/' % model,
#         checkpoint=True,
#         epochs=200)

# ----------------------------------------------------------------------
# # UIUC
# models = ['vgg16', 'inceptionv3', 'resnet50']
# for model in models:
#     print('Training with %s....' % model)
#     train(
#         training_features_path='uiuc/224_224/features_training_%s.npz' %
#                                model,
#         training_label_path='uiuc/224_224/training_label.npz',
#         output_dir='uiuc/224_224/model/%s/' % model,
#         checkpoint=True,
#         epochs=200)



# --------------------------------------------------------------------------
# Codalab SMILE
# last_n = [2, 3, 4]
last_n = [1]
for n in last_n:
    train(
        training_features_path='codalab/224_224/cnn_vgg_last_%s_training.npz' % n,
        training_label_path='codalab/224_224/training_smile_label.npz',
        output_dir='codalab/224_224/model/smile/vgg16/early/%s' % n,
        checkpoint=True,
        epochs=200,
        val_features_path='codalab/224_224/cnn_vgg_last_%s_val.npz' % n,
        val_label_path='codalab/224_224/val_smile_label.npz'
    )


# ---------------------------------------------------------------------------
# Codalab GENDER
# last_n = [4]
last_n = [1,2,3,4]
for n in last_n:
    train(
        training_features_path='codalab/224_224/cnn_vgg_last_%s_training.npz' % n,
        training_label_path='codalab/224_224/training_gender_label.npz',
        output_dir='codalab/224_224/model/gender/vgg16/early/%s' % n,
        checkpoint=True,
        epochs=300,
        val_features_path='codalab/224_224/cnn_vgg_last_%s_val.npz' % n,
        val_label_path='codalab/224_224/val_gender_label.npz'
    )
