# coding=utf-8
import os
import time

from keras import applications
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import numpy as np

from src.utilities import (
    load_dataset, get_full_model, data_path, serialize_object,
    load_dataset_from_path)


IMAGE_SIZE = (224, 224)
VGG_16_N_LAST = {
    1: 17,  # trainable starting from 17
    2: 16,
    3: 15,
    4: 13,
    5: 12}


def get_prepared_model(n_classes, n_last, top_model_weights_path):
    full_model = get_full_model(
        IMAGE_SIZE, 'vgg16', n_target_classes=n_classes,
        top_model_weights_path=top_model_weights_path)

    # Finetune the n_last layer
    starting_finetuned_idx = VGG_16_N_LAST[n_last]
    # Freeze the rest of the layers
    for layer in full_model.layers[:starting_finetuned_idx]:
        layer.trainable = False

    loss = 'categorical_crossentropy' if n_classes > 2 else \
        'binary_crossentropy'
    # Train with 1/1000 * default's SGD lr
    full_model.compile(
        optimizer=SGD(lr=0.000001),
        loss=loss,
        metrics=['accuracy'])

    print(full_model.summary())
    return full_model


def finetune(n_last,
             top_model_weights_path, output_dir, train_data_path,
             train_label_path, val_data_path=None, val_label_path=None,
             batch_size=32, epochs=100, initial_epoch=0, checkpoint=False):
    start_time = time.time()
    abs_output_path = data_path(output_dir)
    os.makedirs(abs_output_path, exist_ok=True)

    # Load training data
    t_data = load_dataset_from_path(train_data_path)
    t_label = load_dataset_from_path(train_label_path)
    t_data = applications.vgg16.preprocess_input(t_data)

    n_classes = len(np.unique(t_label))
    assert n_classes >= 2, 'n_classes should be >= 2, got %s' % n_classes
    if n_classes > 2:
        # OHE training label
        t_label = to_categorical(t_label, n_classes)

    # Load validation data (if any)
    if val_data_path:
        val_data = load_dataset_from_path(val_data_path)
        val_label = load_dataset_from_path(val_label_path)
        val_data = applications.vgg16.preprocess_input(val_data)
        if n_classes > 2:
            val_label = to_categorical(val_label, n_classes)
    else:
        # Split training data into train and validation set
        t_data, val_data, t_label, val_label = train_test_split(
            t_data, t_label, random_state=42, stratify=t_label)

    model = get_prepared_model(n_classes, n_last,  top_model_weights_path)

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
        t_data,
        t_label,
        batch_size=batch_size,
        epochs=epochs,
        initial_epoch=initial_epoch,
        validation_data=(val_data, val_label),
        callbacks=callbacks_list,
        verbose=1)

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
            epoch=initial_epoch + epochs,
            val_acc=history.history['val_acc'][-1]))
    model.save_weights(last_filepath)


epochs = 50
batch_size = 128
# Codalab Gender
top_model_weights_path = 'codalab/224_224/model/gender/vgg16/checkpoint/improved-191-0.82.hdf5'
for i in list(range(1, 2)):
    finetune(
        i,
        top_model_weights_path,
        output_dir='codalab/224_224/model/gender/vgg16/ft/%s' % i,
        train_data_path='codalab/224_224/training_data.npz',
        train_label_path='codalab/224_224/training_gender_label.npz',
        val_data_path='codalab/224_224/val_data.npz',
        val_label_path='codalab/224_224/val_gender_label.npz',
        checkpoint=True,
        epochs=epochs,
        batch_size=batch_size,
    )

# Codalab Smile
top_model_weights_path = 'codalab/224_224/model/smile/vgg16/checkpoint/improved-190-0.74.hdf5'
for i in list(range(1, 2)):
    finetune(
        i,
        top_model_weights_path,
        output_dir='codalab/224_224/model/smile/vgg16/ft/%s' % i,
        train_data_path='codalab/224_224/training_data.npz',
        train_label_path='codalab/224_224/training_smile_label.npz',
        val_data_path='codalab/224_224/val_data.npz',
        val_label_path='codalab/224_224/val_smile_label.npz',
        checkpoint=True,
        epochs=epochs,
        batch_size=batch_size,
    )
