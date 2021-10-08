print("starting")
import time
import random
import keras
import numpy as np
from matplotlib import pyplot as plt

from keras.layers import BatchNormalization
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, InputLayer
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.models import Model
from keras.applications.vgg16 import VGG16
import json

if __name__ == '__main__':
    start = time.time()
    random.seed(8675309)
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen =ImageDataGenerator(rescale=1./255)

#dataset loading
    train_gen = train_datagen.flow_from_directory(
            "dataset/Training/",
            target_size=(200,200),
            color_mode='rgb',
            batch_size=32,
            classes=['1','2','3','4','5','6','7','8'],
            class_mode='categorical'
        )   
    
    val_gen = val_datagen.flow_from_directory(
            "dataset/Testing/",
            target_size=(200,200),
            color_mode='rgb',
            batch_size=32,
            classes=['1','2','3','4','5','6','7','8'],
            class_mode='categorical'
        )

    # pre-process the data for Keras
    vgg_model = keras.applications.VGG16(weights='imagenet',
                               include_top=False,
                               input_shape=(200, 200, 3))    
    layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

    # Getting output tensor of the last VGG layer that we want to include
    x = layer_dict['block2_pool'].output

    # Stacking a new simple convolutional network on top of it
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(8, activation='softmax')(x)
    
    model = Model(vgg_model.input,x)

    for layer in model.layers[:5]:
        layer.trainable = False

    # from keras import optimizers
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    model.summary()
   
    jsonmodel=model.to_json()
    with open('detect.json','w') as fp:
        json.dump(jsonmodel,fp)
  
    epochs = 10
    checkpointer = ModelCheckpoint(filepath='detect.hdf5',
                                   verbose=1)
    history = model.fit_generator(train_gen, epochs=epochs, steps_per_epoch=train_gen.n // 32 + 1, callbacks=[checkpointer],
                                  verbose=1, validation_data=val_gen, validation_steps=val_gen.n // 32)
    print(history.history.keys())
    
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss')
    plt.show()
   
    end = time.time()
    print(end - start)