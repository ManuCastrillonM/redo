from keras import applications, optimizers
from keras.layers import Dropout, Flatten, Dense, Input
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import keras, os, sys

import utils
import fineTuningBase

params = utils.yaml_to_dict('config.yml')

def buildModel():
  # build the VGG16 network
  base_model = applications.VGG16(weights='imagenet',
                                  include_top=False,
                                  input_tensor=Input(shape=(params['img_width'], params['img_height'], 3)))

  # freeze all the layers in the network
  for layer in base_model.layers:
    layer.trainable = False

  top_model = base_model.output
  top_model = Flatten(name="Flatten")(top_model)
  top_model = Dense(512, activation='relu')(top_model)
  top_model = Dense(256, activation='relu')(top_model)
  top_model = Dense(6, activation='softmax')(top_model)

  model = Model(inputs=base_model.input, outputs=top_model)

  # model.summary()

  model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

  return model

def runExperiments():
  batch_sizes = [8,16,32]
  epochs = [10,20,30]
  config = 1

  for batch_size in batch_sizes:
    for epoch in epochs:

      print("*************** Test ", config, " Batch size: ", batch_size, " Epochs: ", epoch, "***************")           
      keras.backend.clear_session()

      model = buildModel()
      train_generator,test_generator = fineTuningBase.generateData(batch_size)
      trained_model = fineTuningBase.fineTuneModel(model,train_generator, test_generator, epoch, batch_size)
      metrics = fineTuningBase.getMetrics(trained_model, batch_size)

      config += 1

runExperiments()