from keras import applications, optimizers
from keras.layers import Dropout, Flatten, Dense, Input
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import keras, os
import utils

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
  top_model = Dense(256, activation='relu')(top_model)
  top_model = Dense(128, activation='relu')(top_model)
  top_model = Dense(6, activation='softmax')(top_model)

  model = Model(inputs=base_model.input, outputs=top_model)

  # model.summary()

  model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

  return model


def generateData(batch_size):
  train_datagen = ImageDataGenerator(rescale=1. / 255)
  validation_datagen = ImageDataGenerator(rescale=1. / 255)

  train_generator = train_datagen.flow_from_directory(
    params['train_data_dir'],
    target_size=(params['img_width'], params['img_height']),
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical')

  validation_generator = validation_datagen.flow_from_directory(
    params['validation_data_dir'],
    target_size=(params['img_width'], params['img_height']),
    batch_size=batch_size,
    class_mode='categorical')

  return [train_generator,validation_generator]


def fineTuneModel(model, train_generator, validation_generator, epochs, batch_size):
  training_samples = 0
  validation_samples = 0

  for path, dirs, files in os.walk(params['train_data_dir']):
    for filename in files:
      training_samples += 1

  for path, dirs, files in os.walk(params['validation_data_dir']):
    for filename in files:
      validation_samples += 1

  model.fit_generator(
    train_generator,
    steps_per_epoch=training_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_samples // batch_size)

  return model


def getMetrics(model, batch_size):
  test_samples = 0

  for path, dirs, files in os.walk(params['test_data_dir']):
    for filename in files:
      test_samples += 1

  test_datagen = ImageDataGenerator(rescale=1. /255)

  test_generator = test_datagen.flow_from_directory(
    params['test_data_dir'],
    target_size=(params['img_width'],params['img_height']),
    shuffle=False,
    batch_size=1,
    class_mode='categorical')

  predicted_results = model.predict_generator(test_generator,steps = test_samples)
  predicted_results = np.argmax(predicted_results, axis=1)
  targets = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

  # confusion matrix
  print('CONFUSION MATRIX:')
  print(confusion_matrix(test_generator.classes, predicted_results))

  # classification report
  print('CLASSIFICATION REPORT:')
  print(classification_report(test_generator.classes, predicted_results, target_names=targets))


if __name__ == '__main__':

  model = buildModel()
  train_generator,test_generator = generateData(params['batch_size'])
  trained_model = fineTuneModel(model,train_generator, test_generator, params['epochs'], params['batch_size'])
  metrics = getMetrics(trained_model, params['batch_size'])