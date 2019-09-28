from imgaug import augmenters as iaa
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
import imgaug as ia
import numpy as np
import pandas as pd
import cv2, os, csv, shutil
import utils

# used to copy files according to each fold
def copy_images(dataframe, directory):
  destination_directory = './dataset/{}'.format(directory)
  print('copying {} files to {}...'.format(directory, destination_directory))

  # remove all files from previous fold
  if os.path.exists(destination_directory):
    shutil.rmtree(destination_directory)

  # create folder for files from this fold
  if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

  # create subfolders for each class
  for image_class in set(list(dataframe['class'])):
    if not os.path.exists(destination_directory + '/' + image_class):
      os.makedirs(destination_directory + '/' + image_class)

  # copy files for this fold from a directory holding all the files
  for i, row in dataframe.iterrows():
    try:
      # this is the path to all of your images kept together in a separate folder
      path_from = '{}/{}/{}'.format(params['original_data_dir'], row['class'],row['image'])
      path_to = "{}/{}".format(destination_directory, row['class'])

      # move from folder keeping all files to training, test, or validation folder (the "directory" argument)
      shutil.copy(path_from, path_to)
    except Exception as e:
      print("Error when copying {}: {}".format(row['image'], str(e)))


def createStratifiedData():
  # create dataframe with the images filenames
  dataset_files = open('./dataset.csv', 'w+')
  writer = csv.writer(dataset_files)

  writer.writerow(['image','class'])
  for path, dirs, files in os.walk(params['original_data_dir']):
    for filename in files:
      if( filename != '.DS_Store'):
        writer.writerow([filename, os.path.basename(path)])

  # dataframe containing the filenames of the images and the classes
  df = pd.read_csv('./dataset.csv')
  df_y = df['class']
  df_x = df['image']

  skf = StratifiedShuffleSplit(n_splits = 1, test_size=0.2)

  for train_index, test_index in skf.split(df_x, df_y):
    x_train, x_test = df_x[train_index], df_x[test_index]
    y_train, y_test = df_y[train_index], df_y[test_index]

    train = pd.concat([x_train, y_train], axis=1)
    test = pd.concat([x_test, y_test], axis = 1)
    # take 20% of the training data from this fold for validation during training
    validation = test.sample(frac = 0.5)

    # make sure validation data does not include training data
    train = train[~train['image'].isin(list(validation['image']))]

    # copy the images according to the fold
    copy_images(train, 'train')
    copy_images(validation, 'validation')
    copy_images(test, 'test')


def augmentData():
  # create numpy array with images paths
  img_paths = np.array([])
  for path, dirs, files in os.walk(params['train_data_dir']):
    for filename in files:
      if(filename != '.DS_Store'):
        path_name = '{}/{}/{}'.format(params['train_data_dir'],os.path.basename(path),filename)
        img_paths = np.append(img_paths, path_name)

  # convert images to numpy arrays
  paths_number, = img_paths.shape

  images = np.zeros(shape=(paths_number,params['img_height'],params['img_width'],3),dtype='uint8')

  for idx, img_path in enumerate(img_paths):
    print('Creating array from ', img_path)
    img = cv2.imread(img_path, 1)
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images[idx] = im_rgb

  # aug configurations
  seq = iaa.Sequential([
    iaa.Affine(scale=(0.8, 1.1), mode=['edge']), # Scale images to a value between 80% and 150%
    iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, mode=['edge']), # Translate images by -10 to +10% on x- and y-axis independently
    iaa.Affine(rotate=(-25, 25), mode=['edge']), # Rotate images by -45 to 45 degrees
    iaa.Fliplr(0.5), # Flip 50% of all images horizontally
    iaa.Flipud(0.5) # Flip 50% of all images vertically
  ], random_order=True)

  # create augmented images
  images_aug = seq.augment_images(images)

  # save augmented images
  for i in range(len(img_paths)):
    img_dest = '{}_AUG.jpg'.format(img_paths[i][:-4])
    Image.fromarray(images_aug[i]).save(img_dest)
    print('Saving ',img_dest)

if __name__ == '__main__':
  params = utils.yaml_to_dict('config.yml')

  createStratifiedData()
  # augmentData()