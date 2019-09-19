# Imports
import glob
import os.path as path
import numpy as np
from PIL import Image
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import LabelEncoder


# Keras Imports
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D
from keras.utils import np_utils
from keras import regularizers
from keras import optimizers
from tensorflow.python.keras.models import load_model

# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)



def parse_args():
    """
    Pass arguments via command line
    :return: args: parsed args
    """
    # Parse the arguments, please do not change
    args = argparse.ArgumentParser()
    args.add_argument("--test_data_dir", default = "data/test_data",
                      help = "path to test_data_dir")
    args = vars(args.parse_args())
    return args
  

def load_images(IMAGE_PATH):
  image_size=(32,32)
  #load file paths
  file_paths = glob.glob(path.join(IMAGE_PATH, '*.jpg'))
  # Load the images
  images = np.array([np.array((Image.open(fname).convert("RGB")).resize([32, 32])) for fname in file_paths])
  # Get image size
  image_size = np.asarray([images.shape[1], images.shape[2], images.shape[3]])
  print(image_size)
  #scale the images
  images = images.astype('float32') / 255
  #retrieve image labels
  labels = retreive_image_labels(images, file_paths)
  #return set with corresponding labels
  return images, labels
  



def retreive_image_labels(images, file_paths):
  n_images = images.shape[0]
  labels= np.empty(n_images, dtype=int)
  for i in range(n_images):
    #0 for cherry, 1 for strawberry and 2 for tomato
      filename = path.basename(file_paths[i])[0]
      if(filename[0]=='c'):
        labels[i]=0
      elif(filename[0]=='s'):
        labels[i]=1
      elif(filename[0]=='t'):
        labels[i]=2
        
  return labels


def concatenate(strawberry_images, strawberry_labels, cherry_images, cherry_labels, tomato_images, tomato_labels):
  #now combine files to create one main data list
  images = np.concatenate((cherry_images, strawberry_images, tomato_images))
  labels = np.concatenate((cherry_labels, strawberry_labels, tomato_labels))
  # encode class values as integers
  encoder = LabelEncoder()
  encoder.fit(labels)
  encoded_Y = encoder.transform(labels)
  # convert integers to dummy variables (i.e. one hot encoded)
  dummy_y = np_utils.to_categorical(encoded_Y)
  
  return images, dummy_y


def evaluate(X_test, y_test):
    # Load Model
    model = load_model('../model/model.h5')
    
    score = model.evaluate(X_test, y_test, verbose=0)
    accuracy = 100*score[1]
    
    
    return accuracy
  
  

if __name__ == '__main__':
    # Parse the arguments
    args = parse_args()

    # Test folder
    test_data_dir = args["test_data_dir"]
    
    
    # Load test images
    cherry_images, cherry_labels = load_images(test_data_dir+ '/cherry')
    strawberry_images, strawberry_labels = load_images(test_data_dir+ '/strawberry')
    tomato_images, tomato_labels = load_images(test_data_dir+ '/tomato')
    
    x_test, y_test = concatenate(strawberry_images, strawberry_labels, cherry_images, cherry_labels, tomato_images, tomato_labels)

    # Image size
    image_size = (32, 32, 3)

    # Evaluation
    accuracy = evaluate(x_test,y_test)
   
    
    print("Accuracy: " + str(accuracy)) 
