# Imports
import glob
import os.path as path
import numpy as np
from PIL import Image
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# Keras Imports
from keras.models import Sequential
from keras.callbacks import History 
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras import optimizers

SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)


def load_images(IMAGE_PATH):
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


def visualize_data(cherry_images,strawberry_images, tomato_images, cherry_labels, strawberry_labels, tomato_labels):
    figure = plt.figure()
    count = 0
    for i in range(cherry_images.shape[0]):
        count += 1
        
        figure.add_subplot(10, cherry_images.shape[0], count)
        plt.imshow(cherry_images[i, :, :])
        plt.axis('off')
        plt.title(cherry_labels[i])

        figure.add_subplot(2, strawberry_images.shape[0], count)
        plt.imshow(strawberry_images[i, :, :])
        plt.axis('off')
        plt.title(strawberry_labels[i])
        
        figure.add_subplot(1, tomato_images.shape[0], count)
        plt.imshow(tomato_images[i, :, :])
        plt.axis('off')
        plt.title(tomato_labels[i])
        
    plt.show()
    
    
    
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


def construct_model(size):
    model = Sequential()
    model.add(Flatten(input_shape=size))
    model.add(Dense(600, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))


    model.compile(loss='categorical_crossentropy', optimizer='sgd', 
               metrics=['accuracy'])
    return model
  

def save_model(model):
  model.save("model.h5")
  print("Model Saved Successfully.")
  
  
def losses(history):
  print(history.history.keys())
  # summarize history for accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  
  
if __name__ == '__main__':
  
  # Load train images
  cherry_images, cherry_labels = load_images('drive/My Drive/Andy_data/Train_data/cherry')
  strawberry_images, strawberry_labels = load_images('drive/My Drive/Andy_data/Train_data/strawberry')
  tomato_images, tomato_labels = load_images('drive/My Drive/Andy_data/Train_data/tomato')
  x_train, y_train = concatenate(strawberry_images, strawberry_labels, cherry_images, cherry_labels, tomato_images, tomato_labels)
  
  # Load valid images
  cherry_images, cherry_labels = load_images('drive/My Drive/Andy_data/valid_data/cherry')
  strawberry_images, strawberry_labels = load_images('drive/My Drive/Andy_data/valid_data/strawberry')
  tomato_images, tomato_labels = load_images('drive/My Drive/Andy_data/valid_data/tomato')
  x_valid, y_valid = concatenate(strawberry_images, strawberry_labels, cherry_images, cherry_labels, tomato_images, tomato_labels)
  
  # Build model
  model = construct_model((32,32,3))
  
  # Training hyperparamters
  EPOCHS = 300
  BATCH_SIZE = 100
  PATIENCE = 10
  
  #setting history as callback
  history = History()
  
  # Train the model
  model.fit(x_train, y_train, epochs=EPOCHS,  verbose=1, validation_data=(x_valid,y_valid),callbacks=[history] )      
  losses(history)
  # Save the model
  save_model(model)
