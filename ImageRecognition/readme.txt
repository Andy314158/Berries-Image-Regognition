

Note: All python files can be found within the scripts folder, and models within the model folder.


---------training scripts---------

train.py  --->> The final CNN script 
baseline_CNN_train --->> The first implemented CNN script
MLP_train.py --->> The Multi-Layered Perceptron baseline script

To run train.py, MLP_train.py, or baseline_CNN_train.py the follwoing lines must be changed to the appropriate directories...

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


-------------models------------------

model.h5 --->> The final CNN model
MLP.5 --->> The MLP model
baseline_CNN --->> The first CNN model

-------------test.py-----------------

To run the test file, you will need to import keras on the ecs machines as follows....

"pip install keras --user"

within the code ensure that the follwoing line is set to the model you want to test...

"model = load_model('model/model.h5')"

Then run test.py (python test.py --test_data_dir your_test_directory)

note 1: you will need to make sure tensorflow is 1.12 as the given models were trained using google colab.

note 2: There is the given test file with 15 images, if you run python test.py --test_data_dir test it should run
