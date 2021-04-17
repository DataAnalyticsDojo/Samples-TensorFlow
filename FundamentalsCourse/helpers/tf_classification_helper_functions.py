# Helper Functions for use in our machine learning experiments

#Import sklearn before tensorflow
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import zipfile
import sys
import os
import pathlib
import random
import urllib.request
import datetime

def show_environment():    
    """
    Shows the system environment information including Python version, TensorFlow version, 
    GPU info, and SciKit Learn info.
    Args:

    Returns:

    """
    print("Setting Up Libraries")
    print(f"\nPython Version: {sys.version}")
    print(f"\n\nTensorflow Version: {tf.__version__}")
    print(f"\n\nSciKit Learn Versions:")
    sklearn.show_versions()

def show_gpu_info():
    """
    Shows information about the detected GPUs on the system.
    Args:

    Returns:

    """    
    print(f"\n\nGPU Device Name: {tf.test.gpu_device_name()}, Physical GPU Device List: {tf.config.list_physical_devices('GPU')}")
    # See if we have a GPU
    if tf.test.gpu_device_name(): 
        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
    else:
        print("No GPU detected or you need to install the GPU dependencies according to the docs at: https://www.tensorflow.org/install/gpu.")        


def walk_directory(directory):
    """
    Walk through the data directory and list number of files. This will start at the directory passed in
    and print the number of files and directories
    Ex:
    There are 2 directories and 0 files in c:/temp/data/
    There are 10 directories and 0 files in c:/temp/data/test
    There are 0 directories and 250 files in c:/temp/data/test/class1

    Args:
        directory: The directory to walk through.
    Returns:

    """
    for dirpath, dirnames, filenames in os.walk(directory):
        print(f"There are {len(dirnames)} directories and {len(filenames)} files in {dirpath}")    


def get_class_names_from_directory(directory):
    """
    Create an array of class names by looking in a directory and putting all the file/directory names
    in that directory in an alphabetically sorted array.

    Args:
        directory: The directory to get the class names from.
    Returns:
        class_names: the names of the classes
    """    
    dir_path = pathlib.Path(directory)    
    # Create a list of class names from the subdirectories
    class_names = np.array(sorted([item.name for item in dir_path.glob("*")]))
    # Remove extra files that may have been added if this file was zipped from a mac
    class_names = np.delete(class_names, np.where(class_names == ".DS_Store"))
    return class_names


def view_random_images_from_directory(directory, class_names, class_to_show=None, num_images=4, figsize=(10,10)):
    """
    From the directory passed in, randomly select images from subfolders in that directory with the 
    class_names specified. Up to num_images with be shown.

    Args:
        directory: The directory to find the images in. Must contain subdirectories 
        which have the names specified in class_names.
        class_names: The names of the image classes.
        class_to_show: If you want to only show images from a single class, pass in the class name here.
        num_images: The number of images to show.
        figsize: The figure size of the matplotlib image.
    Returns:
    """    
    plt.figure(figsize=figsize)
    for i in range(num_images):
        # If a class was specified then show only that class
        if class_to_show:
            selected_class = class_to_show
        # Otherwise randomly pick classes
        else:
            selected_class = random.choice(class_names)

        # Setup the target directory (we'll view images from here)
        target_directory = directory + "/" + selected_class

        plt.subplot(1, num_images, i+1)

        #Get a random image path
        random_image = random.sample(os.listdir(target_directory), 1)
        
        # Read in the image and plot it using matplotlib
        img = mpimg.imread(target_directory + "/" + random_image[0])
        plt.imshow(img)
        plt.title(f"{selected_class}\nShape: {img.shape}\nFile: {random_image[0]}")
        plt.axis("off")

        #print(f"Image Shape: {img.shape}")
        #return img


def create_tensorboard_callback(dir_name, experiment_name):
    """
    Create a TensorBoard callback for being able to track our experiments using TensorBoard.
    tensorboard_callback = create_tensorboard_callback("c:/temp/data/tensorboard", "resnet_1")
    Add into model.fit(callbacks=[tensorboard_callback])
    Args:
        dir_name: The directory the store the TensorBoard log files into.
        experiment_name: The name of the experiment
    Returns:
        The callback to be added to model.fit.
    """    
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir = log_dir,        
    )
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


def plot_loss_curves(history):
    """
    Plots separate loss curves for training and validation metrics.
    If training loss is decreasing, but validation loss is increasing, then it shows our model is overfitting.
    Ideally we'd like the curves to line up and follow each other
    If the model is overfitting (learning the training data too well) it will get great results on the 
    training data, but it is failing to generalize well to unseen data and it performs poorly on the test data.

    When a models validation loss starts to increase it's likely that the model is overfitting. 
    This means it's learning the patterns in the training dataset too well and thus the model's 
    ability to generalize unseen data will be diminished.

    Args:
        history: The history returned from a model.fit call whose metric was "accuracy"
    Returns:        
    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    epochs = range(len(history.history["loss"])) # how many epochs did we train for

    # Plot loss
    plt.plot(epochs, loss, label="training_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("loss")
    plt.xlabel("epochs")
    plt.legend()

    # Plot Accuracy
    plt.figure() #Start a new plot figure
    plt.plot(epochs, accuracy, label="training_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.title("accuracy")
    plt.xlabel("epochs")
    plt.legend()


def load_and_prep_image(url, img_shape=224):
    """
    Reads an image from url, turns it into a tensor, and reshapes it to (img_shape, img_shape, colour_channels).
    Args:
        url: The URL to load the image from
        img_shape: The shape of the image (the same number is used for width and height)
    Returns:   
        The image that was loaded from the url and resized.
    """
    # Read in the image
    #img = tf.io.read_file(filename)
    image_request = urllib.request.urlopen(url)
    img = image_request.read()
    # Decode the read file into a tensor
    img = tf.image.decode_image(img)
    # Resize the image
    img = tf.image.resize(img, size=[img_shape, img_shape])
    # Rescale the image (get all values between 0 and 1)
    img = img/255.
    return img

def predict_and_plot(model, url, class_names):
    """
    Imports an image located at url, makes a prediction with model, and plots the image with the predicted 
    class as the title.
    Args:
        model: The model to make the prediction with
        url: The URL to load the image from
        class_name: A list of the classes that the model predicts
    Returns:   
        The image that was loaded from the url and resized.
    """
    #Import the target images and preprocess it
    img = load_and_prep_image(url)

    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))
    print(pred)

    # Get the predicted class
    if len(class_names) <= 2:
        # For binary:
        pred_class = class_names[int(tf.round(pred))]
    else:
        # For multiclass:
        pred_class = class_names[np.argmax(pred)]

    # Plot the image and predicted class
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)


if __name__ == "__main__":
    print("Tensorflow experiment helper functions. Import these into your notebook.")