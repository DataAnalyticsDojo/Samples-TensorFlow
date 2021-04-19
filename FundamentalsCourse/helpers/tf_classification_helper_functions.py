# Helper Functions for use in our machine learning experiments

#Import sklearn before tensorflow
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
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
import itertools


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



def view_random_images_from_tf_dataset(dataset, class_names, batches=-1, num_images=4, figsize=(10,10), data_augmentation=None):
    """
    From the tensorflow dataset, randomly select images and show them. Up to num_images with be shown.

    Args:
        dataset: The TensorFlow DataSet to get the images from
        class_names: The names of the image classes.
        batches: The number of batches to take from the dataset. Usually you'll have batches containing 32 images each. 
                 -1 will pick from all batches
        num_images: The number of images to show.
        figsize: The figure size of the matplotlib image.
        data_augmentation: An optional Keras Sequential Layer that can be used to visualize the effects of your data augmentation layer
                           Example follows:
                            data_augmentation = tf.keras.Sequential([
                                tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
                                tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
                                tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
                                tf.keras.layers.experimental.preprocessing.RandomHeight(0.2),
                                tf.keras.layers.experimental.preprocessing.RandomWidth(0.2),
                                #tf.keras.layers.experimental.preprocessing.Rescale(1./255) # Keep for models like Resnet50V2, but EfficientNets have rescaling built in
                            ], name="data_augmentation")        
    Returns:
    """        
    plt.figure(figsize=figsize)
    # Get a sample of train data batch
    # image_dataset_from_directory returns a BatchDataset. We have to call take in a for loop with the
    # number of batches to take
    images = None
    for images_batch, labels_batch in dataset.take(batches):
        #len(images), len(labels)
        #print("Took dataset")
        if images is None:
            images = images_batch
            labels = labels_batch
        else:
            images = tf.concat([images, images_batch], 0)
            labels = tf.concat([labels, labels_batch], 0)
        None # No-op
        
    print (f"Number of Images To Pick From: {len(images)}")

    # Loop and get up to num_images to display
    for i in range(num_images):    
        # Return a number between 0 and the length of our images array (both included)
        random_number = random.randint(0, len(images)-1)
        label_index = np.argmax(labels[random_number])
        print(f"Showing image number {random_number}")
        image_shape = images[random_number].shape
        image = images[random_number]

        # Augment the image
        if data_augmentation:
            # Expand the dimensions to avoid error:
            # Input 0 of layer data_augmentation is incompatible with the layer: expected ndim=4, found ndim=3. Full shape received: (224, 224, 3)
            image = tf.expand_dims(image, axis=0)
            image = data_augmentation(image)
            image = tf.squeeze(image)

        # Either display as an int from 0 to 255, or as a float between 0 and 1. If its a float with 
        # values > 1 then we cast to an int
        if image.dtype == tf.float32 and tf.reduce_max(image) > 1:
            #print(image.dtype)
            #print("TensorFlow Image is a float datatype, but has values > 1. Casting to an int32 datatype for matplotlib.")
            image = tf.cast(image, tf.int32)
        #image = tf.cast(image/255., tf.float32)
        plt.subplot(1, num_images, i+1)            
        plt.imshow(image)
        plt.title(f"{class_names[label_index]}\nShape: {image_shape}")
        plt.axis(False)



def create_tensorboard_callback(dir_name, experiment_name):
    """
    Create a TensorBoard callback for being able to track our experiments using TensorBoard.
    tensorboard_callback = create_tensorboard_callback("c:/temp/data/tensorboard", "resnet_1")
    Add into model.fit(callbacks=[tensorboard_callback])
    The files will be stored in a directory with pattern: dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    Args:
        dir_name: The base directory the store the TensorBoard log files into.
        experiment_name: The name of the experiment or model. This will be appended to dir_name.
    Returns:
        The callback to be added to model.fit.
    """    
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir = log_dir     
    )
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


def create_model_checkpoint_callback(dir_name, experiment_name):
    """
    Create a Checkpoint callback to save the Keras model or model weights at some frequency
    The ModelCheckpoint callback periodically saves our model (the full model or just the weights) during training. 
    This is useeful so we can come and start where we left off.
    Add into model.fit(callbacks=[checkpoint_callback])
    The files will be stored in a directory with pattern: dir_name + "/" + experiment_name
    Args:
        dir_name: The base directory the store the checkpoints to
        experiment_name: The name of the experiment or model. This will be appended to dir_name.
    Returns:
        The callback to be added to model.fit.
    """    
    checkpoint_path = dir_name + "/" + experiment_name + "/" + experiment_name
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True, # If false the whole model is saved, and it takes more time to save
        save_best_only=True, # Only save the model weights which achieved the highest accuracy
        save_freq="epoch", # Save every epoch
        monitor="val_accuracy",
        verbose=1
    )
    return checkpoint_callback



def plot_loss_curves(history, figsize=(10,10)):
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
        figsize: The figure size of the matplotlib image.
    Returns:        
    """
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(len(history.history["loss"])) # how many epochs did we train for

    # Make Plots
    plt.figure(figsize=figsize)

    # Plot Accuracy
    plt.subplot(2, 1, 1)
    plt.plot(accuracy, label="Training Accuracy")
    plt.plot(val_accuracy, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.legend(loc="lower right")  

    # Plot Loss
    plt.subplot(2, 1, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    #Plot a vertical line where the fine tuning started
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.legend(loc="upper right")  



def compare_loss_curves(history_1, history_2, figsize=(10,10)):
    """
    Plots loss curves and compares two TensorFlow History objects.

    Args:
        history_1: The history returned from a model.fit call whose metric was "accuracy"
        history_2: The history returned from a 2nd model.fit call whose metric was "accuracy"
        figsize: The figure size of the matplotlib image.
    Returns:    
    """
    # Get 1st history measurements
    acc = history_1.history["accuracy"]
    loss = history_1.history["loss"]

    val_acc = history_1.history["val_accuracy"]
    val_loss = history_1.history["val_loss"]    

    # Combine 2nd history with the first history
    total_acc = acc + history_2.history["accuracy"]
    total_loss = loss + history_2.history["loss"]

    total_val_acc = val_acc + history_2.history["val_accuracy"]
    total_val_loss = val_loss + history_2.history["val_loss"]    

    history_1_epochs = len(history_1.history["loss"]) # how many epochs did the first history train for

    # Make Plots
    plt.figure(figsize=figsize)

    # Plot Accuracy
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label="Training Accuracy")
    plt.plot(total_val_acc, label="Validation Accuracy")
    plt.plot([history_1_epochs-1, history_1_epochs-1], plt.ylim(), label="End of 1st History")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.legend(loc="lower right")  

    # Plot Loss
    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label="Training Loss")
    plt.plot(total_val_loss, label="Validation Loss")
    #Plot a vertical line where the fine tuning started
    plt.plot([history_1_epochs-1, history_1_epochs-1], plt.ylim(), label="End of 1st History")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.legend(loc="upper right")  



def load_and_prep_image(url, image_shape=224, normalize_image=False):
    """
    Reads an image from url, turns it into a tensor, and reshapes it to (image_shape, image_shape, colour_channels).
    Args:
        url: The URL to load the image from
        image_shape: The shape of the image (the same number is used for width and height)
        normalize_image: True if you want the image normalized/rescaled so all values are between 0 and 1. (divides the values by 255)
    Returns:   
        The image that was loaded from the url and resized.
    """
    # Read in the image
    #img = tf.io.read_file(filename)
    image_request = urllib.request.urlopen(url)
    raw_img = image_request.read()
    # Decode the read file into a tensor
    #Png usually has 4 channels (RGBA), jpg has 3 (RGB). We convert to a common 3 channels
    #expand_animations will convert animated gifs to only have 1 frame
    raw_img = tf.image.decode_image(raw_img, channels=3, expand_animations=False)
    # Resize the image
    prepped_img = tf.image.resize(raw_img, size=[image_shape, image_shape])
    # Rescale the image (get all values between 0 and 1)
    if normalize_image:
        prepped_img = prepped_img/255.

    return raw_img, prepped_img


def predict_and_plot_image(model, class_names, url, image_shape=224, normalize_image=False):
    """
    Imports an image located at url, makes a prediction with model, and plots the image with the predicted 
    class as the title.
    Args:
        model: The model to make the prediction with
        class_names: A list of the classes that the model predicts
        url: The URL to load the image from
        image_shape: The shape of the image (the same number is used for width and height)
        normalize_image: True if you want the image normalized/rescaled so all values are between 0 and 1.
    Returns:   
        The image that was loaded from the url and resized.
    """
    #Import the target images and preprocess it
    raw_img, prepped_img = load_and_prep_image(url=url, image_shape=image_shape, normalize_image=normalize_image)

    # Make a prediction
    pred = model.predict(tf.expand_dims(prepped_img, axis=0))
    #print(pred)

    # Get the predicted class
    if len(class_names) <= 2:
        # For binary:
        pred_class = class_names[int(tf.round(pred))]
        probability_pct = pred
    else:
        # For multiclass:
        pred_class = class_names[np.argmax(pred)]
        probability_pct = tf.reduce_max(pred)

    # Plot the image and predicted class
    plt.figure()
    plt.imshow(raw_img)
    plt.title(f"Prediction: {pred_class} ({probability_pct:.2%})")
    plt.axis(False)



# Note: The following confusion matrix code is a remix of Scikit-Learn's 
# plot_confusion_matrix function - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
# Our function needs a different name to sklearn's plot_confusion_matrix
def make_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
    """
    Makes a labelled confusion matrix comparing predictions and ground truth labels.

    If class_names is passed, confusion matrix will be labelled, if not, integer class values
    will be used.

    Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    class_names: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).

    Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

    Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                            y_pred=y_preds, # predicted labels
                            class_names=class_names, # array of class label names
                            figsize=(15, 15),
                            text_size=10)
    """  
    # Create the confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
    n_classes = cm.shape[0] # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    # Are there a list of classes?
    if class_names:
        labels = class_names
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
            xlabel="Predicted label",
            ylabel="True label",
            xticks=np.arange(n_classes), # create enough axis slots for each class
            yticks=np.arange(n_classes), 
            xticklabels=labels, # axes will labeled with class names (if they exist) or ints
            yticklabels=labels)

    #Set label font sizes
    ax.title.set_fontsize(text_size)
    ax.xaxis.get_label().set_fontsize(text_size)
    ax.yaxis.get_label().set_fontsize(text_size)

    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Change the xaxis to plot x-labels vertically
    plt.xticks(rotation=70, fontsize=text_size)
    plt.yticks(fontsize=text_size)

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                    horizontalalignment="center",
                    color="white" if cm[i, j] > threshold else "black",
                    size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                    horizontalalignment="center",
                    color="white" if cm[i, j] > threshold else "black",
                    size=text_size)

    # Save the figure to the current working directory
    if savefig:
        fig.savefig("confusion_matrix.png")



def plot_classification_report(y_true, y_pred, class_names, figsize=(10, 10), text_size=15, metric="f1-score"):
    """
    Visualize a classification report for our predictions.
    Scikit learn has a helpful function for acquiring many different classification metrics per class (eg precision, recall, and F1) called classification report. This method will plot those.

    Here are various classification model evaluation methods:
        - Accuracy - default metric for classification. Not the best for imbalanced classes.
        - Precision - higher precision leads to less false positives
        - Recall - higher recall leads to less false negatives
        - F1-Score - combination of precision and recall, usually a good overall metric for a classification model
        - Confusion matrix - When comparing predictions to truth labels to see where model gets confused. Can be hard to use with large numbers of classes.    
    Args:
        y_true: A list of the ground truth labels
        y_pred: A list of the predicted labels
        class_names: A list of the classes that the model predicts
        figsize: Size of output figure (default=(10, 10)).
        text_size: Size of output figure text (default=15).        
        metric: One of precision, recall, f1-score. Defaults to f1-score
    Returns:
        
    """      
    # Get a dictionary of the classification report
    classification_report_dict = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
    #print(classification_report_dict)

    # Plot all of the classes F1 Scores
    # Create empty dictionary
    class_metrics = {}
    # Loop through the classification report dictionary items
    for k, v in classification_report_dict.items():
        # stop once we get to the end of the classification report keys that are for the overall totals
        if k == "accuracy" or \
           k == "macro avg" or \
           k == "weighted avg": 
            break
        else:
            # Add class names and F1 scores to new dictionary            
            class_metrics[class_names[int(k)]] = v[metric]
    
    #print(class_metrics)
    
    # Turn F1 scores into dataframe for visualization
    metrics = pd.DataFrame({
        "class_names": list(class_metrics.keys()),
        metric: list(class_metrics.values())
    }).sort_values(metric, ascending=False)
    #print(metrics)

    fig, ax = plt.subplots(figsize=figsize)
    scores = ax.barh(
        range(len(metrics)), 
        metrics[metric].values,
        color="#ccc"
    )

    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics["class_names"], fontsize=text_size)
    
    ax.set_xlabel(metric, fontsize=text_size)
    ax.set_title(f"{metric} by Class", fontsize=text_size)
    ax.invert_yaxis() # reverse the order of the y axis so highest score in on top

    #Set label font sizes
    plt.xticks(fontsize=text_size)

    #Attach a text label above each bar displaying its height
    #TODO, take from https://matplotlib.org/2.0.2/examples/api/barchart_demo.html.See the autolabel function
    for rect in scores:
        width = rect.get_width()
        #print(width)
        ax.text(width, rect.get_y() + rect.get_height()/2.,
                f"{width:0.3}",
                va='center', 
                ha='right',
                color="black")





if __name__ == "__main__":
    print("Tensorflow experiment helper functions. Import these into your notebook.")