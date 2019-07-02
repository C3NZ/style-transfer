"""
    Style transfer learning with keras and tensorflow
"""
# Std lib library imports
import functools
import os
import subprocess
import time

# DS library imports
import matplotlib as matplot
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_core.contrib.eager as tfe
from keras import backend, layers, losses, models
from keras.applications import vgg19
from keras.preprocessing import image as keras_img
from PIL import Image

# Transform matplotlib plot.
matplot.rcParams["figure.figsize"] = (10, 10)
matplot.rcParams["axes.grid"] = False


def get_test_imgs():
    """
        Get any necessary images for testing.
    """
    img_dir = "/tmp/nst"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    img_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/d/d7/Green_Sea_Turtle_grazing_seagrass.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/b/b4/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/0/00/Tuebingen_Neckarfront.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/6/68/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
    ]

    for img in img_urls:
        subprocess.run(["wget", "--quiet", "-P", "/tmp/nst/", img])


def enable_eager():
    """
        Enable and then check if eager is running.
    """
    tf.enable_eager_execution()
    print(f"Eager execution: {tf.executing_eagerly()}")


def load_img(img_path: str) -> np.array:
    """
        Load the image and apply scaling to it.
    """
    # Scale the image to be 512
    max_size = 512
    img = Image.open(img_path)
    longest_dim = max(img.size)
    scale = max_size / longest_dim
    scaled_width, scaled_height = round(img.size[0] * scale), round(img.size[1] * scale)
    scaled_img = img.resize((scaled_width, scaled_height), Image.ANTIALIAS)

    # convert the img to an np array
    scaled_img_arr = keras_img.img_to_array(scaled_img)

    # Add one more dimension so that the image fits the batch size
    batched_img = np.expand_dims(scaled_img_arr, axis=0)
    return batched_img


def plt_img(img_data: np.array, title: str = None) -> None:
    """
        Plot the image within matplotlib
    """
    # Normalize the image to be plotted by matplotlib
    unbatched_img = np.squeeze(img_data, axis=0)
    normalized_img = unbatched_img.astype("uint8")

    # Plot the normalized image data
    plt.imshow(normalized_img)

    # Create a title for the image if passed.
    if title is not None:
        plt.title(title)


def display_test_images() -> None:
    """
        Display the test images that will be used for training purposes
    """
    content_path = "/tmp/nst/Green_Sea_Turtle_grazing_seagrass.jpg"
    style_path = "/tmp/nst/The_Great_Wave_off_Kanagawa.jpg"

    plt.figure(figsize=(10, 10))

    content_img = load_img(content_path).astype("uint8")
    style_img = load_img(style_path).astype("uint8")

    plt.subplot(1, 2, 1)
    plt_img(content_img, "Content img")

    plt.subplot(1, 2, 2)
    plt_img(style_img, "Style img")

    plt.show()


def create_model_from_vgg() -> models.Model:
    """
       Create our neural network from layers that have already been trained within
       the vgg19 model. This will not use the entire vgg16 model, rather take some of it's
       already trained layers that we need from vgg in order to create the model.

       Returns:
        A keras functional model output
    """
    # Define the layers that we specifically want.
    content_layers = ["block5_conv2"]
    style_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]
    # Obtain the model and make all layers untrainable.
    vgg = vgg19.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False

    # Obtain style and content output layers and then merge them
    style_layer_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_layer_outputs = [vgg.get_layer(name).output for name in content_layers]
    all_model_outputs = style_layer_outputs + content_layer_outputs

    return models.Model(vgg.input, all_model_outputs)


def load_and_preprocess_img(img_path: str) -> np.array:
    """
        Load and preprocess an image with vggs image preprocessor that applies 
        normalization to each channel with a specified mean.
        
        Returns:
            
    """
    img = load_img(img_path)
    processed_img = vgg19.preprocess_input(img)
    return processed_img


if __name__ == "__main__":
    enable_eager()
    display_test_images()

    load_and_preprocess_img("/tmp/nst/Green_Sea_Turtle_grazing_seagrass.jpg")
