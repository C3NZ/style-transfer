"""
    Style transfer learning with keras and tensorflow
"""
# Std lib library imports
import functools
import os
import subprocess
import time

import IPython.display
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
    print(f"Eager execution is running: {tf.executing_eagerly()}")


def load_img(img_path: str) -> np.array:
    """
        Load the image and apply scaling to it.

        Returns:
            An np array of our scaled image.
    """
    # Scale the image to be 512
    max_size = 256
    img = Image.open(img_path)

    longest_dim = max(img.size)
    scale = max_size / longest_dim
    print(img.size)
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

        Params:
            img_data - The image 4d (batch, w, h, c) in array representation.
            title (None) - The title of the image to be plotted.
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
    # Content layer where we obtain our feature maps
    content_layers = ["block5_conv2"]

    # The styling layers we want
    style_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]
    # Obtain the model and make all layers untrainable.
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False

    # Obtain style and content output layers and then merge them
    style_layer_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_layer_outputs = [vgg.get_layer(name).output for name in content_layers]
    all_model_outputs = style_layer_outputs + content_layer_outputs

    return tf.keras.models.Model(vgg.input, all_model_outputs)


def load_and_preprocess_img(img_path: str) -> np.array:
    """
        Load and preprocess an image with vggs image preprocessor that applies 
        normalization to each channel with a specified mean.

        Returns:
            An np array containing the image that we're looking to deprocess
    """
    img = load_img(img_path)
    processed_img = vgg19.preprocess_input(img)
    return processed_img


def deprocess_img(processed_img: np.array) -> np.array:
    """
        Takes in a vgg19 preprocessed image and reverses the preprocessing by adding 
        approximate values applied to the channels and then reversing the channels.
    """
    # Create a copy of the processed image
    processed_img = processed_img.copy()

    # If the image is shaped to fit within a batch, squeeze out the fourth dimension.
    if len(processed_img.shape) == 4:
        processed_img = np.squeeze(processed_img, axis=0)

    # Ensure that our image only contains the X, Y, and channel data.
    assert len(processed_img.shape) == 3

    # Apply the inverse of the vgg19 preprocessing that's applied to return all channels  back to their
    # nearest original values (approximation)
    processed_img[:, :, 0] += 103.939
    processed_img[:, :, 1] += 116.779
    processed_img[:, :, 2] += 123.68

    # Reverse the channels
    processed_img = processed_img[:, :, ::-1]

    # Clip all values to be within uint8 range (0 - 255) for pixel data
    processed_img = np.clip(processed_img, 0, 255).astype("uint8")
    return processed_img


def get_content_loss(og_content: np.array, target_content: np.array):
    """
        Obtain the euclidean distance between our original content and our target
        content and then take the mean distance of all points.
    """
    return tf.reduce_mean(tf.square(og_content - target_content))


def create_gram_matrix(tensor: np.array):
    """
        Create a gram matrix by obtaining the dot product of the feature map
        of the original image.

        Returns:
            The gram matrix of the content img
    """
    channels = int(tensor.shape[-1])
    style_img = np.reshape(tensor, [-1, channels])
    wtf = np.shape(style_img)[0]
    print(wtf)
    gram_matrix = tf.matmul(style_img, style_img, transpose_a=True)

    return gram_matrix / tf.cast(wtf, tf.float32)


def get_style_loss(og_style: np.array, target_style: np.array):
    """
        Obtain the style loss by taking the euclidean distance between the gram matricies of
        the target style and input style.

        Returns:
            The mean euclidean distance between all points.
    """
    og_gram_matrix = create_gram_matrix(og_style)
    return tf.reduce_mean(tf.square(og_gram_matrix - target_style))


def get_feature_representations(model, content_img_path, style_img_path):
    """
        Obtain the feature representations of our images through one forward
        propagation of each model.
    """
    # Load in our content and sytle images prepared to be inputs within
    # our model.
    content_img = load_and_preprocess_img(content_img_path)
    style_img = load_and_preprocess_img(style_img_path)

    # Computes one batch of content & style and features
    content_outputs = model(content_img)
    style_outputs = model(style_img)

    # Obtain the features for every style layer that is within our model. (5 total)
    content_features = [content_layer[0] for content_layer in content_outputs[5:]]
    style_features = [style_layer[0] for style_layer in style_outputs[:5]]

    return style_features, content_features


def compute_total_loss(
    model, loss_weights, init_image, gram_style_features, content_features
) -> tuple:
    style_weight, content_weight = loss_weights

    # This gives us the content and style features at the desired layers.
    model_outputs = model(init_image)

    style_output_features = model_outputs[:5]
    content_output_features = model_outputs[5:]

    style_score = 0
    content_score = 0

    # 5 total style layers
    weight_per_style_layer = 1.0 / 5.0

    # Compute the total style loss for all layers
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        # Obtain the total score
        style_score += weight_per_style_layer * get_style_loss(
            comb_style[0], target_style
        )

    weight_per_content_layer = 1

    # Obtain the total content loss from all layers
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(
            comb_content[0], target_content
        )

    # Multiply the scores by their corresponding loss weights
    style_score *= style_weight
    content_score *= content_weight

    total_loss = style_score + content_score

    return total_loss, style_score, content_score


def compute_gradients(config) -> None:
    """
        Compute the gradients within our losses
    """
    with tf.GradientTape() as tape:
        all_losses = compute_total_loss(**config)

        total_loss = all_losses[0]

        return tape.gradient(total_loss, config["init_image"]), all_losses


def compute_style_transfer(
    content_img_path,
    style_img_path,
    num_iterations=1000,
    content_weight=1e3,
    style_weight=1e-2,
):
    # Obtain our stripped down vgg model.
    model = create_model_from_vgg()

    # Make all layers not trainable
    for layer in model.layers:
        layer.trainable = False

    # Obtain the style and content features from our intermediate layers
    style_features, content_features = get_feature_representations(
        model, content_img_path, style_img_path
    )

    gram_style_features = [
        create_gram_matrix(style_feature) for style_feature in style_features
    ]

    init_image = load_and_preprocess_img(content_img_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)

    optimizer = tf.train.AdamOptimizer(learning_rate=10, beta1=0.99, epsilon=1e-1)

    iter_count = 1

    best_loss, best_img = float("inf"), None

    loss_weights = (style_weight, content_weight)

    config = {
        "model": model,
        "loss_weights": loss_weights,
        "init_image": init_image,
        "gram_style_features": gram_style_features,
        "content_features": content_features,
    }
    # For displaying
    num_rows = 2
    num_cols = 5
    display_interval = num_iterations / (num_rows * num_cols)
    start_time = time.time()
    global_start = time.time()

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    imgs = []
    for i in range(num_iterations):
        print(f"Iteration: {i}")
        grads, all_loss = compute_gradients(config)
        loss, style_score, content_score = all_loss
        optimizer.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)

        if loss < best_loss:
            # Update best loss and best image from total loss.
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())

    return best_img, best_loss


def show_results(best_img, content_path, style_path, show_large_final=True):
    plt.figure(figsize=(10, 5))
    content = load_img(content_path)
    style = load_img(style_path)

    plt.subplot(1, 2, 1)
    plt_img(content, "Content Image")

    plt.subplot(1, 2, 2)
    plt_img(style, "Style Image")

    if show_large_final:
        plt.figure(figsize=(10, 10))

        plt.imshow(best_img)
        plt.title("Output Image")
        plt.show()


if __name__ == "__main__":
    enable_eager()
    content_path = "./imgs/content/mels.jpg"
    style_path = "./imgs/styles/deepfry.jpg"
    best, best_loss = compute_style_transfer(
        content_path, style_path, num_iterations=1000
    )
    Image.fromarray(best)
    show_results(best, content_path, style_path)
