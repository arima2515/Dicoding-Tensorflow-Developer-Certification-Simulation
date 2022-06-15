# =============================================================================
# PROBLEM B2
#
# Build a classifier for the Fashion MNIST dataset.
# The test will expect it to classify 10 classes.
# The input shape should be 28x28 monochrome. Do not resize the data.
# Your input layer should accept (28, 28) as the input shape.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 83%
# =============================================================================

import tensorflow as tf
import numpy as np

def solution_B2():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Normalizing and Reshape
    training_images = training_images / 255.
    training_images = np.expand_dims(training_images, axis=-1)
    test_images = test_images / 255.

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(
        training_images,
        training_labels,
        epochs=10,
        verbose=1,
        validation_data=[test_images, test_labels],
        validation_batch_size=32
    )

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_B2()
    model.save("model_B2.h5")


