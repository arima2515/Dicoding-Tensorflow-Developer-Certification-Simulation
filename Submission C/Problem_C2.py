# =============================================================================
# PROBLEM C2
#
# Create a classifier for the MNIST Handwritten digit dataset.
# The test will expect it to classify 10 classes.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 91%
# =============================================================================

import tensorflow as tf
import numpy as np


def solution_C2():
    mnist = tf.keras.datasets.mnist

    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

    # Normalizing and Reshape
    training_images = training_images / 255.
    training_images = np.expand_dims(training_images, axis=-1)
    test_images = test_images / 255.

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
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
    if __name__ == '__main__':
        model = solution_C2()
        model.save("model_C2.h5")
