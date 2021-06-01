#! /usr/bin/python

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from contextlib import redirect_stdout


def load_dataset():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    NUM_CLASSES = 10
    cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    return train_images, train_labels, test_images, test_labels


def build_and_compile_model():
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    with open('model_summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def plot(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.savefig('model_training.png')


def main():
    train_images, train_labels, test_images, test_labels = load_dataset()

    mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0", "/cpu:1", "/cpu:2"])
    with mirrored_strategy.scope():
        model = build_and_compile_model()

    history = model.fit(train_images, train_labels, epochs=3,
                        validation_data=(test_images, test_labels))
    plot(history)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    with open('model_loss_and_accuracy.txt', 'w') as outfile:
        outfile.write(f'loss: {test_loss:.4f} - accuracy: {test_acc:.4f}')


if __name__ == "__main__":
    main()
