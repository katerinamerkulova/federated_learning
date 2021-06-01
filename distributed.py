#! /usr/bin/python

import tensorflow as tf
from tensorflow.keras import datasets, layers, models


def load_dataset():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # NUM_CLASSES = 10
    # cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer",
    #                    "dog", "frog", "horse", "ship", "truck"]

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

    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def main():
    train_images, train_labels, test_images, test_labels = load_dataset()

    mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0", "/cpu:1", "/cpu:2"])
    with mirrored_strategy.scope():
        model = build_and_compile_model()

    model.fit(train_images, train_labels, epochs=3,
              validation_data=(test_images, test_labels))

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
    print(f'loss: {test_loss:.4f} - accuracy: {test_acc:.4f}')


if __name__ == "__main__":
    main()
