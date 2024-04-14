
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Model
from keras import layers

def preprocess(array):
    array = array.astype("float32") / 255.0
    return array

def noise(array):
    noise_factor = 0.4
    noisy_array = array + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=array.shape
    )
    return np.clip(noisy_array, 0.0, 1.0)

def display(array1, array2):
    n = 10
    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    plt.figure(figsize=(20, 8))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def build_autoencoder():
    input_img = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    decoded = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    return autoencoder

def main():
    (train_data, _), (test_data, _) = cifar10.load_data()
    train_data = preprocess(train_data)
    test_data = preprocess(test_data)
    noisy_train_data = noise(train_data)
    noisy_test_data = noise(test_data)
    display(train_data, noisy_train_data)
    
    autoencoder = build_autoencoder()
    autoencoder.summary()
    
    autoencoder.fit(x=train_data, y=train_data, epochs=20, batch_size=128, shuffle=True, validation_data=(test_data, test_data))
    predictions = autoencoder.predict(test_data)
    display(test_data, predictions)
    
    autoencoder.fit(x=noisy_train_data, y=train_data, epochs=20, batch_size=128, shuffle=True, validation_data=(noisy_test_data, test_data))
    predictions = autoencoder.predict(noisy_test_data)
    display(noisy_test_data, predictions)

if __name__ == "__main__":
    main()