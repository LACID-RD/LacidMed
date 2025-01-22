import tensorflow as tf
from tensorflow.keras import layers, models


import pydicom
import matplotlib.pyplot as plt
import numpy as np

# lacid_med directory append.
import sys
sys.path.append("C:/Users/santi/Desktop/Fuesmen/code/LacidMed")

from lacid_med.src.processing.operations import Operations
from lacid_med.src.visualization.plotter import DicomPlotter
from lacid_med.src.handler.loader import DicomLoaderMRI
from lacid_med.src.handler.writer import MultipleFileWriter

# Cargar y preprocesar el conjunto de datos CIFAR-10
(train_images, _), (_, _) = tf.keras.datasets.cifar10.load_data()
train_images = (train_images.astype('float32') - 127.5) / 127.5  # Normalizar las imágenes al rango [-1, 1]

# Definir el generador
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(4 * 4 * 256, input_shape=(100,)),
        layers.Reshape((4, 4, 256)),
        layers.Conv2DTranspose(128, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        layers.Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        layers.Conv2DTranspose(3, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='tanh'),
    ])
    return model

# Definir el discriminador
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=[32, 32, 3]),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Crear el generador y el discriminador
generator = build_generator()
discriminator = build_discriminator()

# Compilar el discriminador
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                      loss='binary_crossentropy', metrics=['accuracy'])

# Crear el modelo GAN
discriminator.trainable = False
gan_input = tf.keras.Input(shape=(100,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = tf.keras.models.Model(gan_input, gan_output)
gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy')

# Función para mostrar imágenes generadas y originales lado a lado
def show_generated_images_with_input(epoch, generator, examples=10, dim=(2, 10), figsize=(10, 2)):
    noise = tf.random.normal([examples, 100])
    generated_images = generator.predict(noise)
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], 2*i+1)
        plt.imshow((generated_images[i] + 1) / 2)  # Desnormalizar las imágenes generadas para visualización
        plt.axis('off')
        plt.subplot(dim[0], dim[1], 2*i+2)
        plt.imshow((train_images[i] + 1) / 2)  # Desnormalizar las imágenes originales para visualización
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)
    plt.show()

# Entrenar el modelo GAN
epochs = 50
batch_size = 128
steps_per_epoch = len(train_images) // batch_size

for epoch in range(epochs): 
    for step in range(steps_per_epoch):
        noise = tf.random.normal([batch_size, 100])
        generated_images = generator.predict(noise)
        real_images = train_images[np.random.randint(0, train_images.shape[0], size=batch_size)]

        combined_images = np.concatenate([real_images, generated_images])
        labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
        labels += 0.05 * np.random.random(labels.shape)  # Etiquetas con ruido suave

        d_loss = discriminator.train_on_batch(combined_images, labels)

        noise = tf.random.normal([batch_size, 100])
        misleading_targets = np.zeros((batch_size, 1))
        a_loss = gan.train_on_batch(noise, misleading_targets)

    print(f'Epoch {epoch+1}/{epochs}, Discriminator Loss: {d_loss[0]}, Generator Loss: {a_loss}')
    if (epoch+1) % 10 == 0:
        show_generated_images_with_input(epoch, generator)