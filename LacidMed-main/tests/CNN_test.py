import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np


# Define the CNN architecture
def create_cnn(input_shape, num_classes):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Flatten layer to transition from convolutional layers to fully connected layers
    model.add(layers.Flatten())

    # Fully connected layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer with softmax activation for classification

    return model

# Set the input shape and number of classes
input_shape = (28, 28, 1)  # Example input shape for MNIST dataset
num_classes = 10  # Number of classes in MNIST dataset (digits 0-9)

# Create the CNN model
model = create_cnn(input_shape, num_classes)

# Display the model summary
model.summary()

# Load and preprocess a default image (e.g., from the MNIST dataset)
default_image = np.random.random((28, 28, 1))  # Example random image
default_image = np.expand_dims(default_image, axis=0)  # Add batch dimension

# Predictions on the default image
predictions = model.predict(default_image)

# Show the default image
plt.figure(figsize=(4, 4))
plt.imshow(default_image[0, :, :, 0], cmap='gray')
plt.title('Example Default Image')
plt.axis('off')
plt.show()

# Show the predictions
print("Predictions probabilities:", predictions)
predicted_class = np.argmax(predictions)
print("Predicted Class:", predicted_class)
