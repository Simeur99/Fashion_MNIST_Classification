# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import matplotlib.pyplot as plt 

# Load the Fashion MNIST dataset using Keras
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the image data to scale pixel values between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape the data to include a channel dimension (needed for CNN input)
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Define the CNN model with six layers
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Layer 1: Conv2D layer
    layers.MaxPooling2D((2, 2)),                                            # Layer 2: Max pooling
    layers.Conv2D(64, (3, 3), activation='relu'),                           # Layer 3: Conv2D layer
    layers.MaxPooling2D((2, 2)),                                            # Layer 4: Max pooling
    layers.Flatten(),                                                      # Layer 5: Flatten layer
    layers.Dense(64, activation='relu'),                                   # Layer 6: Dense layer
    layers.Dense(10, activation='softmax')                                 # Output layer: 10 classes
])

# Compile the model with optimizer, loss function, and evaluation metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, validation_split=0.1)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# Make predictions on two test images
predictions = model.predict(test_images[:2])

# Display the predictions and actual labels
for i in range(2):
    plt.figure()
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {tf.argmax(predictions[i]).numpy()}, Actual: {test_labels[i]}")
    plt.show() 