# Load reticulate library first
library(reticulate)
#Set your Python path here
use_python("C:/Python313/python.exe", required = TRUE)
py_config()

# Load other libraries
library(keras3)
library(tensorflow)

# Load the Fashion MNIST dataset
fashion_mnist <- dataset_fashion_mnist()
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

# Normalize the image data
train_images <- train_images / 255
test_images <- test_images / 255

# Reshape the data to include a channel dimension
train_images <- array_reshape(train_images, c(nrow(train_images), 28, 28, 1))
test_images <- array_reshape(test_images, c(nrow(test_images), 28, 28, 1))

# Convert labels to integers (if needed)
train_labels <- as.integer(train_labels)
test_labels <- as.integer(test_labels)

# Define the CNN model with six layers
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3),
                activation = "relu", input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

# Compile the model
model %>% compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)

# Train the model
model %>% fit(
  train_images, train_labels,
  epochs = 5,
  validation_split = 0.1
)

# Evaluate the model
score <- model %>% evaluate(test_images, test_labels)
cat("Test accuracy:", score$accuracy, "\n")


# Make predictions on two test images
# Make predictions
predictions <- model %>% predict(test_images[1:2, , , , drop = FALSE])

# Display the 2 predictions and actual labels using for loop
for (i in 1:2) {
  predicted_label <- which.max(predictions[i, ]) - 1
  actual_label <- test_labels[i]
  image(matrix(test_images[i, , , 1], 28, 28)[28:1, ], col = gray.colors(256),
        main = paste("Predicted:", predicted_label, "Actual:", actual_label))
}
