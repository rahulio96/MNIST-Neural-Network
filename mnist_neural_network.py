import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from functions import *

# Neural Nework for hand written digit identification
# Using the MNIST data set for training and testing

# Read training data from train csv file
training_data = pd.read_csv("./data/train.csv")
training_data = np.array(training_data)

# Randomize the training data
np.random.shuffle(training_data)

# Use a smaller subset of data for faster results (just for light testing)
sub_data = training_data[0:1000].T  # Add a number to the right of the colon such as 1000

m, n = sub_data.shape

# Contains pixel data
x_train = sub_data[1:n]

# Contains labels for the images
y_train = sub_data[0]

input_size = 784    # 28 pixels * 28 pixels = 784 pixels
hidden_size = 15
output_size = 10    # There can be only 10 outputs (numbers 0-9)

# Initialize Weights
# Weights are matrices and are randomly initialized
w1 = np.random.randn(hidden_size, input_size) * 0.01
w2 = np.random.randn(output_size, hidden_size) * 0.01

# Initialize Bias
# Biases are zero vectors initially, not random like weights
b1 = np.zeros((hidden_size, 1))
b2 = np.zeros((output_size, 1))

learning_rate = 0.02
iterations = 2000

# Convert integer labels to one-hot encoded vectors
y_train_encoded = one_hot_encode(y_train, output_size)

# Training Loop
for iteration in range(iterations):

    # Forward propagation
    weight_sum1, weight_sum2, active_output1, active_output2 = forward_propagation(w1, w2, b1, b2, x_train)  


    #-----------------------------------------------------------------------------------------------------------------------#

    # Make predictions
    predictions = np.argmax(active_output2, axis=0)  # Get the index of the maximum probability for each training sample

    # Convert the actual labels to integers from one-hot encoded vectors
    y_train = np.argmax(y_train_encoded, axis=1)

    # Calculate the number of correct predictions
    num_correct = np.sum(predictions == y_train)

    # Calculate the accuracy
    num_samples = y_train.shape[0]
    accuracy = (num_correct / num_samples) * 100

    # Print the accuracy
    if iteration % 100 == 0:
        print(f"Training accuracy: {accuracy}%")

    #-----------------------------------------------------------------------------------------------------------------------#


    # Number of training samples
    training_samples = x_train.shape[0]

    # Calculate cost (Cross-entropy loss function)
    cost = (-1 / training_samples) * np.sum(y_train_encoded * np.log(active_output2).T + (1 - y_train_encoded) * np.log(1 - active_output2).T)

    # Print the cost
    if iteration % 100 == 0:
        print(cost)
    
    # Back Propagation

    # Derivative of the cost function with respect to the weighted sum of the output layer
    d_cost_sum = active_output2.T - y_train_encoded

    # Derivative of the cost function with respect to the weights and biases of the output layer
    d_w2 = (1 / training_samples) * np.dot(d_cost_sum.T, active_output1.T)
    d_b2 = (1 / training_samples) * np.sum(d_cost_sum, axis=0, keepdims=True)

    # Derivative of the cost function with respect to the weighted sum of the hidden layer
    d_sigmoid1 = np.dot(w2.T, d_cost_sum.T) * active_output1 * (1 - active_output1)

    # Derivative of the cost function with respect to the weights and biases of the hidden layer
    d_w1 = (1 / training_samples) * np.dot(d_sigmoid1, x_train.T)
    d_b1 = (1 / training_samples) * np.sum(d_sigmoid1, axis=1, keepdims=True)

    # Update weights and biases
    w1 -= learning_rate * d_w1
    w2 -= learning_rate * d_w2
    b1 -= learning_rate * d_b1
    b2 -= learning_rate * d_b2.T


"""
# Save w1 to 'w1.csv'
pd.DataFrame(w1).to_csv('w1.csv', index=False)

# Save w2 to 'w2.csv'
pd.DataFrame(w2).to_csv('w2.csv', index=False)

# Save b1 to 'b1.csv'
pd.DataFrame(b1).to_csv('b1.csv', index=False)

# Save b2 to 'b2.csv'
pd.DataFrame(b2).to_csv('b2.csv', index=False)
print("csv files exported")
"""
print(output_size)

# Test our neural network

# Load and preprocess the test data
test_data = pd.read_csv("./data/test.csv")
test_data = np.array(test_data).T
x_test = test_data[:, 650, None]

# Perform forward propagation on the test data
weight_sum1, weight_sum2, active_output1, active_output2 = forward_propagation(w1, w2, b1, b2, x_test)

# Make predictions
predictions = np.argmax(active_output2, axis=0)  # Get the index of the maximum probability for each test sample

# Print the predictions
print(predictions)

# Display the image
current_image = x_test.reshape((28, 28)) * 255
plt.gray()
plt.imshow(current_image, interpolation='nearest')
plt.show()
