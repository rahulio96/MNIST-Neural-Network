from functions import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load the weights from CSV files
w1_data = pd.read_csv('./trained_weights/w1.csv')
w2_data = pd.read_csv('./trained_weights/w2.csv')
b1_data = pd.read_csv('./trained_biases/b1.csv')
b2_data = pd.read_csv('./trained_biases/b2.csv')

# Convert the data to numpy arrays
w1 = np.array(w1_data)
w2 = np.array(w2_data)
b1 = np.array(b1_data)
b2 = np.array(b2_data)

# Test neural network

# Load and preprocess the test data
test_data = pd.read_csv("./data/test.csv")
test_data = np.array(test_data).T
x_test = test_data[:, 6620, None]

# 1620
# 6620
# 766

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