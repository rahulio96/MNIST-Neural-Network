import numpy as np

# Activation Function: Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Activation Function: Softmax
def softmax(x):
    exp_scores = np.exp(x)
    return exp_scores / np.sum(exp_scores, axis=0)

# Convert integer labels to one-hot encoded vectors
def one_hot_encode(labels, num_classes):
    num_samples = labels.shape[0]
    one_hot = np.zeros((num_samples, num_classes))
    one_hot[np.arange(num_samples), labels] = 1
    return one_hot

# Forward Propagation
def forward_propagation(w1, w2, b1, b2, x_train):
    weight_sum1 = np.dot(w1, x_train) + b1
    active_output1 = sigmoid(weight_sum1)

    weight_sum2 = np.dot(w2, active_output1) + b2
    active_output2 = softmax(weight_sum2)

    return weight_sum1, weight_sum2, active_output1, active_output2