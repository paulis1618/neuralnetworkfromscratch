import numpy as np

#we are going to implement a two layer neural network


#one input with four features
inputs = np.array([[1, 2, 3, 2.5]])

#weights of the first layer. 3x4 matrix meaning 3 neurons and made for 
# input of 4 features
weights = np.array([[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]])

#biases of the 3 neurons of the first layer
biases = [2, 3, 0.5]

#weights of the first layer. 3x3 matrix because the second layer has 3 neurons
#and the the first layer has 3 outputs, so it produces 
weights2 = np.array([[0.1, -0.14, 0.5],
                    [-0.5, 0.12, -0.33],
                    [-0.44, 0.73, -0.13]])

#biases of the 3 neurons of the first layer
biases2 = [-1, 2, -0.5]


layer1_outputs = np.dot(inputs, weights.T) + biases

layer2_outputs = np.dot(layer1_outputs, weights2.T) + biases2

print(layer1_outputs)
print(layer2_outputs)

