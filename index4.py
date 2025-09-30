import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

    
class Layer_Dense:
  def __init__(self, n_inputs, n_neurons):
    self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))

  def forward(self, input):
    self.output = np.dot(input, self.weights) + self.biases

class Activation_ReLU:
  def forward(self, inputs):
    self.output = np.maximum(0, inputs)

class Activation_Softmax:
  def forward(self, inputs):
   #to prevent overflow we substract the maximum value of the row, so our x is between - infinite and 0
   #axis = 1 and keepdims = True are there to make sure that we get the sum and
   # the max of our batch / our row and not of the whole matrix
   exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) 
   probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
   self.output = probabilities

X, y = spiral_data(100, 3)

#the 2 is defined from our spiral database(features of database)
#the 3 is the output to the next layer
#we define relu as our activation, suitable for all hidden layers
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

#the second and final layer is the output layer
#our input from the previous layers is of dimension 3 
#so this is why the first dimension of layer2 is 3
#the second dimension is also 3, because as you remember the 
#spiral data database is of 3 classes, so each neuron of that layer
#will output a number that is the probability that sample belongs
# to each class
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

#in each layer we do the computations first and then apply the activation function later
dense1.forward(X)

activation1.forward(dense1.output)

dense2.forward(activation1.output)

activation2.forward(dense2.output)

print(activation2.output[:5])