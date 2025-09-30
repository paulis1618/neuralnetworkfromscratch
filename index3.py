import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
nnfs.init()

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3)

class Activation_ReLU:
  def forward(self, inputs):
    self.output = np.maximum(0, inputs)

    
class Layer_Dense:
  def __init__(self, n_inputs, n_neurons):
    self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))

  def forward(self, input):
    self.output = np.dot(input, self.weights) + self.biases



  
  
layer1 = Layer_Dense(2, 5)

activation1 = Activation_ReLU()

layer1.forward(X)

# print(layer1.output)
activation1.forward(layer1.output)

print(activation1.output)