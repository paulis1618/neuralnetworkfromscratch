import numpy as np
import nnfs
from nnfs.datasets import spiral_data


nnfs.init() #used for making sure that every time we run the code, we get the same results and dont and dont rely on luck for example when initializing values
            # could be removed, and the result will be the same, it is there just to make sure that we get the same results as the tutorial that i followed

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


print(X.shape)
  
# the dimension 2 is there because input has two features
# input has two features is same with input has dimensions (300, 2) where 300 = 3 * 100
layer1 = Layer_Dense(2, 5)

activation1 = Activation_ReLU()

layer1.forward(X)

activation1.forward(layer1.output)

print(activation1.output)