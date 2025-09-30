import numpy as np


"""np.random.seed(0) is a way to set the seed for NumPy’s random number
generator so that you get the same random numbers every time you run 
the code. """

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

#we are going to define two hidden layers 

class Layer_Dense:
  def __init__(self, n_inputs, n_neurons):
    self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))

  def forward(self, input):
    self.output = np.dot(input, self.weights) + self.biases



  
  
layer1 = Layer_Dense(4, 5)

#the input should be the output of the previous layer
layer2 = Layer_Dense(5, 2)

layer1.forward(X)

layer2.forward(layer1.output)
print(layer2.output)