#implementing the function of a single neuron

import numpy as np

#input of the neuron
#one single input with four features 
inputs = [3.0, 4.3, 5.4, 6.8]

#weights of the neuron
weights = [0.29, 0.56, 0.33, 0.12]

#bias of the neuron
bias = [3]

output = 0.0

#each item in the inputs list is multiplied by an item in the
#weights list of same position, and then they are added all together
for i in range(len(inputs)):
  output += inputs[i] * weights[i]
  
output += bias[0]

print(output)

# or if we implement it with the np.dot function that is used to multiply matrices
# effectively, by using parallel hardware
#we get the same result

output2 = np.dot(inputs, weights) + bias

print(output2[0])