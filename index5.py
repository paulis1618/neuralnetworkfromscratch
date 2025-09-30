#categorical cross entropy

import math

# lets say we have a classification problem where we want to predict if the pictured animal is a cat(1) or a dog(0)
# the softmac outputs the probability of the pictured animal being a cat, and when the output is 0.7 for example
# this means that it is 70% likely the pictured animal is a cat, and 30% likely the pictured animal is a dog
softmax_output = [0.7, 0.1, 0.2]

# whether the pictured animal is actually a cat or a dog
target_output = [1, 0, 0]

# to count the total loss we compute the log of each softmax output (prediction) times the actual target output
# and add them all together
loss = - (math.log(softmax_output[0]) * target_output[0] +
          math.log(softmax_output[1]) * target_output[1] +
          math.log(softmax_output[2]) * target_output[2]
          )

print(loss)