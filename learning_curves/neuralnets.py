"""
Back-propagation neural network, using numpy

Heavily modified from http://stackoverflow.com/a/3143318
"""

from numpy import *

def sigmoid(xs):
  """Compute sigmoid at each x in vector"""
  return 2.0 / (1.0 + exp(-xs)) - 1.0

def dsigmoid(ys):
  """Compute sigmoid derivative from each y in vector"""
  return 0.5 * (1.0 - ys*ys)

class NN(object):

  def __init__(self, n_in, *hidden):
    """Create a neural network with random initial weights"""
    hidden, n_out = hidden[:-1], hidden[-1]

    # Initialize the count of nodes in each layer
    self.layers = array((n_in,) + hidden + (n_out,))
    self.n_layers = len(self.layers)

    # Initialize activation values for each node
    self.activations = [ones(node_count) for node_count in self.layers]
    
    # Initialize weights between levels, and past changes for momentum
    # Full connections between levels are used
    self.weight_shapes = zip(self.layers[1:], self.layers[:-1] + 1)
    self.weights = [random.uniform(-0.5, 0.5, shape) for shape in self.weight_shapes]
    self.past_change = [zeros(shape) for shape in self.weight_shapes]

    # Initialize scales used on training set
    # Scaling helps prevent weights from growing extremely large or small
    self.input_mean = zeros(n_in)
    self.input_scale = ones(n_in)
    self.output_mean = zeros(n_out)
    self.output_scale = ones(n_out)

  def activate(self, inputs):
    """Activate all nodes, and output last layer"""
    inputs = (inputs - self.input_mean) / self.input_scale
    return self._activate(inputs) * self.output_scale + self.output_mean

  def _activate(self, inputs):
    """Activate all nodes, and output last layer. Don't scale"""
    self.activations[0] = inputs

    # Activate each hidden layer, using sigmoid function
    for i in xrange(self.n_layers - 2):
      self.activations[i + 1] = sigmoid(self.weights[i].dot(append(self.activations[i], 1.0)))

    # Don't use sigmoid on last layer
    # Simulates a perceptron when there are no hidden layers
    self.activations[-1] = self.weights[-1].dot(append(self.activations[-2], 1.0))

    return self.activations[-1]

  def backPropagate(self, targets, a, M):
    """Update weights assuming neural network is activated to achieve targets""" 
    current_change = [zeros(shape) for shape in self.weight_shapes]
    
    error = self.activations[-1] - targets
    # Compute square error for return
    err = dot(error * self.output_scale, error * self.output_scale) 
    deltas = error

    current_change[-1] = outer(deltas, append(self.activations[-2], 1.0))

    # Compute the gradient with respect to each weight matrix using recurrence
    # The heart of the backpropagation algorithm
    for i in reversed(xrange(self.n_layers - 2)):    
      error = self.weights[i + 1].T.dot(deltas)[:-1]
      deltas = dsigmoid(self.activations[i + 1]) * error

      # Store the gradient with respect to the current layer's weights for update
      current_change[i] = outer(deltas, append(self.activations[i], 1.0))

    # Update all weights by going in the opposite direction of the gradient
    for i in xrange(self.n_layers - 1):
      self.weights[i] -= M*self.past_change[i] + a*current_change[i]
      self.past_change[i] = current_change[i]

    return err

  def train(self, data, targets, num_epochs=1000, a=0.02, M=0.002, e=0.000001, verbose=True):
    """Trains the neural network"""

    # Compute and scale the dataset
    self.input_mean = data.mean(0)
    self.input_scale = data.std(0)
    self.output_mean = targets.mean(0)
    self.output_scale = targets.std(0)

    data = (data - self.input_mean) / self.input_scale
    targets = (targets - self.output_mean) / self.output_scale

    # Keep track of the error in each round of training
    # If the error changes by less than e, neural network has converged, so halt training
    past_error = -2 * e
    error = 0.0

    # In each epoch, train the neural network on the entire dataset
    for i in xrange(num_epochs):
      error = 0.0
      for x,y in zip(data, targets):
        self._activate(x)
        error += self.backPropagate(y, a, M)

      if verbose and i % max(num_epochs / 10, 1) == 0:
        print "Iteration: %s of %s, Error: %s" % (i, num_epochs, error / len(data))
      
      # Halt if converged
      if abs(past_error - error) < e:
        if verbose:
          print "Converged on iteration %s of %s, Error: %s" % (i, num_epochs, error / len(data))
        break
      past_error = error

    # Return mean squared error
    return error / len(data)