import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets



np.random.seed(0)
X, y = datasets.make_moons(n_samples=100, shuffle=True, noise=None, random_state=None)

plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

# Defining the dimensionalities

number_example = len(X)
nn_input_dim = 2
nn_output_dim = 2

# Gradient Descent Parameters
eps= 0.01 #epsilion(Learning rate)
reg_lambda = 0.01 # also sometimes refered as C value. Stands for speed of light. jk!!

# Loss Evaluation function
def loss_func(model):
	W1, b1, W2, b2 = model['W1'],model['b1'],model['W2'],model['b2']
    
    # Forward propagation to calculate our predictions
	Z1 = X.dot(W1)+b1
	a1 = np.tanh(Z1)

	Z2 = a1.dot(W2)+b2

	exp_value = np.exp(Z2)
	probability = exp_value / np.sum(exp_value, axis=1, keepdims=True)


	# Calculating loss:
	correct_logprobs= -np.log(probability[range(number_example),y])
	data_loss = np.sum(correct_logprobs)

	# Add regulatization term to loss (optional)
	data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
	return 1./number_example * data_loss


def predict(model,x):
	W1, b1, W2, b2 = model['W1'],model['b1'],model['W2'],model['b2']
	# feed forward network
	Z1 = X.dot(W1)+b1
	a1 = np.tanh(Z1)

	Z2 = a1.dot(W2)+b2
	exp_value = np.exp(Z2)
	probability = exp_value / np.sum(exp_value, axis=1, keepdims=True)
	return np.argmax(probability, axis=1)


# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(nn_hdim, num_passes=20000, print_loss=False):
     
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))
 
    # This is what we return at the end
    model = {}
     
    # Gradient descent. For each batch...
    for i in xrange(0, num_passes):
 
        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
 
        # Backpropagation
        delta3 = probs
        delta3[range(number_example), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
 
        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1
 
        # Gradient descent parameter update
        W1 += -eps * dW1
        b1 += -eps * db1
        W2 += -eps * dW2
        b2 += -eps * db2
         
        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
         
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
          print "Loss after iteration %i: %f" %(i, loss_func(model))
     
    return model


# Build a model with a 3-dimensional hidden layer
model = build_model(3, print_loss=True)
