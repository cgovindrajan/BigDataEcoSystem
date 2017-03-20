import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    #raise NotImplementedError
    dotProdofdataNW1 = np.dot(data, W1)
    val = sigmoid(dotProdofdataNW1 + b1 )
    dotProdofval1NW2 = np.dot(val, W2)
    predict = sigmoid(dotProdofval1NW2 + b2)
    
    costCurrent = labels * np.log(predict)
    costPrevious = (1 -labels) * np.log(1 - predict)
    cost = np.sum(costCurrent + costPrevious)
    cost = (-1) * cost 
    ### END YOUR CODE
    
    ### YOUR CODE HERE: backward propagation
    #raise NotImplementedError
    diff = predict - labels 
    diffVal = np.dot(diff, W2.T) * sigmoid_grad(val)  
    
    gradW2 = np.dot(val.T, diff) 
    gradb2 = np.sum(diff, 0)  
    gradW1 = np.dot(data.T, diffVal)
    gradb1 = np.sum(diffVal, 0)


     ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
        gradW2.flatten(), gradb2.flatten()))
    
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
