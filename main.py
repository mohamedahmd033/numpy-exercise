import numpy as np

def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    #Your code here
    A = np.random.rand(n,1)
    return A


def operations(h, w):
    """
    Takes two inputs, h and w, and makes two Numpy arrays A and B of size
    h x w, and returns A, B, and s, the sum of A and B.

    Arg:
      h - an integer describing the height of A and B
      w - an integer describing the width of A and B
    Returns (in this order):
      A - a randomly-generated h x w Numpy array.
      B - a randomly-generated h x w Numpy array.
      s - the sum of A and B.
    """
    #Your code here
    A = np.random.rand(h,w)
    B = np.random.rand(h,w)
    s = A + B
    return A,B,s



def norm(A, B):
    """
    Takes two Numpy column arrays, A and B, and returns the L2 norm of their
    sum.

    Arg:
      A - a Numpy array
      B - a Numpy array
    Returns:
      s - the L2 norm of A+B.
    """
    #Your code here
    s=A+B
    s= np.linalg.norm(s)
    return s



def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    #Your code here
    z= np.array(np.tanh(np.dot(inputs,weights)))
    return z


def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    #Your code here
    if x <= y:
        return x*y
    else:
        return x/y


def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y 
    """
    #Your code here
    newfunction = np.vectorize(scalar_function)
    return newfunction(x, y)


#############TESTS###########################
print('randomization test')
print(randomization(7))

print('operation test')
print(operations(3, 4))

print('norm test')
x=np.array([1,2,3])
y=np.array([6,9,11])
print(norm(x,y))

print('neural_network test')
A=np.array([2,6])
B=np.array([3,7])
print(neural_network(A,B))

print('scalar_function test')
print(scalar_function(3,5))
print(scalar_function(7,5))

print('vector_function test')
print(vector_function([8, 6], [2, 4]))
