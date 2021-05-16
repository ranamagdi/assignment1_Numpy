

import numpy as np




def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    #Your code here
    A = np.random.random([n,1])
    return A
    raise NotImplementedError



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
    A = np.random.random([h,w])
    B = np.random.random([h,w])
    S=A+B
    return A,B,S
   
    raise NotImplementedError




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
   
   
    s = np.linalg.norm(A + B)
    return s
    raise NotImplementedError



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
    return np.tanh(np.matmul(weights.transpose(), inputs))
    raise NotImplementedError







def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    #Your code here
    if x<=y: 
        return x * y
    else  :
        return  x/y
    raise NotImplementedError




def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y 
    """
    #Your code here
    output = np.vectorize(scalar_function)
    return output(x,y)
    raise NotImplementedError



def main():
    A = np.random.random([4,])
    B = np.random.random([4,])
   
    x=randomization(3)
    print(x)
    y=operations(2, 2)
    print(y)
    z=norm(A, B)
    print(z)
    F=neural_network(np.random.random([2, 1]), np.random.random([2, 1]))
    print(F)
    v=scalar_function(3, 2)
    print(v)
    b= vector_function(3,2)
    print(b)
    
        
if __name__ == "__main__":
        main()
   





