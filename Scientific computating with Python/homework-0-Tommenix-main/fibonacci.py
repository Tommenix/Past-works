"""
fibonacci

functions to compute fibonacci numbers

Complete problems 2 and 3 in this file.
"""

import numpy as np
import matplotlib.pyplot as plt
import time # to compute runtimes
from tqdm import tqdm # progress bar

"""
fibonacci
functions to compute fibonacci numbers
Complete problems 2 and 3 in this file.
"""

# Question 2
def fibonacci_recursive(n):
    """
    fibonacci sequence with each term computed by its recursive definition, starting form the first two terms.
    """

    if n<0 or type(n) != int:                # in case some other thing were typed
        return "n should be a positive integer."
    if n ==0:                #set up base for recursion
        return 0
    if n == 1: 
        return 1 
    return fibonacci_recursive(n-1)+fibonacci_recursive(n-2) #define the number by its recursive definition


# Question 2
def fibonacci_iter(n):
    """
    fibonacci sequence with each term computed the iterative method given in question.
    """

    if n<0 or type(n) != int:               # in case some other thing were typed
        return "n should be a positive integer."
    lst = [0,1]
    if n == 0: 
        return 0
    for i in range(n-1):            #compute using only a for loop instead of a recursive definition, but still by recursive definition.
        lst = [lst[1],lst[0]+lst[1]]
    return lst[1]

print('''
    I\'m guessing that the recursive method will be much slower since, if for the nth term we need [n] operations then for the Nth term we need [n-1]+[n-2]+1 operations, which is even larger than the Fibonacci sequence itself! Since we know the kth term of the Fibonacci sequence is A(a)^n+B(b)^n (where a and b are solutions to the characteristic function x^2 = x+1), the number of operations will grow exponentially in n, which is very large.
    As for the iteration method, there's only O(N) operations per term and thus there's O(N^2) operations in total. Also, there's only one call of the function per term, which reduces some time(hopefully?) for the computer to activate the sourse codes of the definition of the function. ''')

# Question 3
def matrix_power(A, n):
    """
    returns the product A ** n for 
    assume n is a nonegative integer
    """
    try:
        if type(A) != np.ndarray:                    # check if A is an array
            return "you should enter a square matrix."
        if np.shape(A)[0] != np.shape(A)[1] or len(np.shape(A))!=2: # check if A is a matrix
            return "you should enter a square matrix."
        if n<0 or type(n) != int:            # in case some other thing were typed
            return "n should be a positive integer."
        if n == 0:                     # define 0-th degree to be I
            return np.identity(len(A))  
        if n == 1:                  
            return A
        if n % 2 == 0:                 #Egyptian algorithm
            return matrix_power(np.matmul(A,A), n//2)
        return np.matmul(matrix_power(np.matmul(A,A), n//2),A)                   
    except:
        return "you should enter a square matrix."

def fibonacci_power(n):
    """
    fibonacci sequence with each term computed the matrix power method.
    """

    if n<0 or type(n) != int:          # in case some other thing were typed
        return "n should be a non-negative integer."
    if n == 0:                         #set up base for recursion
        return 0
    if n == 1: 
        return 1
    a = np.array([[1,1],[1,0]])         #how one vector is evolved into the next
    An= matrix_power(a,n-1)         
    x0 = np.array([[1],[0]])            #set up base
    xn = np.matmul(An,x0) #computing
    return xn[0,0]

table_data = [['recursive', 'iter', 'power']]
for i in range(1,31):
    table_data = table_data + [[fibonacci_recursive(i),fibonacci_iter(i),fibonacci_power(i)]]
for row in table_data:
    print("{: >20} {: >20} {: >20}".format(*row))
    
print('''
    There's finite amount of operations in one 2x2 matrix operation, and there's in total log2(n)+#(n)-1 matrix multiplications, so O(log(n)) here.
    For each n, there's one more matrix multiplication with x_0, but that's still negligible. So There's in total O(\sum_{i=1}^N logN )= O(N logN) operations (by integration).
    When n is super large, there might be some issue with multiplications in the matrix multiplication since it scales up the minimal error.
    I think np.int64 is better since its more exact.
        ''')

if __name__ == '__main__':
    """
    this section of the code only executes when
    this file is run as a script.
    """
    def get_runtimes(ns, f): 
        """
        get runtimes for fibonacci(n)
        e.g.
        trecursive = get_runtimes(range(30), fibonacci_recusive)
        will get the time to compute each fibonacci number up to 29
        using fibonacci_recursive
        """
        ts = []
        for n in tqdm(ns):                 #recording time to compute each fibonacci number
            t0 = time.time()
            fn = f(n)
            t1 = time.time()
            ts.append(t1 - t0)

        return ts


    nrecursive = range(35) #defining the functions for plotting later
    trecursive = get_runtimes(nrecursive, fibonacci_recursive)

    niter = range(10000) 
    titer = get_runtimes(niter, fibonacci_iter)

    npower = range(10000) 
    tpower = get_runtimes(npower, fibonacci_power)

    plt.loglog(nrecursive,trecursive, label='recursive') #below are lines to actually do the graph
    plt.loglog(niter,titer, label='iter')
    plt.loglog(npower,tpower, label='power')
    plt.xlabel('n')
    plt.ylabel('time')
    plt.title('Time needed for different ways of finding Fibionacci numbers')
    plt.legend()
    plt.show()

print('''
    The recursive one grows up high with exponential rate since we have computed it to be exponential of A to n. So log log plot yields n on the right and log n on left, which is still exponential.
    The iterative one grows linearly since it's O(n^2) and the slope should be 2.
    The power method gives a slope weaker than log, but the computer need time to startup, so it stays the lowest.
        ''')
