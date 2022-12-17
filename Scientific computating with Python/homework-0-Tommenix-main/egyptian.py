"""
Egyptian algorithm
"""
a = """
    The number of total addition is [log2(n)+#(n)-1], where [x] means the biggest integer smaller or equal to x.
    The reason for this is because each time we enter the egyptian multiplication function, we do 1 addition if n is even and 2 if n is odd, then we let our new n (binary number) be the number with the last digit removed from the old n.
    To see why, we simply notice that taking out 1 (when n is odd) and taking out none both makes the last digit 0, and to take the half of an even number in binary operation is to take off the last 0, which is because when timing one half, we minus one on the exponents of powers of 2.
    And since for each 1 (except the first) in the binary form of n, we need to do one additional addition to remove it, so there are in total #(n)-1 addition.
    As for how many times we will be taking off one digit, it is (the number of digit of n -1) =  [log2(n)].
    Therefore the answer is [log2(n)]+#(n)-1.
"""
print(a)

def egyptian_multiplication(a, n):
    """
    returns the product a * n
    assume n is a nonegative integer
    """
    def isodd(n):
        """
        returns True if n is odd
        """
        return n & 0x1 == 1

    if n == 1:          #special initial case
        return a
    if n == 0: 
        return 0

    if isodd(n):           #core of the algorithm,explained in problem sets
        return egyptian_multiplication(a + a, n // 2) + a
    else:
        return egyptian_multiplication(a + a, n // 2)


if __name__ == '__main__':
    # this code runs when executed as a script
    for a in [1,2,3]:                #try for some interesting numbers
        for n in [1,2,5,10]:
            print("{} * {} = {}".format(a, n, egyptian_multiplication(a,n)))


def power(a, n):
    """
    computes the power a ** n
    assume n is a nonegative integer
    """
    '''
    This function computes the answer of a to the n-th degree, where n should be a non-negative integer, 
    with the method analogous to the Egyptian multiplication algorithm for multiplication.
    '''
    try:                                                                          #want n to be at least a number
        if n <0 or type(n) != int:                                               # want n to be a non negative integer
            return "n should be a non negative integer"                          #give some instructions on what n should be
        if n == 0:                 # define 0-th degree to be 1
            return 1 
        if n == 1:                 #this is to end the recursion when the degree has been simplified to 1, since a**1=a
            return a
        if n % 2 == 0:             #to put half of the degree in the base, when n is even
            return power(a*a, n//2)
        return power(a*a, n//2)*a           #to put almost half of the degree in the base, when n is even
    except: 
        return "n should be a non negative integer"
    
if __name__ == '__main__':
    # this code runs when executed as a script
    for a in [1,2,3,0,3.14]:                   #try for some interesting numbers
        for n in [1,2,5,10,0]:
            print("{} ** {} = {}".format(a, n, power(a,n)))
    print("{} ** {} = {}".format(3,3, power(3,3)))              #required in problem
    print("{} ** {} = {}".format(4,4, power(4,4)))              #required in problem
    print("{} ** {} = {}".format(5,3, power(5,3)))              #required in problem
