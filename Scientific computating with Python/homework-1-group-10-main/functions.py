#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import numbers

class AbstractFunction:
    """
    An abstract function class
    """

    def derivative(self):
        """
        returns another function f' which is the derivative of x
        """
        raise NotImplementedError("derivative")


    def __str__(self):
        return "AbstractFunction"


    def __repr__(self):
        return "AbstractFunction"


    def evaluate(self, x):
        """
        evaluate at x
        assumes x is a numeric value, or numpy array of values
        """
        raise NotImplementedError("evaluate")


    def __call__(self, x):
        """
        if x is another AbstractFunction, return the composition of functions
        if x is a string return a string that uses x as the indeterminate
        otherwise, evaluate function at a point x using evaluate
        """
        if isinstance(x, AbstractFunction):
            return Compose(self, x)
        elif isinstance(x, str):
            return self.__str__().format(x)
        else:
            return self.evaluate(x)


    # the rest of these methods will be implemented when we write the appropriate functions
    def __add__(self, other):
        """
        returns a new function expressing the sum of two functions
        """
        return Sum(self, other)


    def __mul__(self, other):
        """
        returns a new function expressing the product of two functions
        """
        return Product(self, other)


    def __neg__(self):
        return Scale(-1)(self)


    def __truediv__(self, other):
        return self * other**-1


    def __pow__(self, n):
        return Power(n)(self)


    def plot(self, vals=np.linspace(-1,1,100), **kwargs):
        """
        plots function on values
        pass kwargs to plotting function
        """
        raise NotImplementedError("plot")
        
    def taylor_series(self, x0, deg=5):
        """
        Returns the Taylor series of f centered at x0 truncated to degree k.
        """
        return Taylor(self,x0,deg)


# In[2]:


class Polynomial(AbstractFunction):
    """
    polynomial c_n x^n + ... + c_1 x + c_0
    """

    def __init__(self, *args):
        """
        Polynomial(c_n ... c_0)
        Creates a polynomial
        c_n x^n + c_{n-1} x^{n-1} + ... + c_0
        """
        self.coeff = np.array(list(args))


    def __repr__(self):
        return "Polynomial{}".format(tuple(self.coeff))


    def __str__(self):
        """
        We'll create a string starting with leading term first
        there are a lot of branch conditions to make everything look pretty
        """
        s = ""
        deg = self.degree()
        for i, c in enumerate(self.coeff):
            if i < deg-1:
                if c == 0:
                    # don't print term at all
                    continue
                elif c == 1:
                    # supress coefficient
                    s = s + "({{0}})^{} + ".format(deg - i)
                else:
                    # print coefficient
                    s = s + "{}({{0}})^{} + ".format(c, deg - i)
            elif i == deg-1:
                # linear term
                if c == 0:
                    continue
                elif c == 1:
                    # suppress coefficient
                    s = s + "{0} + "
                else:
                    s = s + "{}({{0}}) + ".format(c)
            else:
                if c == 0 and len(s) > 0:
                    continue
                else:
                    # constant term
                    s = s + "{}".format(c)

        # handle possible trailing +
        if s[-3:] == " + ":
            s = s[:-3]

        return s


    def evaluate(self, x):
        """
        evaluate polynomial at x
        """
        if isinstance(x, numbers.Number):
            ret = 0
            for k, c in enumerate(reversed(self.coeff)):
                ret = ret + c * x**k
            return ret
        elif isinstance(x, np.ndarray):
            x = np.array(x)
            # use vandermonde matrix
            return np.vander(x, len(self.coeff)).dot(self.coeff)


    def derivative(self):
        if len(self.coeff) == 1:
            return Polynomial(0)
        return Polynomial(*(self.coeff[:-1] * np.array([n+1 for n in reversed(range(self.degree()))])))


    def degree(self):
        return len(self.coeff) - 1


    def __add__(self, other):
        """
        Polynomials are closed under addition - implement special rule
        """
        if isinstance(other, Polynomial):
            # add
            if self.degree() > other.degree():
                coeff = self.coeff
                coeff[-(other.degree() + 1):] += other.coeff
                return Polynomial(*coeff)
            else:
                coeff = other.coeff
                coeff[-(self.degree() + 1):] += self.coeff
                return Polynomial(*coeff)

        else:
            # do default add
            return super().__add__(other)


    def __mul__(self, other):
        """
        Polynomials are clused under multiplication - implement special rule
        """
        if isinstance(other, Polynomial):
            return Polynomial(*np.polymul(self.coeff, other.coeff))
        else:
            return super().__mul__(other)
        
    def plot(self, vals=np.linspace(-1,1,100), **kwargs):
        """
        plots function on values
        pass kwargs to plotting function
        """
        x = vals
        y = self.evaluate(vals)
        return plt.plot(x, y, **kwargs)


class Affine(Polynomial):
    """
    affine function a * x + b
    """
    def __init__(self, a, b):
        super().__init__(a, b)


# In[3]:


class Sum(AbstractFunction):
    def __init__(self, f, g): # store function f and g
        self.f = f
        self.g = g
    
    def __str__(self):
        return "{}+{}".format(self.f,self.g)
    
    def __repr__(self):
        return f"Sum({self.f},{self.g})"
    
    def derivative(self):
        p1 = self.f.derivative() # f'
        p2 = self.g.derivative() # g'
        return p1 + p2 # Sum rule: (f+g)' = f' + g'
    
    def evaluate(self, x):
        if isinstance(self.f, numbers.Number):
            return Polynomial(self.f) + self.g(x)
        elif isinstance(self.g, numbers.Number):
            return Polynomial(self.g) + self.f(x)
        elif isinstance(self.f, Symbolic) and isinstance(self.g,Symbolic):
            return "{}+{}".format(self.f(x),self.g(x))
        elif isinstance(self.f, str) and isinstance(self.g,str):
            return self.f+ ' + ' +self.g
        else:
            return self.f(x) + self.g(x)


class Product(AbstractFunction):
    def __init__(self, f, g): # store fucntion f and g
        self.f = f
        self.g = g
    
    def __str__(self):
        return "{}*{}".format(self.f,self.g)
    
    def __repr__(self):
        return f"Product({self.f},{self.g})"
    
    def derivative(self):
        p1 = self.f.derivative() # f'
        p2 = self.g.derivative() # g'
        if isinstance(self.f, Symbolic) and isinstance(self.g,Symbolic):
            return p1 * self.g + p2 * self.f
        else:
            return p1 * self.g + p2 * self.f # product rule: (fg)' = f'g + g'f
    
    def evaluate(self, x):
        if isinstance(self.f, numbers.Number):
            return Polynomial(self.f)* self.g(x)
        elif isinstance(self.g, numbers.Number):
            return Polynomial(self.g)* self.f(x)
        elif isinstance(self.f, Symbolic) and isinstance(self.g,Symbolic):
            ss = "{}*{}".format(self.f(x),self.g(x))
            return ss
        else:
            return self.f(x)* self.g(x)




class Compose(AbstractFunction):
    def __init__(self, f, g): ## need two functions f,g for composing functions
        self.f = f
        self.g = g
    
    def __str__(self):
        if isinstance(self.f, Symbolic) and isinstance(self.g, Symbolic):
            return self.f.evaluate(self.g)
        else:
            return "{}({})".format(self.f,self.g)
    
    def __repr__(self):
        return "Composing{} and {}".format(self.f,self.g)
    
    
    def derivative(self):
        p1 = self.f.derivative() # f'
        p2 = self.g.derivative() # g'
        return p1(self.g) * p2 # chain rule: (f(g))' = f'(g)*g'
    
    def evaluate(self,x):
        return self.f(self.g(x)) # f(g(x))
    
    def plot(self, vals=np.linspace(-1,1,100), **kwargs):
        """
        plots function on values
        pass kwargs to plotting function
        """
        x = vals
        y = self.evaluate(vals)
        return plt.plot(x, y, **kwargs)


# In[4]:


class Scale(Polynomial):
    def __init__(self, a):
        super().__init__(a, 0)
    def plot(self, vals=np.linspace(-1,1,100), **kwargs):
        """
        plots function on values
        pass kwargs to plotting function
        """
        x = vals
        y = self.evaluate(vals)
        return plt.plot(x, y, **kwargs)

class Constant(Polynomial):
    def __init__(self, a):
        super().__init__(a)
    def plot(self, vals=np.linspace(-1,1,100), **kwargs):
        """
        plots function on values
        pass kwargs to plotting function
        """
        x = vals
        y = self.evaluate(vals)
        return plt.plot(x, y, **kwargs)
        
class Symbolic(AbstractFunction):
    '''
    this class converts a string to a function.
    '''
    def __init__(self,f):
        self.f = f
        self.f2 = f
        
    def evaluate(self,x):
        return self.f2+"({})".format(x)
        
    
    def __str__(self):
        return "{}".format(self.f)+"({0})"
    
    def __repr__(self):
        return "Symbolic {}".format(self.f)
    
    def derivative(self):
        return Symbolic(self.f+'\'')

class Recenter(AbstractFunction):
    '''
    this is just to recenter the function at x-x0.
    '''
    def __init__(self,x0):
        self.x0 = x0
    
    def evaluate(self,x):
        return (x-self.x0)
    
class Taylor(AbstractFunction):
    '''
    computes the Taylor function
    '''
    def __init__(self, f, x0,n):
        self.f = f
        self.x0 = x0
        self.n = n
    
    def derivative(self):
        fp = self.f.derivative()
        return fp.taylor_series(self.x0,self.n)
        
    def evaluate(self,x):
        def factorial(a):
            facc =1
            for i in range(1,a+1):
                facc = facc*i
            return facc
        fp = self.f.derivative()
        g = Recenter(self.x0)
        h = self.f(self.x0)
        for i in range(1,self.n+1):
            h = h+ (1/factorial(i))*fp(self.x0)*(g(x)**i)
            fp = fp.derivative()
        return h
        


# In[5]:


class Power(AbstractFunction):
    """
   x^(n)
    """
    def __init__(self, n, f = Polynomial(1,0)): # store the degree of power
        self.n = n
        self.f = f
    
    def __str__(self):
        return f"({self.f})^({self.n})"
    
    def __repr__(self):
        return f"Power({self.f}^{self.n})"
    
    def evaluate(self,x): # x^(n)
        return self.f(x) ** float(self.n)
    
    def derivative(self):
        fp = self.f.derivative()
        nn = Polynomial(self.n)
        return  fp*nn*Power(self.n-1)(self.f)
    
    def plot(self, vals=np.linspace(-1,1,100), **kwargs):
        """
        plots function on values
        pass kwargs to plotting function
        """
        x = vals
        y = self.evaluate(vals)
        return plt.plot(x, y, **kwargs)


# In[6]:


def newton_root(f,x0,tol = 1e-8):
    fr = Newton_root(f,x0,tol)
    return fr(0)


# In[7]:


class Newton_root(AbstractFunction):
    """
    find a point x so that f(x) is close to 0,
    measured by abs(f(x)) < tol
    Use Newton's method starting at point x0
    """
    def __init__(self,f,x0,tol):
        self.f = f
        self.x0 = x0
        self.tol = tol
    
    def __str__(self):
        # IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # This function as a string returns the root!
        # If want to use root, use evaluate to destring!!!!!!
        if isinstance(self.f,AbstractFunction) and not isinstance(self.f,Symbolic):
            x=self.x0
            fx = self.f(x)
            fp = self.f.derivative()
            while (abs(fx) > self.tol):
                x = x - fx / fp(x)
                fx = self.f(x)
            return "{}".format(x)
        else:
            raise ValueError("Input should be a non-symbolic Abstract Function")
    
    def __repr__(self):
        return self.__str__()
    def evaluate(self,y):
        if isinstance(self.f,AbstractFunction) and not isinstance(self.f,Symbolic):
            x = self.x0
            fx = self.f(x)
            fp = self.f.derivative()
            while (abs(fx) > self.tol):
                x = x - fx / fp(x)
                fx = self.f(x)
            return x
        else:
            raise ValueError("Input should be a non-symbolic Abstract Function")


# In[8]:


def newton_extremum(f,x0,tol = 1e-8):
    fr = Newton_extremum(f,x0,tol)
    return fr(2)


# In[9]:


class Newton_extremum(AbstractFunction):
    """
    find a point x which is close to a local maximum or minimum of f,
    measured by abs(f'(x)) < tol
    Use Newton's method starting at point x0
    """
    def __init__(self,f,x0,tol):
        self.f = f
        self.x0 = x0
        self.tol = tol
    
    def __str__(self):
    # IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # This function as a string returns the extremum!
    # If want to use root, use evaluate to destring!!!!!!
        k = self.f.derivative()
        kk = newton_root(k, self.x0, self.tol)
        return "{}".format(kk)
    
    def __repr__(self):
        return self.__str__()
    
    def evaluate(self,x):
        k = self.f.derivative()
        kk = newton_root(k, self.x0, self.tol)
        return kk



# In[10]:


class Exponential(AbstractFunction):
    
    def __init__(self,f = Polynomial(1,0)):
        self.f = f
    
    def __str__(self):
        return "e^{}".format(self.f.__str__())
    
    def __repr__(self):
        return "exponential function of {}".format(self.f.__str__())
    def evaluate(self,x):
        if isinstance(self.f, numbers.Number):
            return np.exp(self.f)
        else:
            return np.exp(self.f(x))
    def derivative(self):
        return self.f.derivative()*Exponential(self.f)

    def plot(self, vals=np.linspace(-3,3,100), **kwargs):
        """
        plots function on values
        pass kwargs to plotting function
        """
        x = vals
        y = self.evaluate(vals)
        return plt.plot(x, y, **kwargs)


class Log(AbstractFunction):
    
    def __init__(self,f = Polynomial(1,0)):
        self.f = f
    
    def __str__(self):
        return "log({})".format(self.f.__str__())
    
    def __repr__(self):
        return "Logarithm function of {}".format(self.f.__str__())
    def evaluate(self,x):
        return np.log(self.f(x))
    def derivative(self):
        fp = self.f.derivative()
        return fp*(Power(-1)(self.f))
    def plot(self, vals=np.linspace(0,3,100), **kwargs):
        """
        plots function on values
        pass kwargs to plotting function
        """
        x = vals
        y = self.evaluate(vals)
        return plt.plot(x, y, **kwargs)

class Cos(AbstractFunction):
    
    def __init__(self,f = Polynomial(1,0)):
        self.f = f
    
    def __str__(self):
        return "cos{}".format(self.f.__str__())
    
    def __repr__(self):
        return "cos function of{}".format(self.f.__str__())
    def evaluate(self,x):
        return np.cos(self.f(x))
    def derivative(self):
        return (Polynomial(-1))*Sin(self.f)
    def plot(self, vals=np.linspace(-6,6,100), **kwargs):
        """
        plots function on values
        pass kwargs to plotting function
        """
        x = vals
        y = self.evaluate(vals)
        return plt.plot(x, y, **kwargs)
    
class Sin(AbstractFunction):
    
    def __init__(self,f = Polynomial(1,0)):
        self.f = f
    
    def __str__(self):
        return "sin{}".format(self.f.__str__())
    
    def __repr__(self):
        return "sin function of{}".format(self.f.__str__())
    def evaluate(self,x):
        return np.sin(self.f(x))
    def derivative(self):
        return Cos(self.f)
    def plot(self, vals=np.linspace(-6,6,100), **kwargs):
        """
        plots function on values
        pass kwargs to plotting function
        """
        x = vals
        y = self.evaluate(vals)
        return plt.plot(x, y, **kwargs)


# In[11]:


## problem 0 part a
p = Polynomial(5,3,1)
p.plot(color='red')


# In[12]:


## problem 0 part b Scale(2)
f = Scale(2)
f.plot(color = "black")


# In[13]:


## problem 0 part b Constant(1)
g = Constant(1) 
g.plot(color = "green")


# In[14]:


## problem 0 part c
f = Compose(Polynomial(1,0,0), Polynomial(1,0,0))
f.plot(color="blue")
## the equivalent function expressed as a polynomial is x^4 or Polynomial(1,0,0,0)


# In[15]:


## problem 0 part d power(-1)
f = Power(-1, Polynomial(1,0))
f.plot(color = "red")


# In[16]:


## problem 0 part d log(x)
f = Log(Polynomial(1,0))
f.plot(color = "red")


# In[17]:


## problem 0 part d exp(x)
f = Exponential(Polynomial(1,0))
f.plot(color = "green")


# In[18]:


## problem 0 part d sin(x)
f = Sin(Polynomial(1,0))
f.plot(color = "blue")


# In[19]:


## problem 0 part d cos(x)
f = Cos(Polynomial(1,0))
f.plot(color = "black")


# In[20]:


##problem 0 part f (1)
f = Polynomial(5,3,1)
g = f.derivative()
print(g)


# In[25]:


f = Symbolic('f')
g = Symbolic('g')
h = Symbolic('g/h')
t = h.derivative()
print(t)


# In[21]:


## problem 0 part f (3)
f = Sin(Polynomial(1,0))
g = Power(2,f)
h = g.derivative()
print(h)


# In[22]:


## problem 0 part f(4)
f = Exponential(Polynomial(5,0))
g = f.derivative()
h = g.derivative()
print(h)


# In[60]:


f = Sin()
vals = np.linspace(-3,3,300)
f1 = plt.plot(vals,f(vals), label = "sinx")
k = [0,1,3,5]
for j in range(4):
    g = f.taylor_series(0,k[j])
    lst = []
    lst2 = []
    for i in range(300):
        lst.append(0.02*i-3)
        lst2.append(float(g(0.02*i-3)))
    f2 = plt.plot(lst,lst2,label = "deg {}".format(k[j]))
plt.legend()
plt.show()


# In[ ]:




