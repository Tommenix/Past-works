# Homework 1 - Object Oriented Programming and Functions

In this homework you'll create a library of function classes which are related by class inheritance.  This is a common design pattern in a variety of packages, such as SciPy, SymPy, and PyTorch.  This is a group assignment, so you will practice working in a shared repository.

You can run
```
conda install --file requirements.txt
```
to install the packages in [`requirements.txt`](requirements.txt)

Please put plots and written answers in the Jupyter notebook [`answers.ipynb`](answers.ipynb)

## Important Information

### Due Date
This assignment is due Thursday, October 13 at midnight Chicago time.

You should make sure you have pushed your completed work to GitHub by this time.


### Group Assignment

This is a group assignment.  You will share a `git` repository, and work will be shared.  Evaluation will be done on the group repository, but everyone should understand the code and answers.  You can (and should) look at code that any group member writes and look for ways to improve it, for example, by adding comments, improving docstrings, or catching bugs. You can talk to people in other groups, but shouldn't share code/answers.

You should communicate with your group early on to discuss who will do what, and check in regularly to make sure work is being completed.  You may wish to video chat or use Slack to troubleshoot or debug together.  It is perfectly fine to write code collectively e.g. get on video and "pair program".  Work should be divided roughly equally, and you'll be asked to say who was primarily responsible for what (you don't have to be super detailed).

If you having trouble communicating with members of your group e.g. someone is just not responding to emails at all, please let the course staff know.  In extreme cases, group members may be re-assigned.

If you are new to using git, see [collaboriating with git](https://github.com/caam37830-fall-2022/git-tutorial/blob/main/collaborating.md).  It is recommended to use a branch for each team member.

### Grading Rubric

The following rubric will be used for grading.

|   | Autograder | Correctness | Style | Total |
|:-:|:-:|:-:|:-:|:-:|
| Problem 0 |   |   |   |  /55 |
| Part A    |    |  /3   | /2  |  /5 |
| Part B    | /2 |  /2   | /1  |  /5 |
| Part C    | /6 |  /7   | /2  |  /15 |
| Part D    | /5 |  /7   | /3  |  /15 |
| Part E    | /2 |  /2   | /1  |  /10 |
| Part F    |    |  /5   |     |  /5 |
| Problem 1 |   |   |   | /25 |
| Part A    | /3 |  /8   | /4  |  /15 |
| Part B    | /2 |  /6   | /2  |  /10 |
| Problem 2 |   |  |   | /20 |
| Part A    | /4  |  /4   | /2  |  /10 |
| Part B    |   |  /8   | /2  |  /10 |

Correctness will be based on code (i.e. did you provide was was asked for) and the content of [`answers.ipynb`](answers.ipynb).

To get full points on style you should use comments to explain what you are doing in your code and write docstrings for your functions.  In other words, make your code readable and understandable.

### Autograder

You can run tests for the functions you will write in problems 1-3 using either `unittest` or `pytest` (you may need to `conda install pytest`).  From the command line:
```
python -m unittest test.py
```
or
```
pytest test.py
```
The tests are in [`test.py`](test.py).  You should not modify this code, and don't need to understand it, but you can read it to see what is being tested.  The output of the autograder should tell you where you have issues that need to be resolved.

You need to pass all tests to receive full points.

Keep in mind that the autograder is just doing some basic sanity checks, which is why there are also separate points for "correctness".  However, if you are passing the autograder checks, chances are you're on your way to success.


## Problem 0

In this problem, you will implement a library of function classes related by class inheritance.

We want this library of functions to behave nicely with NumPy arrays.  The `evaluate` method should be compatible with NumPy element-wise operations.

Put all your class and function definitions in [`functions.py`](functions.py), and use [`answers.ipynb`](answers.ipynb) for written answers, plots, or displaying output of expressions.

### Part A: Finish the parent class (5 points)

The parent class for this library is `AbstractFunction`.  Implement
Complete the `plot` method in this class.  You should return the output of the `plot` function in PyPlot.

Use the `evaluate` method on the keyword argument `vals` to make a plot of the function (evaluate will be implemented in child classes).  Pass any additional keyword arguments to the `plot` function in PyPlot.

Note that we will need to implement child classes:
* `Compose` (composition of functions)
* `Sum` (sum of functions)
* `Product` (product of functions)
* `Scale` (scaling function)
* `Power` (function that applies a power)

Some additional notes on methods which will be defined in child classes:
* `evaluate` - evaluate the function on a value, or element-wise on a numpy array
* `derivative` - return another child class of `AbstractFunction` as the derivative of the function.

The `__call__` method allows you to use an object as a function in Python.  For example, if `f` is some derived class of `AbstractFunction`, `f(x)` will use the `__call__` method.  You can see that this method does different things depending on what the type of `x` is.

Test your function by plotting `5*x^2 + 3*x + 1` using the provided Polynomial class
```python
p = Polynomial(5,3,1)
p.plot(color='red')
```

### Part B: Implement Scale and Constant functions (5 points)

A class for Polynomial functions, named `Polynomial` has been provided in `functions.py`.  This is more complex than anything you'll need to do.

When implementing derived classes, you can access methods of the parent class (even as you specialize them) using `super()`.  See the definition of `Affine` for an example of how we can implement the `__init__` method for functions of the form `a*x + b` using the `__init__` method of `Polynomial`.

Implement classes `Scale` and `Constant` as child classes of `Polynomial`.  You should only need to implement the`__init__` method for both classes.

`Scale(a)` should be equivalent to the polynomial `a * x + 0`

`Constant(c)` should be equivalent to the polynomial `c`

Make plots of `Scale(2)` and `Constant(1)` using the `plot` method.

### Part C: Implement Compose, Product, and Sum functions (15 points)

Implement child classes of `AbstractFunction`:
1. `Compose`, where `Compose(f, g)(x)` acts as `f(g(x))`
2. `Product`, where `Product(f, g)(x)` acts as `f(x) * g(x)`
3. `Sum`, where `Sum(f,g)(x)` acts as `f(x) + g(x)`

You should provide implementations for:
* `__init__`
* `__str__`
* `__repr__`
* `derivative` (recall chain rule and product rule)
* `evaluate`

When implementing `__str__`, place `{0}` where indeterminates in the function would go.
You can look at the implementation of `Polynomial` for examples of this.  If you call the `format` method on the string, you need to escape the sequence (so it isn't formatted), by enclosing in an extra set of braces: `"{{0}}"`

Note that functions don't generally simplify nicely, e.g. removing terms with zero coefficients, combining the composition of powers to a single power etc.  In order to do this, you would need to encode the rules to simplify expressions, which would add more complexity to this assignment (so you don't need to do it).  Some of this is done in the implementation of `Polynomial` (in `__add__` and `__mul__`) if you want to take a look.

Make a plot of `Compose(Polynomial(1,0,0), Polynomial(1,0,0))`.  What is the equivalent function expressed as a `Polynomial`?

### Part D: Implement Power (and some other functions) (15 points)

Implement additional classes of `AbstractFunction`
1. `Power`: `Power(n)(x)` acts as `x**n` (n can be negative, or non-integer)
2. `Log`: `Log()(x)` acts as `np.log(x)`
3. `Exponential`: `Exponential()(x)` acts as `np.exp(x)`
4. `Sin`: `Sin()(x)` acts as `np.sin(x)`
5. `Cos`: `Cos()(x)` acts as `np.cos(x)`

You should provide implementations for:
* `__init__`
* `__str__`
* `__repr__`
* `derivative`
* `evaluate`

When implementing `__str__`, place `{0}` where indeterminates in the function would go.
You can look at the implementation of `Polynomial` for examples of this.

Make plots of each of these functions (for Power, use `Power(-1)`) - you may need to adjust the domain of your plot function using the `vals` argument.


### Part E: Symbolic Functions (10 points)

Implement a class for symbolic functions.  The data for a symbolic function is a string, which is the name of the function.  For example, we should be able to define a symbolic function
```python
f = Symbolic('f')
```
The string method should output:
```python
str(f)
"f({0})"
```
The evaluate method should just print a string with whatever the input is, so when the `__call__` method is used, we have
```python
f(5)
"f(5)"
```
The derivative method should add an apostrophe to the end of the name:
```python
f.derivative()
Symbolic("f'")
```

Note that Symbolic functions won't be compatible with the `plot` method defined in the `AbstractFunction` class.

In `answers.ipynb`, take the derivative of the product of two symbolic functions to get an expression for product rule.

### Part F: Use the Module (5 points)

Use the module you just wrote to answer the following questions.

1. What is the derivative of `5x^2 + 3x + 1`?
2. Derive a rule for the derivative of $f = g/h$ using symbolic functions.  Does this reduce to quotient rule?
3. What is the derivative of `sin(x)**2`?
4. What is the second derivative of `exp(5*x)`?


## Problem 1 (25 points)

### Part A: Newton's method for root finding (15 points)
Implement Newton's method for root finding using the call signature
```python
def newton_root(f, x0, tol=1e-8):
    """
    find a point x so that f(x) is close to 0,
    measured by abs(f(x)) < tol

    Use Newton's method starting at point x0
    """
```
The function should assume that `f` is an `AbstractFunction`, and that `x` is a real number.  Put in a type check to verify that `f` is an `AbstractFunction` but it is not `Symbolic`.

Implement this function in `functions.py`

Find a root of `sin(exp(x))` starting at `x0=1.0`. Make a plot of `sin(exp(x))` and visualize the root using `plt.scatter` on the same plot.

### Part B: Newton's method for finding extrema (10 points)

Newton's method can also be used to find a local extremum (maximum or minimum) of a function.  The observation is that `f` is at a local maximum or minimum if its derivative is zero (we'll assume that the second derivative is non-zero which avoids some potential problems).  Thus, finding roots of a derivative finds maxima or minima of a function.

Implement a function that finds a local extremum for a function using the call signature
```python
def newton_extremum(f, x0, tol=1e-8):
    """
    find a point x which is close to a local maximum or minimum of f,
    measured by abs(f'(x)) < tol

    Use Newton's method starting at point x0
    """
```
Again, assume that `f` is an `AbstractFunction`, and `x` is a real number.

Implement this function in `functions.py`

Find a minimum or maximum of `sin(exp(x))` starting at `x0=0.0`. Make a plot of `sin(exp(x))` and visualize the extremum using `plt.scatter`.

## Problem 2 - Taylor Series (20 points)

### Part A - add a method to AbstractFunction (10 points)
Recall the Taylor series of a function `f` centered at `x0` is the polynomial
```
Tf = f(x0) + f'(x0)(x - x0) + 1/2 * f''(x0)(x - x0)**2 + ...
```
Add a method `taylor_series` to the `AbstractFunction` base class
```python
def taylor_series(self, x0, deg=5):
    """
    Returns the Taylor series of f centered at x0 truncated to degree k.
    """
```
The return type should be another `AbstractFunction`.

### Part B - use your Taylor series (10 points)

Make a plot that displays `sin(x)` as well as its degree-k taylor series for `k in [0,1,3,5]` on the interval `[-3,3]`

Your plot should include labels for each line displayed, as well as a legend.

## Feedback

If you'd like share how long it took you to complete this assignment, it will help adjust the difficulty for future assignments.  You're welcome to share additional feedback as well.
