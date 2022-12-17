# Homework 0 - Basics

In this homework, you'll practice some basic tools you'll use throughout the course.

Please put any written answers in [`answers.md`](answers.md)

You may need to use conda to install `tqdm` to run the starter code for problem 4.  You can run
```
conda install --file requirements.txt
```
to install the packages in [`requirements.txt`](requirements.txt)

## Important Information

### Due Date
This assignment is due Thursday, October 6 at 11:59 pm (midnight) Chicago time.

### Git
You will need to use basic `git` in this assignment.  See [this tutorial](https://github.com/caam37830-fall-2022/git-tutorial) if you have never used it before.  The basic commands to know are:
* `git clone`
* `git pull`
* `git add`
* `git commit`
* `git push`



### Grading Rubric

The following rubric will be used for grading.

|   | Autograder | Correctness | Style | Total |
|:-:|:-:|:-:|:-:|:-:|
| Survey    |   | /10 | /0 |  /10 |
| Problem 0 |   | /8  | /2  |  /10 |
| Problem 1 | /5 | /10  | /5  | /20 |
| Problem 2 | /10 | /12  | /3  | /25 |
| Problem 3 | /5 | /12  | /3  | /20 |
| Problem 4 |   |  /10 | /5  | /15 |

Correctness will be based on code (i.e. did you provide was was aksed for) and the content of [`answers.md`](answers.md).

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
The tests are in [`test.py`](test.py).  You do not need to modify (or understand) this code.

You need to pass all tests to receive full points.

## Survey (10 points)

Write a script in `survey.py` which outputs answers to the following questions.

1. Your name
2. Your program / year
3. What are your academic interests? (research/coursework)
4. What programming languages do you have experience with?
5. What is your experience with Python?  (is is ok to have no experience)
6. What time zone are you in? (Chicago is UTC -5)
7. What is something you would like to learn in this course?
8. Do you have any questions or concerns you would like to share?


Feel free to make this a reverse [code golf](https://en.wikipedia.org/wiki/Code_golf), e.g. make functions/variables to "help" you print your answers in an unnecessarily complicated way.  There are no rules, and points are based on completion (you can get some bonus style points for creativity in your code).  However, your output should look like the following when run from a terminal
```
$ python survey.py
1. <answer to question 1>
2. <answer to question 2>
...
```


## Problem 0 (10 points)
Implement the following [Fizzbuzz](https://imranontech.com/2007/01/24/using-fizzbuzz-to-find-developers-who-grok-coding/) program:

Write a python script [`fizzbuzz.py`](fizzbuzz.py) which prints the numbers from 1 to 100, but for multiples of 3 print "fizz" instead of the number, for multiples of 5 print "buzz" instead of the number, and for multiples of both 3 and 5 print "fizzbuzz" instead of the number.

The first few lines of output should look like
```
1
2
fizz
4
buzz
```

## Problem 1 (20 points)

The [Egyptian multiplication algorithm](https://en.wikipedia.org/wiki/Ancient_Egyptian_multiplication) (or Russian peasant algorithm) computes the product `a * n` using repeated additions. The algorithm uses the basic rule `a * n = (a + a) * (n // 2)` if `n` is even, and `a * n = (a + a) * (n // 2) + a` if `n` is odd.  Recall `//` is integer (floor) division.

For the purposes of this problem, you can assume `n` will always be an integer with `n >= 0`.

You can view [`egyptian.py`](egyptian.py) for a basic Python implementation.

How many additions are done in the Egyptian multiplication algorithm?  Your answer should be in terms of `log2(n)` and the number of non-zero bits of `n` (when `n` is represented as a binary integer), denoted `#(n)`.  Hint: how many additions are done at each level of recursion, and how many levels of recursion are there?

Write a function called `power(a, n)` to compute `a**n` via repeated multiplications by adapting the Egyptian multiplication algorithm.  Implement this function in the file `egyptian.py`.  Have the script print a few interesting powers, including `3**3`, `4**4` and `5**3`.

In general, you can use the Egyptian multiplication algorithm on any associative operation (meaning `a` can be an element of any semi-group) c.f. Stepanov and Rose, "From Mathematics to Generic Programming."

## Problem 2 - Fibonacci Numbers (25 points)

The [Fibonacci sequence](https://en.wikipedia.org/wiki/Fibonacci_number) is defined as a linear recursion

`F_0 = 0, F_1 = 1, F_n = F_{n-1} + F_{n-2}`

The first few numbers in the sequence are 0, 1, 1, 2, 3, 5, 8, 13, 21, ...

One method to compute the Fibonacci number `F_n` is to compute each number recursively. In pseudocode:
```
fibonacci(n)
	return fibonacci(n-1) + fibonacci(n-2)
```

Write a recursive function in Python to compute `fibonacci(n)` - name this function `fibonacci_recursive`.  Put in the appropriate checks for if `n=0` or `n=1`.

Another way to compute `fibonacci(n)` is through a simple iteration.  In pseudocode:
```
fibonacci(n):
	a = 0
	b = 1
	for n-1 iterations:
		a, b = b, a+b
	return b
```
Write a Python function named `fibonacci_iter` to implement this algorithm.


Print the first 30 Fibonacci numbers using both these functions.

Which of these algorithms do you expect to be asymptotically faster?  Why?  Full aysmptotic analysis of `fibonacci_recursive` is not trivial, and you do not need to do it.  Just give a high-level argument for which algorithm you expect to be asymptotically faster.

## Problem 3 - Numpy Basics (20 points)

We can re-write the Fibonacci recurrence using a matrix-vector product:
```
[F_n    ] = [1, 1] * [F_{n-1}]
[F_{n-1}]   [1, 0]   [F_{n-2}]
```
Let `x_n = [F_n, F_{n-1}]^T` above, and let `A` denote the matrix `[[1,1],[1,0]]` above.  Then we can write `x_n = A^(n-1) x_1`, where `x_1 = [1,0]`.

Write a Python function `fibonacci_power` that computes `F_n` using the above relation, using numpy for vectors and matrices.  Use a modified version of your adaptation of the Egyptian algorithm in part 1 to compute the power `A^n`.

Print the first 30 Fibonacci numbers using this function.  Compare to your answer from problem 3.

What is the asymptotic number of operations done in this algorithm?  Addition and multiplication both count as single operations (there are circuits on a CPU to do either in one clock cycle). How does this compare to the algorithms in problem 2?

What are potential issues you might run into with large values of `n` in this algorithm?  (You don't need to address them in code.)  Do you think it would be better to use `np.float64` or `np.int64` in your arrays for large values of `n`?  You can specify the data type of an array using the keyword `dtype`, for example `x = np.array([1,2,3], dtype=np.float64)`.

## Problem 4 - PyPlot Basics (15 points)

Use PyPlot to create a plot of the run times of your three functions to compute Fibonacci numbers.  

There is starter code at the end of [`fibonacci.py`](fibonacci.py) which will do the timing tests for you.

* Use a log-log scale for your plot.
* There should be three lines.
* Your plot should have a legend.
* Your plot should have labels for its x and y axes.
* Your plot should have a title
* save your plot as `fibonacci_runtime.png`

embed the image in `answers.md` and give a short interpretation of what you see.

## Feedback

If you'd like share how long it took you to complete this assignment, it will help adjust the difficulty for future assignments.  You're welcome to share additional feedback as well.
