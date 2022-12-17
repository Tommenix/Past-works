# Homework 2 - Linear Algebra

Please put any written answers in [`answers.md`](answers.md)

Reminder: you can embed images in markdown.  If you are asked to make a plot, save it as a `png` file, commit it to git, and embed it in this file.

Please complete the code [`script.ipynb`](script.ipynb) to finish all tasks.

You can run
```
conda install --file requirements.txt
```
to install the packages in [`requirements.txt`](requirements.txt)

## Important Information

### Due Date
This assignment is due Friday, October 21 at midnight Chicago time.

### Grading Rubric

The following rubric will be used for grading. There is no autograder for this assignment.

|   | Autograder | Correctness | Style | Total |
|:-:|:-:|:-:|:-:|:-:|
| Problem 0 |  |   |  | /50 |
| Part A |  | /8 | /2 | /10 |
| Part B |  | /8 | /2 | /10 |
| Part C |  | /8 | /2 | /10 |
| Part D |  | /8 | /2 | /10 |
| Part E |  | /8 | /2 | /10 |
| Problem 1 |  |   |  | /50 |
| Part A |  | /2 | /3 | /5 |
| Part B |  | /15 | /5 | /20 |
| Part C |  | /10  | /5 | /15|
| Part D |  |  |  | /10 |
| Part E (bonus)|  |  |  | /10 |


Correctness will be based on code (i.e. did you provide was was aksed for) and the content of [`answers.md`](answers.md).

To get full points on style you should use comments to explain what you are doing in your code and write docstrings for your functions.  In other words, make your code readable and understandable.

### Autograder

For this homework, there is no autograder to test your code. Give answers based on your own experiments.


## Problem 0 - Scipy linear algebra (50 points)

In this question the goal is to gain some familiarity with scipy's linear algebra routines by doing some experiments on random matrices.

### Part A (10 points)
Construct one hundred `1000x1000` **symmetric** matrices whose entries are independent standard normal random variables (note that for your matrices, `A[i,j] = A[j,i]`). Using scipy, compute the eigenvalues of each of these matrices and plot a histogram of **all** of the eigenvalues of all `100` matrices. In other words, you should obtain `100,000` eigenvalues in total from your `100` random matrices. Try to guess what the true distribution of eigenvalues is by looking at the histogram. Plot the error in the fit of your guess and the empirical distribution obtained from the histogram. 

Now redo the experiments with `200x200`, `400x400`, `800x800`, and `1600x1600` matrices. How do any parameters of your model scale with the size of the matrices?

Hints: Do not create all the matrices and then process them. Use a loop to sample a matrix and find its eigenvalues. Do not store all the matrices (just the eigenvalues!).

For the histogram plots: look at the documentation for matplotlib.pyplot.hist, ie. in ipython, type

`import matplotlib.pyplot as plt`

`plt.hist()`

In your response, include the code you used to run your experiments, plots of the resulting histograms, model, and fits. In your write up include your guess at the model.

### Part B (10 points)
Construct one thousand `200x200` **symmetric** matrices whose entries are independent standard normal random variables. Compute the largest eigenvalue of each matrix and plot the histogram. Can you guess what form the distribution of the largest eigenvalue takes? Include the code you ran for your experiments and a histogram of the results.

### Part C (10 points)
Construct one thousand `200x200` **symmetric** matrices whose entries are independent standard normal random variables. Plot a histogram of the largest gap between **consecutive** eigenvalues (if they are sorted in increasing order). Can you guess what form the distribution of the largest eigenvalue takes? Include the code you ran for your experiments and a histogram of the results.

### Part D (10 points)
Using scipy, investigate the behavior of the singular values of symmetric random matrices. Plot a histogram of your results for various sizes (`200,400,800,1600`) using `100` trials each.

### Part E (10 points)
Plot histograms of the condition number of random matrices (the largest singular value of a matrix divided by its smallest singular value).



## Problem 1 - Jacobi rotation （50 points）
In this question the goal is to gain some familiarity with a method for finding the eigenvalues of a matrix (Jacobi rotations), and doing common operations on matrices with Python.

### Part A (5 points)
For a `2x2` symmetric matrix `C`, defined by

`[[a, b]
[b,c]]`

Define `theta` by 
`theta = arctan2(b,(c-a)/2)/2`
and the matrix `R` by

`[[cos(theta),-sin(theta)]
[sin(theta), cos(theta)]]`

Confirm that `R*C*(R.T)` is diagonal. You do not need to submit anything for this part. For `arctan2`, look up the documentation for `numpy.arctan2` (or `np.arctan2`).

### Part B (20 points)
Write a python function that takes in a general symmetric `nxn` array `A` and does the following:

***************
i) Finds the largest element (in absolute value) not on the diagonal, and its position. If there are multiple which are identical, pick any.

ii) If `i` is the row and `j` is the column of the entry in part (A) (note that `i` is not equal to `j`), then consider the matrix

`[[A[i,i],A[i,j]],
[A[j,i],A[j,j]]`

Use the answer in part (A) to construct the appropriate `R` for this 2x2 matrix.

iii) Now, construct the matrix `W` by taking the `nxn` identity matrix and replacing the entries at `(i,i)`, `(i,j)`, `(j,i)` and `(j,j)` by `R[0,0]`, `R[0,1]`, `R[1,0]` and `R[1,1]`, respectively. Here `R` is the 2x2 matrix you just constructed.

iv) Output `W*A*W.T`

*******************

Confirm that the output of your function is a `nxn` matrix with `0` in the `(i,j)` and `(j,i)` th entries. Also, confirm that the eigenvalues have not changed.

Submit your code, as well as the output of your code run on the `3x3` matrix

`[[1,2,3],
[2,4,5],
[3,5,6]]`

### Part C (15 points)
Write a function that takes in a general symmetric `nxn` array `A` and a tolerance tol (a floating point number), and does the following:
While the largest element (in absolute value) on the off-diagonal is greater than the tolerance, it calls the function you wrote in part (B) on the matrix `A`, and then replaces `A` by the output. 
Once the loop has terminated, output the result.

Using `3x3` symmetric random matrices (whose entries are independent standard normal random variables) confirm that when your function terminates, the result is a diagonal matrix whose entries are the eigenvalues of the original matrix.

Submit your code, as well as the output of your function when the input is 

`[[1,2,3],
[2,4,5],
[3,5,6]]`

and the tolerance is `10^(-8)`.

For `10x10` matrices plot the largest off-diagonal entry as a function of number of rotations applied. Estimate the order of convergence.

Submit a plot showing the error as a function of number of rotations and in your write up include some description of your estimate of convergence.

### Part D (10 points)
Once you have a working code for finding the eigenvalues using the method described in part (A)-(C), speed it up as much as possible using the techniques we discussed in class (numba, vectorization, loop orders, vectorization, etc.).

Hint for one possible acceleration: rather than form the matrix `W` and do two matrix-matrix multiplications, you can just operate on the `i`th and `j`th rows and the `i`th and `j`th columns.

Use timings to show what improvements, if any, you get. This is an open-ended question and there is no "right" answer. Instead, it is an opportunity for you to play around, and get experience trying to speed up a code. If nothing you try works, then that's ok! Write up what you tried.

### Part E: Challenge (bonus 10 points)
Can you calculate the singular values this way? What about the eigenvectors?



## Feedback

If you'd like share how long it took you to complete this assignment, it will help adjust the difficulty for future assignments.  You're welcome to share additional feedback as well.
