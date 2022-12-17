# Answers

Put any answers to questions in the assignment in this file, or any commentary you don't include in the code.

This is a markdown file (the `.md` extension gives it away). If you have never used markdown before, check out [this short guide](https://guides.github.com/features/mastering-markdown/).

## Problem 0
A: 

At first, I guessed it to be normal. But after plotting I realized it's not. (all plot retained in ipynb)

![A1-normal](https://user-images.githubusercontent.com/114457056/196768776-e0eb7fad-56df-4d3a-82b7-c769d3944c15.jpg)

After discussion with classmates I used semicircle and it worked extremely well.

![A1](https://user-images.githubusercontent.com/114457056/196768778-cc37dece-2e09-4155-8fd7-4dd7b941c305.jpg)


B:

It is a normal distribution (as plotted).

![B1](https://user-images.githubusercontent.com/114457056/196768779-2ef9fd0b-d0be-417b-b96c-5e4b947005a6.jpg)


C:

It is a Gamma distribution and when fitted it works pretty well.

![C1](https://user-images.githubusercontent.com/114457056/196768782-98f32f59-4b6d-4f78-8c9d-68e61a361b1d.jpg)

D:


E: 

I plotted two version. The first is the one required. But it is obvious that there's outliers due to some near 0 singular values.

![E1](https://user-images.githubusercontent.com/114457056/196767843-c6c57883-0ac6-45c0-b8c6-8c49ad01e7bb.jpg)

I tried to manually discard near zero singular values, but that did not work out well.
So then I used a code I found on web (cited in the quoted part of the finding outlier function) to find outliers and remove them. 
Then, the second version of the plot is more readable.

![E2](https://user-images.githubusercontent.com/114457056/196767959-7e968cf1-5390-42ac-a1ca-dd1590d622f7.jpg)


## Problem 1
A:

Indeed the result is as expected (there are 1e-16 order residues).

![A2](https://user-images.githubusercontent.com/114457056/196813904-8043b357-18f0-488e-819f-74a4091ef5dd.jpg)


B:

I followed the instruction and printed the result and the difference of eigenvalues. 

![B2](https://user-images.githubusercontent.com/114457056/196813906-5e10f887-63f7-42a5-a6e0-17436b1efe10.jpg)

Indeed the result has two 0 at the right place and the eigenvalues are not changed.


C:

I did it for 10X10 in semilogy plot. It appears to be like a downward parabola, which means that the convergence is quadratic.

![BC](https://user-images.githubusercontent.com/114457056/196813907-1a8a6344-cff8-455c-8dea-40211bb8fa27.jpg)


D:
The time for 10x10 matrix is 0.03 if I apply the code above. This is not even worth doing optimization. 

But I did for 100x100 matrix and the result is around 130s:

![D2](https://user-images.githubusercontent.com/114457056/196813908-2eeea337-5c5a-4fc1-ba3d-5272d304e97b.jpg)

So I started doing optimizations.

The first thing I did was to move the symmrtric check out to the prob c function. This should help a little bit.

I have always used rows as inner loop so I don't have to modify that.

It's hard not to notice that I repeated the process of finding the largest element a huge amount of times, so I used Numba to jit that, which proves to work extremely well since I now can do the 100x100 matrix case with only around 2.4 seconds!

![D22](https://user-images.githubusercontent.com/114457056/196813909-a78ebe0f-b497-4741-94c4-d6de26c2776f.jpg)

I think that maybe I can also jit the arctan part to speed up more, and after noticing I cannot compile matrix multiplication due to some type error, I just compiled the construction of the multiplier matrix, which in the end get me to around 2.1 second. 

![D23](https://user-images.githubusercontent.com/114457056/196813903-679fc5c2-04b2-4a1a-9576-20b8fd6c9109.jpg)

I feel like that's about enough work to be done here.


E:

Yes we can. Since A is symmetric, the singular values are nothing but the absolute values of eigenvalues of the matrix. 

So since we are preserving eigenvalues in each step (as verified in B), we end up with a diagonal matrix (with epsilon errors on non-diagonal entries of course). The eigenvalues of that matrix is just the diagonal terms, and the singular values are just the absolute values of those. So indeed, our method gets us the singular value and eigenvalues in 2.1 seconds, which is a lot slower than the precompiled version.

Well for general matrices, we can of course get the singular values this way by inputing A^\*A which is hermitian. But we'd not know the eigenvalues since only the norm of them is encoded. This might tell us if a matrix is going to 0 or infinity when multiplied by itself n times. For the columns of the SVD decomposition, if the matrix is square then we know it's just piling up each rotation. I think just multiplying each W will yield the result since it is pretty well constructed.

## Feedback
