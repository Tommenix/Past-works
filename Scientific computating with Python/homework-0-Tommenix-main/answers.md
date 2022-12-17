# Answers

Put any answers to questions in the assignment in this file, or any commentary you don't include in the code.

This is a markdown file (the `.md` extension gives it away). If you have never used markdown before, check out [this short guide](https://guides.github.com/features/mastering-markdown/).

## Problem 0
You don't need to say anything here.  Just complete [`fizzbuzz.py`](fizzbuzz.py).

## Problem 1
The number of total addition is ⌊log2(n)⌋+#(n)-1, where ⌊x⌋ means the biggest integer smaller or equal to x.

The reason for this is because each time we enter the egyptian multiplication function, we do 1 addition if n is even and 2 if n is odd, then we let our new n (binary number) be the number with the last digit removed from the old n.

To see why, we simply notice that taking out 1 (when n is odd) and taking out none both makes the last digit 0, and to take the half of an even number in binary operation is to take off the last 0, which is because when timing one half, we minus one on the exponents of powers of 2.

And since for each 1 (except the first) in the binary form of n, we need to do one additional addition to remove it, so there are in total #(n)-1 addition.

As for how many times we will be taking off one digit, it is (the number of digit of n -1) =  ⌊log2(n)⌋.

Therefore the answer is ⌊log2(n)⌋+#(n)-1.

## Problem 2
![微信图片_20221004194317](https://user-images.githubusercontent.com/114457056/193955608-62b57fd4-9450-46f9-b8dc-a145f52002e1.jpg)


## Problem 3
![2](https://user-images.githubusercontent.com/114457056/193955729-3efebfc3-77f2-45fb-913d-a36e5fa21e63.jpg)

Also, if I actually compute the power method to the 10000-th term, the result is a negative number which I think is because that 64 digit is way to less for an exponential growth function. But this is still better than 'nan' if we do it in float64(and even there is a number it's far wrong).


## Problem 4
<img width="435" alt="Screen Shot 2022-10-04 at 7 52 36 PM" src="https://user-images.githubusercontent.com/114457056/193956061-37b3126d-74f4-45ed-9201-8c36197baaf2.png">


The recursive one grows up high with exponential rate since we have computed it to be exponential of A to n. So log log plot yields log t = ce^{log n} on left, which is still exponential.

The iterative one grows linearly since it's O(n) and the slope should be 1.

The power method gives a curve like log n, but the computer need time to startup, so it stays the lowest in the farther side.

## Feedback
