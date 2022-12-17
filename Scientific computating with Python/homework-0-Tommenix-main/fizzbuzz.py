"""
fizzbuzz

Write a python script which prints the numbers from 1 to 100,
but for multiples of 3 print "fizz" instead of the number,
for multiples of 5 print "buzz" instead of the number,
and for multiples of both 3 and 5 print "fizzbuzz" instead of the number.
"""
m = list(range(1,101))           
for n in range(1,101):
    if n % 3 == 0 :             #finding all multiples of 3
        if n % 5!=0:            #labeing all multiples of 3 but not multiples of 5 fizz
            m[n-1]="fizz"
        else:                   #I think in this way the program only go through each multiple of 3 once.
            m[n-1]="fizzbuzz" 
    elif n %5==0:               #labeling all multiples of 5 that are not multiples of 3 buzz
        m[n-1] = "buzz"
for i in m:
    print(i)
