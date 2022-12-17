"""
Print answers to the following

1. Your name
2. Your program / year
3. What are your academic interests? (research/coursework)
4. What programming languages do you have experience with?
5. What is your experience with Python?  (is is ok to have no experience)
6. What time zone are you in? (Chicago is UTC -5)
7. What is something you would like to learn in this course?
8. Do you have any questions or concerns you would like to share?
"""
l1=[]
l1.append('Tommenix Yu (Offical name Jiming Yu)')
l1.append("I am a first year graduate in MCAM program.")
l1.append('''My academic interest are in the analysis track, and I like to have more probability theory.''')
l1.append('I\'ve only had experience with python.')
l1.append('I had read some basics on my own, other than that nothing I guess.')
l1.append('UTC -5')
l1.append('''I\'d like to understand how many source codes are functioning and the clever codes in the reading. )
    Also, I\'m also wondering if I can write a python code with its output the same as its code.''')
l1.append('Not for now.')
for count, ele in enumerate(l1, 1):
    print (count, ".", ele)
