# Say "Hello, World!" With Python
print("Hello, World!")

# Python If-Else
#!/bin/python3
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n = int(input().strip())
    if n % 2 == 1:  # Odd case
        print("Weird")
    else:  # Even case
        if 2 <= n <= 5:
            print("Not Weird")
        elif 6 <= n <= 20:
            print("Weird")
        else:  # n > 20
            print("Not Weird")
#idk if the constraint 1<= n <=100 regards the test case or it's a request from the problem 

# Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)

# Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b)
    print(a/b)

# Loops
if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        print(i**2) # I was having problem since I was using ^ instead of ** for squaring
        
        

# Write a function
def is_leap(year):
    leap = False
    if year%4 ==0: #multiple of 4
        leap=True
        if year%100 ==0:
            leap=False
            if year%400 ==0:
                leap = True
    # Write your logic here
    
    return leap

# Print Function
if __name__ == '__main__':
    n = int(input())
    for i in range(1,n+1):
        print(i, end="") #end removes the \n added by the print function by default

# Symmetric Difference
# Enter your code here. Read input from STDIN. Print output to STDOUT
M = int(input())
a = set(map(int, input().split()))
N = int(input())
b = set(map(int, input().split()))
symmetric_difference = a.symmetric_difference(b)
for elem in sorted(symmetric_difference):
    print(elem)

# No Idea!
# Enter your code here. Read input from STDIN. Print output to STDOUT
n, m = map(int, input().split())
arr = list(map(int, input().split()))
A = set(map(int, input().split()))
B = set(map(int, input().split()))
happiness = 0
for elem in arr:
    if elem in A:
        happiness += 1
    elif elem in B:
        happiness -= 1
print(happiness)

# Set .add()
# Enter your code here. Read input from STDIN. Print output to STDOUT
country_stamps = set()
n = int(input())  # 7
# To do not count repeated country
for i in range(n):
    stamp = input().strip()
    country_stamps.add(stamp)
print(len(country_stamps))

# collections.Counter()
from collections import Counter
# Read input
n = int(input())
sizes = list(map(int, input().split()))
inventory = Counter(sizes) #create inv
customers = int(input())
revenue = 0
# Loop
for i in range(customers):
    size, price = map(int, input().split())
    
    # If shoe size is available, sell it
    if inventory[size] > 0:
        revenue += price
        inventory[size] -= 1
print(revenue)


# DefaultDict Tutorial
from collections import defaultdict
n, m = list(map(int,input().split()))
group_A = defaultdict(list)
for _ in range(n):
    a = str(input())
    group_A[a].append(_ + 1)
#print(group_A)
group_B = defaultdict(list)
for _ in range(m):
    b = str(input())
    #group_B[b].append(_)
#print(group_B)
    if b in group_A:
        print(' '.join(map(str, group_A[b])))
    else:
        print(-1)


# List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    result = [[i, j, k] for i in range(x + 1) 
                        for j in range(y + 1) 
                        for k in range(z + 1) 
                        #rn we run every possible combination
                        if i + j + k != n] #if the sum is equal to n we are gonna exclude it
    
    print(result)
#we don't need to sort

# Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    unique_scores = set(arr) #if we have same score
    unique_scores.remove(max(unique_scores))
    runner_up = max(unique_scores)
    print(runner_up)

# Nested Lists
if __name__ == '__main__':
    
    students = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        #print(name)
        #print(score)
        students.append([name, score])
    # we sort them by score (second element of the list) and name(in case of tie)  
    students_sorted = sorted(students, key=lambda x: (x[1], x[0]))  
    #print(students_sorted)
    unique_scores = sorted(set([score for name, score in students_sorted]))
    second_lowest_score = unique_scores[1]
    second_lowest_students = [name for name, score in students_sorted if score == second_lowest_score]
    for student in second_lowest_students:
        print(student)

# Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
        #print(student_marks)
    query_name = input()
    
    query_scores = student_marks[query_name]
    average_score = sum(query_scores) / len(query_scores)
    print(f"{average_score:.2f}")

# Mod Divmod
# Enter your code here. Read input from STDIN. Print output to STDOUT
a = int(input())
b = int(input())
print(a // b)
print(a % b)
print(divmod(a, b))

# Power - Mod Power
# Enter your code here. Read input from STDIN. Print output to STDOUT
a=int(input())
b=int(input())
m=int(input())
print(pow(a,b))
print(pow(a,b,m))



# Integers Come In All Sizes
# Enter your code here. Read input from STDIN. Print output to STDOUT
a = int(input())
b = int(input())
c = int(input())
d = int(input())
result = pow(a,b)+pow(c,d)
print(result)

# Triangle Quest
for i in range(1,int(input())):
    print((10**i // 9) * i)

# Lists
if __name__ == '__main__':
    N = int(input())
    list = []
    for _ in range(N):
        command = input().split() 
        #print(command)
        #['insert', '0', '5'] we have this format
        if command[0] == "print":
            print(list)
        else:
            eval(f'list.{command[0]}({",".join(command[1:])})') #dynamic string
#eval() allow us to execute the string as a py code
#command[0] is the command string
#command[1:] are the values

# Merge the Tools!
import linecache
def merge_the_tools(string, k):
    for i in range(0, len(string), k):
        substring = string[i:i+k] #we have divided the string in k
        #print(substring)
        unique_chars = ""
        for char in substring:
            if char not in unique_chars: #check if it was already added
                unique_chars += char #if not, add it
        print(unique_chars)


# Text Wrap

def wrap(string, max_width):
    
    return textwrap.fill(string, max_width)


# Designer Door Mat
# Enter your code here. Read input from STDIN. Print output to STDOUT
N, M = map(int, input().split())
#print(N,M)
# Above
for i in range(1, N, 2):
    pattern = ('.|.' * i).center(M, '-')
    print(pattern)
# Center
print('WELCOME'.center(M, '-'))
# Below (just mirroring upside down)
for i in range(N-2, 0, -2):
    pattern = ('.|.' * i).center(M, '-')
    print(pattern)

# String Formatting
def print_formatted(number):
    # your code goes here
    width = len(bin(number)) -2  # Subtract 2 to remove the '0b' prefix
    #print(width)
    #print(number)
    #print(bin(number))
    #print(len(bin(number)))
    for i in range(1, number + 1):
        print(f'{i:{width}} {i:{width}o} {i:{width}X} {i:{width}b}')
# I was using a different approach, the output was correct but the test were looking for diff space between each value that was not posssible with my method, i'll share answer and output, the one I sent was taken from discussion section with a similar method to mine
'''
        print(f"{i:{width}d}  {oct(i)[2:]:{width}} {hex(i)[2:].upper():{width}} {bin(i)[2:]:{width}}") #[2:] to remove the prefix
        
 1  1  1  1 
 2  2  2  10
 '''

# Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    #print(integer_list)
    t = tuple(integer_list)
    print(hash(t))

# Find Angle MBC
# Enter your code here. Read input from STDIN. Print output to STDOUT
import math
ab = int(input())
bc = int(input())
ac= (ab**2+bc**2)**(1/2)
#print(ac)
t_radians = math.asin(ab / ac)
t_degrees = math.degrees(t_radians)
degree_sign = u'\N{DEGREE SIGN}'
#solution on degree by Stackoverflow, using (degree symbol on keyboard) was giving error since the test doesn't accept non ascii character
print(f"{round(t_degrees)}{degree_sign}")

# Triangle Quest 2
for i in range(1,int(input())+1):
    print(((10**i - 1) // 9)**2) #I checked discussion section

# sWAP cASE
def swap_case(s):
    
    return s.swapcase()
    

# String Split and Join

def split_and_join(line):
    # write your code here
    split = line.split(" ") 
    result = "-".join(split)
    return result
if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

# What's Your Name?
#
# Complete the 'print_full_name' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING first
#  2. STRING last
#
def print_full_name(first, last):
    # Write your code here
    print(f"Hello {first} {last}! You just delved into python.")

# Mutations
def mutate_string(string, position, character):
    output = string[:position] + character + string[position + 1:]
    return output

# Find a string
def count_substring(string, sub_string):
    count = 0
    # explained we iterate each part of len(sub_string) of the string and check if it match the value of our sub_string, if it does we update the count
    for i in range(len(string) - len(sub_string) + 1):
        if string[i:i+len(sub_string)] == sub_string:
            count += 1
    return count

# String Validators
if __name__ == '__main__':
    s = input()
    
    print (any(c.isalnum() for c in s))
    print (any(c.isalpha()for c in s))
    print (any(c.isdigit()for c in s))
    print (any(c.islower()for c in s))
    print (any(c.isupper()for c in s))

# Text Alignment
# Enter your code here. Read input from STDIN. Print output to STDOUT
thickness = int(input())  #height of \n
c = 'H'
'''
    H    
   HHH   
  HHHHH  
 HHHHHHH 
HHHHHHHHH
'''
for i in range(thickness):
    print((c * (2*i + 1)).center(thickness * 2 - 1))
'''
  HHHHH               HHHHH             
  HHHHH               HHHHH             
  HHHHH               HHHHH             
  HHHHH               HHHHH             
  HHHHH               HHHHH             
  HHHHH               HHHHH             
'''
for i in range(thickness + 1):
    print((c * thickness).center(thickness * 2) + (c * thickness).center(thickness * 6))
'''
  HHHHHHHHHHHHHHHHHHHHHHHHH   
  HHHHHHHHHHHHHHHHHHHHHHHHH   
  HHHHHHHHHHHHHHHHHHHHHHHHH  
'''
for i in range((thickness + 1) // 2):
    print((c * thickness * 5).center(thickness * 6))
#same as second part
for i in range(thickness + 1):
    print((c * thickness).center(thickness * 2) + (c * thickness).center(thickness * 6))
'''
HHHHHHHHH 
 HHHHHHH  
  HHHHH   
   HHH    
    H 
'''               
for i in range(thickness):
    print(((c * (2*(thickness-i) - 1)).center(thickness * 2)).rjust(thickness * 6))

# Alphabet Rangoli
def print_rangoli(size):
    # your code goes here
    import string
    alphabet = string.ascii_lowercase
    #print(alphabet)
    rows = []
    for i in range(size):
        s = "-".join(alphabet[size-1:i:-1] + alphabet[i:size])
        #print(s) it gives letter sequence starting from the center
        rows.append(s.center(4*size - 3, '-'))
    #print(rows) center and bottom half looks good
    #print(rows[::-1]) is upper (containing center)
    #print(rows[1:]) is lower without center
    #put \n for each line 
    print('\n'.join(rows[::-1] + rows[1:]))


# Capitalize!

# Complete the solve function below.
def solve(s):
    #result = s.title() it was easy answer but fail test 2
    '''
Wrong Answer
Input (stdin)
1 w 2 r 3g
Expected Output
1 W 2 R 3g
Hidden Test Case
    '''
    words = s.split(' ')
    result = []
    for word in words:
        if len(word) > 0:
            result.append(word[0].upper() + word[1:])
        else:
            result.append(word)
    
    return ' '.join(result)

# The Minion Game
def minion_game(string):
    # your code goes here
    vowels = 'AEIOU'
    kevin_score = 0
    stuart_score = 0
    length = len(string)
    
    for i in range(length):
        if string[i] in vowels:
            # Kevin scores for substrings starting with a vowel
            kevin_score += length - i
        else:
            # Stuart scores for substrings starting with a consonant
            stuart_score += length - i
    if kevin_score > stuart_score:
        print(f"Kevin {kevin_score}")
    elif stuart_score > kevin_score:
        print(f"Stuart {stuart_score}")
    else:
        print("Draw")


# Polar Coordinates
# Enter your code here. Read input from STDIN. Print output to STDOUT
import cmath
z = complex(input().strip())
#print(z)
r = abs(z)
phi = cmath.phase(z)
print(f"{r:.3f}")
print(f"{phi:.3f}")

# Introduction to Sets
def average(array):
    # your code goes here
    #print(set(array))
    return (sum(set(array))/len(set(array)))
    


# Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))
num_commands = int(input())
# Execute
for _ in range(num_commands):
    command = input().split()
    if command[0] == "pop":
        s.pop()
    elif command[0] == "remove":
        s.remove(int(command[1]))
    elif command[0] == "discard":
        s.discard(int(command[1]))
print(sum(s))

# Set .union() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n_e = int(input())
e_r = set(map(int, input().split()))
n_f = int(input())
f_r = set(map(int, input().split()))
tot = e_r.union(f_r)
print(len(tot))

# Set .intersection() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n_e = int(input())
e_r = set(map(int, input().split()))
n_f = int(input())
f_r = set(map(int, input().split()))
tot = e_r.intersection(f_r)
print(len(tot))

# Set .difference() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n_e = int(input())
e_r = set(map(int, input().split()))
n_f = int(input())
f_r = set(map(int, input().split()))
tot = e_r.difference(f_r)
print(len(tot))

# Set .symmetric_difference() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n_e = int(input())
e_r = set(map(int, input().split()))
n_f = int(input())
f_r = set(map(int, input().split()))
tot = e_r.symmetric_difference(f_r)
print(len(tot))
# they are finished now but the previous 4/5 ex were only copy pasting, changing only the operator

# Set Mutations
# Enter your code here. Read input from STDIN. Print output to STDOUT
A = int(input())
el_a = set(map(int,input().split()))
N = int(input())
for _ in range(N):
    operation, _ = input().split()
    other_set = set(map(int, input().split()))
    # Execute
    if operation == 'intersection_update':
        el_a.intersection_update(other_set)
    elif operation == 'update':
        el_a.update(other_set)
    elif operation == 'symmetric_difference_update':
        el_a.symmetric_difference_update(other_set)
    elif operation == 'difference_update':
        el_a.difference_update(other_set)
print(sum(el_a))

# The Captain's Room
# Enter your code here. Read input from STDIN. Print output to STDOUT
k = int(input())
r_n = list(map(int, input().split()))
u_r_n = set(r_n)
sum_list = sum(r_n) #113
sum_set = sum(u_r_n) #29
captain_room = (k * sum_set - sum_list) // (k - 1)
print(captain_room)

# Check Subset
# Enter your code here. Read input from STDIN. Print output to STDOUT
t = int(input())
for _ in range(t):
    n_A = int(input())
    A = set(map(int, input().split()))
    n_B = int(input())
    B = set(map(int, input().split()))
    print(A.issubset(B))

# Check Strict Superset
# Enter your code here. Read input from STDIN. Print output to STDOUT
A = set(map(int, input().split()))
n = int(input())
is_strict_superset = True
for _ in range(n):
    other_set = set(map(int, input().split()))
    if not (A.issuperset(other_set) and len(A) > len(other_set)):
        is_strict_superset = False
        break
print(is_strict_superset)

# itertools.product()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from itertools import product
A = list(map(int, input().split()))
B = list(map(int, input().split()))
print (*product(A,B))

# itertools.permutations()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from itertools import permutations
s, k = input().split()
k = int(k)
perm = permutations(sorted(s), k)
for p in perm:
    print(''.join(p))

# itertools.combinations()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from itertools import combinations
s, k = input().split()
k = int(k)
for i in range(1, k + 1):
    for combo in combinations(sorted(s), i):
        print(''.join(combo))

# itertools.combinations_with_replacement()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from itertools import combinations_with_replacement
s, k = input().split()
k = int(k)
comb = combinations_with_replacement(sorted(s), k)
for c in comb:
    print(''.join(c))
# same code as permutations case

# Compress the String!
# Enter your code here. Read input from STDIN. Print output to STDOUT
from itertools import groupby
s = input()
result = [(len(list(g)), int(k)) for k, g in groupby(s)]
print(' '.join(f"({count}, {char})" for count, char in result))

# Iterables and Iterators
# Enter your code here. Read input from STDIN. Print output to STDOUT
from itertools import combinations
N = int(input())
l = input().split()
k = int(input())  
comb = list(combinations(l, k))
count = 0
for c in comb:
    #print(c)
    if 'a' in c:  
        count += 1

print(count/len(comb))

# Maximize It!
# Enter your code here. Read input from STDIN. Print output to STDOUT
from itertools import product
K, M = list(map(int,input().split()))
lists = []
for _ in range(K):
    data = list(map(int, input().split()))[1:]
    lists.append(data)
#print(lists)
max_value = 0
for c in product(*lists):
    # we multiply one by one the "i" value on the first with the "j" value of the second list and the "k" value of the third
    current_value = sum(x**2 for x in c) % M 
    #print(current_value) obtain all the product of their combination
    max_value = max(max_value, current_value)
print(max_value)

# Arrays

 
def arrays(arr):
    # complete this function
    # use numpy.array
    return numpy.flip(numpy.array(arr,float))

# Shape and Reshape
import numpy
list = list(map(int, input().split()))
arr = numpy.array(list)
#print(arr.shape)
arr.shape = (3,3)
print(arr)

# Transpose and Flatten
import numpy as np
N, M = list(map(int, input().split()))
matrix = np.array([input().split() for _ in range(N)], int)
#print(matrix)
print(matrix.T)
print(matrix.flatten())

# Concatenate
import numpy as np
N, M, P = list(map(int,input().split()))
arr1 = []
for _ in range(N):
    n_p = list(map(int,input().split()))
    arr1.append(n_p)
    #print(arr1)
    
arr2 = []
for _ in range(M):
    m_p = list(map(int,input().split()))
    arr2.append(m_p)
    #print(arr2)
arr1= np.array(arr1)
arr2= np.array(arr2)
conc = np.concatenate ((arr1,arr2),axis = 0)
print(conc)

# Zeros and Ones
import numpy as np
dimensions = list(map(int, input().split()))
if len(dimensions) == 2:
    x, y = dimensions
    print(np.zeros((x, y), dtype=int))
    print(np.ones((x, y), dtype=int))
elif len(dimensions) == 3:
    x, y, z = dimensions
    print(np.zeros((x, y, z), dtype=int))
    print(np.ones((x, y, z), dtype=int))
elif len(dimensions) == 4:
    x, y, z,k = dimensions
    print(np.zeros((x, y, z, k), dtype=int))
    print(np.ones((x, y, z, k), dtype=int))
#Test case was giving problem for 3d array because one of the test input cointained only 2 value, this way we handle that case as well
# There is also a 4 case scenario :(

# Eye and Identity
import numpy as np
np.set_printoptions(legacy='1.13')
N, M = list(map(int,input().split()))
print(np.eye(N,M))

# Array Mathematics
import numpy as np
N, M = list(map(int,input().split()))
'''
arr_A = list(map(int,input().split()))
arr_B = list(map(int,input().split()))
''' #in test scenario we need to handle also case where N>1 
arr_A = []
arr_B = []
for _ in range(N):
    arr_A.append(list(map(int, input().split())))
for _ in range(N):
    arr_B.append(list(map(int, input().split())))
a = np.array(arr_A, int)
b = np.array(arr_B, int)
print(a+b)
print(a-b)
print(a*b)
print(a // b)
print(a % b)
print(a**b)

# Floor, Ceil and Rint
import numpy as np
np.set_printoptions(legacy='1.13')
A = np.array(list(map(float, input().split())))
print(np.floor(A))
print(np.ceil(A))
print(np.rint(A))

# Sum and Prod
import numpy as np
N, M = list(map(int,input().split()))
arr = []
for _ in range(N):
    arr.append(list(map(int, input().split())))
sum = (np.sum(arr, axis = 0))
print (np.prod(sum, axis = 0))

# Min and Max
import numpy
N, M = list(map(int,input().split()))
arr = []
for _ in range(N):
    arr.append(list(map(int, input().split())))
min = numpy.min(arr, axis = 1)
#print(min)
print (numpy.max(min, axis = None))

# Mean, Var, and Std
import numpy
N, M = list(map(int,input().split()))
arr = []
for _ in range(N):
    arr.append(list(map(int,input().split())))
print(numpy.mean(arr, axis = 1))
print(numpy.var(arr, axis = 0))
std = (numpy.std(arr, axis = None))
print(round(std,11))

# Dot and Cross
import numpy
N = int(input())
arr_A = []
arr_B = []
for _ in range(N):
    arr_A.append(list(map(int, input().split())))
for _ in range(N):
    arr_B.append(list(map(int, input().split())))
    
A = numpy.array(arr_A)
B = numpy.array(arr_B)
print(numpy.dot(arr_A,arr_B))
# we can also use matmul

# Inner and Outer
import numpy
arr_A = list(map(int,input().split()))
arr_B = list(map(int,input().split()))
A = numpy.array(arr_A)
B = numpy.array(arr_B)
print (numpy.inner(A, B))
print (numpy.outer(A, B))

# Polynomials
import numpy
coeff = list(map(float, input().split()))
x = float(input())
#print(coeff, x)
p = numpy.polyval(coeff,x)
print(p)

# Linear Algebra
import numpy
N = int(input())
arrays = {}
for i in range(N):
    arrays[f'arr{i}'] = []
#print(arrays)
#print(len(arrays))
for j in range (len(arrays)):
    arrays[f'arr{j}'] = list(map(float, input().split()))
#print(arrays)
matrix = numpy.array([arrays[f'arr{i}'] for i in range(N)])
#print(matrix)
det = (numpy.linalg.det(matrix))
print(round(det,2))

# Collections.namedtuple()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import namedtuple
N = int(input())
columns = input().split()
Student = namedtuple('Student', columns)
total_marks = 0
for _ in range(N):
    student = Student(*input().split())
    #print(student)
    total_marks += int(student.MARKS)
avg_mark = round((total_marks / N),2)
print(avg_mark)

# Collections.OrderedDict()
from collections import OrderedDict
N = int(input())
ordered_items = OrderedDict()
for _ in range(N):
    *item_name, price = input().split()
    item_name = " ".join(item_name)
    price = int(price)
    if item_name in ordered_items:
        # if item it's already in dict
        ordered_items[item_name] += price
    else:
        #if not, add it
        ordered_items[item_name] = price
for item, total_price in ordered_items.items():
    print(f"{item} {total_price}")

# Word Order
from collections import OrderedDict
n = int(input())
word_count = OrderedDict()
for _ in range(n):
    word = input().strip()  
    if word in word_count:
        word_count[word] += 1
    else:
        word_count[word] = 1
print(len(word_count))
# except for the input format is the same exercise as the previous one
#print(word_count.values())
result = " ".join(map(str, word_count.values())) 
print(result)

# Collections.deque()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import deque
d = deque()
n = int(input())
# Execute
for _ in range(n):
    command = input().split()
    if command[0] == "append":
        d.append(command[1])
    elif command[0] == "appendleft":
        d.appendleft(command[1])
    elif command[0] == "pop":
        d.pop()
    elif command[0] == "popleft":
        d.popleft()
print(" ".join(d))

# Piling Up!
from collections import deque
# since we loop over the number of test case
for _ in range(int(input())):
    input()  # Skip T
    cubes = deque(map(int, input().split()))
    top = float('inf') #to ensure top is never smaller than the cube we pick , i used 100 but test case failed, with inf worked
    
    # Check if we can place a cube (either from the left or right) on top
    while cubes and (cubes[0] <= top and cubes[-1] <= top):
        # if the while is false we have place all the cubes correctly
        # now we pick the greater value
        if cubes[0] >= cubes[-1]:
            top = cubes.popleft()
        else:
            top = cubes.pop()
    # cubes empty after loop = correctly stacked
    print("Yes" if not cubes else "No")

# Company Logo
from collections import Counter
s = str(input())
char_count = Counter(s)
#print(char_count.items())
# in lambda function element 1 refers to numeric value while 0 to character
sorted_chars = sorted(char_count.items(), key=lambda x: (-x[1], x[0]))
# top 3
for char, count in sorted_chars[:3]:
    print(char, count)


# Calendar Module
# Enter your code here. Read input from STDIN. Print output to STDOUT
import calendar
MM, DD, YYYY = list(map(int,input().split()))
index_weekday = calendar.weekday(YYYY, MM, DD)
print(calendar.day_name[index_weekday].upper())

# Time Delta
# Complete the time_delta function below.
import os
import re
from datetime import datetime
def time_delta(t1, t2):
    format_ = '%a %d %b %Y %H:%M:%S %z'
    
    # Parse the two timestamps into datetime objects
    dt1 = datetime.strptime(t1, format_)
    dt2 = datetime.strptime(t2, format_)
    diff_S = abs(int((dt1 - dt2).total_seconds()))
    return str(diff_S)
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        t1 = input()
        t2 = input()
        delta = time_delta(t1, t2)
        fptr.write(delta + '\n')
    fptr.close()

# Exceptions
# Enter your code here. Read input from STDIN. Print output to STDOUT
T = int(input())
for _ in range(T):
    try:
        a, b = input().split()
        print(int(a) // int(b)) 
    except ZeroDivisionError as e: # Handle division by zero
        print("Error Code:", e)
    except ValueError as e: # Handle invalid integer conversion
        print("Error Code:", e)

# Incorrect Regex
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
T = int(input())
invalid_patterns = ['*+', '+*', '++', '**', '?*', '*?', '??', '+?']
def has_invalid_pattern(s):
    for pattern in invalid_patterns:
        if pattern in s:
            return True
    return False
for _ in range(T):
    s = input()
    if has_invalid_pattern(s):
        print(False)
    else:
        try:
            re.compile(s)
            print(True)
        except re.error:
            print(False)
# took ispiration from here: https://stackoverflow.com/questions/19630994/how-to-check-if-a-string-is-a-valid-regex-in-python
        

# Words Score
def is_vowel(letter):
    return letter in ['a', 'e', 'i', 'o', 'u', 'y']
def score_words(words):
    score = 0
    for word in words:
        num_vowels = 0
        for letter in word:
            if is_vowel(letter):
                num_vowels += 1
        if num_vowels % 2 == 0:
            score += 2
        else:
            score +=1 #increment operator here was wrong
    return score

# Default Arguments

def print_from_stream(n, stream=None): #EvenStream() was defined here, instead of making a new one each time the code was runned, it was continuing from what it was left
    if stream is None:
        stream = EvenStream()
    for _ in range(n):
        print(stream.get_next())

# Detect Floating Point Number
# Enter your code here. Read input from STDIN. Print output to STDOUT
N = int(input())
import re
def is_valid_float(string):
    pattern = r'^[-+]?\d*\.\d+$'
    #^[-+]? allows to start with + and -
    #\d* allows digit before decimal point
    #\. need exactly one decimal point
    #\d+$ at least one digit after decimal point
    
    try:
        if re.match(pattern, string):
            float(string)
            return True
        else:
            return False
    except ValueError:
        return False
# We catch as False if it doesn't match the pattern or it can't be converted to float
for _ in range(N):
    test_string = input().strip()
    print(is_valid_float(test_string))

# Classes: Dealing with Complex Numbers

class Complex(object):
    #inizializzation
    def __init__(self, real, imaginary):
        self.real = real
        self.imaginary = imaginary
        #print(f"{self.real} + {self.imaginary}i")
    # +
    def __add__(self, other):
        real = self.real + other.real
        imaginary = self.imaginary + other.imaginary
        return Complex(real, imaginary)
        #print(f"+: {real} + {imaginary}i")
    # -
    def __sub__(self, other):
        real = self.real - other.real
        imaginary = self.imaginary - other.imaginary
        return Complex(real, imaginary)
        #print(f"-: {real} + {imaginary}i")
    # *
    def __mul__(self, other):
        real = (self.real * other.real) - (self.imaginary * other.imaginary)
        imaginary = (self.real * other.imaginary) + (self.imaginary * other.real)
        return Complex(real, imaginary)
        #print(f"*: {real} + {imaginary}i")
    # /
    def __truediv__(self, other):
        denominator = other.real ** 2 + other.imaginary ** 2
        real = (self.real * other.real + self.imaginary * other.imaginary) / denominator
        imaginary = (self.imaginary * other.real - self.real * other.imaginary) / denominator
        return Complex(real, imaginary)
        #print(f"/: {real} + {imaginary}i")
    # modulus (absolute value) 
    def mod(self):
        modulus = math.sqrt(self.real ** 2 + self.imaginary ** 2)
        return Complex(modulus, 0)
        #print(f"mod: {modulus}")
# Formatting
    def __str__(self):
        if self.imaginary == 0:
            result = "%.2f+0.00i" % (self.real)
        elif self.real == 0:
            if self.imaginary >= 0:
                result = "0.00+%.2fi" % (self.imaginary)
            else:
                result = "0.00-%.2fi" % (abs(self.imaginary))
        elif self.imaginary > 0:
            result = "%.2f+%.2fi" % (self.real, self.imaginary)
        else:
            result = "%.2f-%.2fi" % (self.real, abs(self.imaginary))
        return result

# Class 2 - Find the Torsional Angle

class Points(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    # Subtraction of two points to form a vector AB = B - A
    def __sub__(self, other):
        return Points(self.x - other.x, self.y - other.y, self.z - other.z)
    # Dot product of two vectors X,Y
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    # Cross product of two vectors AB * BC
    def cross(self, other):
        cross_x = self.y * other.z - self.z * other.y
        cross_y = self.z * other.x - self.x * other.z
        cross_z = self.x * other.y - self.y * other.x
        return Points(cross_x, cross_y, cross_z)
    # Magnitude (absolute value) of a vector
    def absolute(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)


# Zipped!
# Enter your code here. Read input from STDIN. Print output to STDOUT
N, X = map(int, input().split()) 
marks = [list(map(float, input().split())) for _ in range(X)]
for student_marks in zip(*marks):
    print(f"{sum(student_marks) / X:.1f}")

# Input()
# Enter your code here. Read input from STDIN. Print output to STDOUT
x, k = map(int, input().split())
polynomial = input()
if eval(polynomial) == k:
    print(True)
else:
    print(False)

# Python Evaluation
# Enter your code here. Read input from STDIN. Print output to STDOUT
expression = input()
eval(expression)

# Athlete Sort
#!/bin/python3
import math
import os
import random
import re
import sys
if __name__ == '__main__':
    n, m = map(int, input().split())
    arr = []
    for _ in range(n):
        arr.append(list(map(int, input().split())))
    k = int(input())
    sorted_arr = sorted(arr, key=lambda x: x[k])
    
    for row in sorted_arr:
        print(' '.join(map(str, row)))

# Any or All
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
lst = list(map(int, input().split()))
all_positive = all(x > 0 for x in lst)
any_palindromic = any(str(x) == str(x)[::-1] for x in lst)
if all_positive and any_palindromic:
    print(True)
else:
    print(False)

# ginortS
# Enter your code here. Read input from STDIN. Print output to STDOUT
def custom_sort(char):
# Sorts by priority
    if char.islower():
        return (0, char)
    elif char.isupper():
        return (1, char)
    elif char.isdigit():
        if int(char) % 2 == 1:  # Odd
            return (2, char)
        else:                   # Even
            return (3, char)
s = input()
sorted_string = ''.join(sorted(s, key=custom_sort))
print(sorted_string)

# Re.split()
import re
regex_pattern = r'[,.]'

# Group(), Groups() & Groupdict()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
s = input()
match = re.search(r'([a-zA-Z0-9])\1+', s)
if match:
    print(match.group(1))
else:
    print(-1)

# Re.findall() & Re.finditer()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
s = input()
pattern = r'(?<=[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM])([aeiouAEIOU]{2,})(?=[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM])'
matches = re.findall(pattern, s)
if matches:
    for match in matches:
        print(match)
else:
    print(-1)

# Re.start() & Re.end()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
s = input()
k = input()
matches = list(re.finditer(r'(?={})'.format(re.escape(k)), s))
if not matches:
    print((-1, -1))
else:
    for match in matches:
        start_index = match.start()
        end_index = start_index + len(k) - 1
        print((start_index, end_index))

# Regex Substitution
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
N = int(input())
lines = [input() for _ in range(N)]
for line in lines:
    line = re.sub(r'(?<= )&&(?= )', 'and', line)
    line = re.sub(r'(?<= )\|\|(?= )', 'or', line)
    print(line)

# Validating Roman Numerals
import re
regex_pattern = r'^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'

# Validating phone numbers
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
N = int(input())
for _ in range(N):
    mobile_number = input().strip() 
    if re.match(r'^[789]\d{9}$', mobile_number):
        print("YES")
    else:
        print("NO")

# Validating and Parsing Email Addresses
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
import email.utils
email_pattern = r'^[a-zA-Z][\w.-]+@[a-zA-Z]+\.[a-zA-Z]{1,3}$'
n = int(input())
for _ in range(n):
    full_entry = input().strip()
    name, email_address = email.utils.parseaddr(full_entry)
    if re.match(email_pattern, email_address):
        print(full_entry)  

# Map and Lambda Function
cube = lambda x: x ** 3
def fibonacci(n):
    fibonacci = [0, 1]
    for i in range(2, n):
        fibonacci.append(fibonacci[i-1] + fibonacci[i-2])
    return fibonacci[:n]

# Validating Email Addresses With a Filter
import re
def fun(s):
    # return True if s is a valid email, else return False
    pattern = r'^[a-zA-Z0-9_-]+@[a-zA-Z0-9]+\.[a-zA-Z]{1,3}$'
    return re.match(pattern, s) is not None

# Hex Color Code
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
def find_hex_codes(n, lines):
    hex_color_pattern = re.compile(r'(?<!^)(#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}))\b')
    for line in lines:
        matches = hex_color_pattern.findall(line)
        for match in matches:
            print(match)
n = int(input()) 
lines = [input().strip() for _ in range(n)]
find_hex_codes(n, lines)

# HTML Parser - Part 1
# Enter your code here. Read input from STDIN. Print output to STDOUT
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    # start tags
    def handle_starttag(self, tag, attrs):
        print(f"Start : {tag}")
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1] if attr[1] is not None else 'None'}")
    # end tags
    def handle_endtag(self, tag):
        print(f"End   : {tag}")
    # empty tags
    def handle_startendtag(self, tag, attrs):
        print(f"Empty : {tag}")
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1] if attr[1] is not None else 'None'}")
n = int(input())
html_code = ""
for _ in range(n):
    html_code += input().strip()
parser = MyHTMLParser()
parser.feed(html_code)

# HTML Parser - Part 2
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    # comments
    def handle_comment(self, data):
        if '\n' in data:
            print(">>> Multi-line Comment")
        else:
            print(">>> Single-line Comment")
        print(data)
    
    # data
    def handle_data(self, data):
        if data.strip():  # skip data if \n
            print(">>> Data")
            print(data)
html = ""
for i in range(int(input())):
    html += input().rstrip() + '\n'
parser = MyHTMLParser()
parser.feed(html)
parser.close()

# Detect HTML Tags, Attributes and Attribute Values
# Enter your code here. Read input from STDIN. Print output to STDOUT
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    # start tag
    def handle_starttag(self, tag, attrs):
        print(tag)
        # and their attributes
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1]}")
    
    # empty tags
    def handle_startendtag(self, tag, attrs):
        print(tag)
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1]}")
    
    # pass comments
    def handle_comment(self, data):
        pass
n = int(input())
html_code = ""
for i in range(n):
    html_code += input().strip() + '\n'
parser = MyHTMLParser()
parser.feed(html_code)
parser.close()

# Validating UID
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
def validate_uid(uid):
    if len(uid) != 10:
        return "Invalid"
    if not uid.isalnum():
        return "Invalid"
    if len(re.findall(r'[A-Z]', uid)) < 2:
        return "Invalid"
    if len(re.findall(r'\d', uid)) < 3:
        return "Invalid"
    if len(set(uid)) != len(uid):
        return "Invalid"
    return "Valid"
n = int(input())
for _ in range(n):
    uid = input().strip()
    print(validate_uid(uid))

# Validating Credit Card Numbers
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
def is_valid_credit_card(card):
    pattern = r'^([4-6]\d{3}-\d{4}-\d{4}-\d{4}|[4-6]\d{15})$'
    if not re.match(pattern, card):
        return "Invalid"
    
    # Remove -
    card_digits = card.replace('-', '')
    
    if re.search(r'(\d)\1{3,}', card_digits):
        return "Invalid"
    return "Valid"
n = int(input())
for _ in range(n):
    card = input().strip()
    print(is_valid_credit_card(card))

# Validating Postal Codes
import re
# Matches integers between 100000 and 999999
regex_integer_in_range = r"^[1-9][0-9]{5}$"
# Finds alternating repetitive digit pairs
regex_alternating_repetitive_digit_pair = r"(?=(\d)\d\1)" 

# Matrix Script
#!/bin/python3
import math
import os
import random
import re
import sys
n, m = map(int, input().split())
matrix = [input() for _ in range(n)]
#print(matrix)
# T by column and concatenate
decoded_script = ''.join([matrix[i][j] for j in range(m) for i in range(n)])
#print(decoded_script)

cleaned_script = re.sub(r'(?<=\w)[^\w]+(?=\w)', ' ', decoded_script)
print(cleaned_script)

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)

# Decorators 2 - Name Directory

def person_lister(f):
    def inner(people):
# x[2] is age
        sorted_people = sorted(people, key=lambda x: int(x[2]))
        return [f(person) for person in sorted_people]
    return inner


# Reduce Function

def product(fracs):
    t = reduce(lambda x, y: x * y, fracs)# complete this line with a reduce statement
    return t.numerator, t.denominator

# Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        # Normalize +91 xxxxx xxxxx
        formatted_numbers = ['+91 ' + number[-10:-5] + ' ' + number[-5:] for number in l]
        return f(formatted_numbers)
    return fun


# XML2 - Find the Maximum Depth

maxdepth = 0
def depth(elem, level):
    global maxdepth
    for i in elem:
        depth(i, level +1)
    maxdepth = max(maxdepth, level+1)


# XML 1 - Find the Score

def get_attr_number(node):
    return sum([len(i.items()) for i in node.iter()])

# Birthday Cake Candles
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#
def birthdayCakeCandles(candles):
    tallest = max(candles)
    return candles.count(tallest)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    candles_count = int(input().strip())
    candles = list(map(int, input().rstrip().split()))
    result = birthdayCakeCandles(candles)
    fptr.write(str(result) + '\n')
    fptr.close()

# Number Line Jumps
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#
def kangaroo(x1, v1, x2, v2):
    # v1 == v2 and x1 != x2 they can never meet
    if v1 == v2:
        return "YES" if x1 == x2 else "NO"
 # the difference in starting position is actually divisible by the difference in velocity (to ensure they have meetable point) and the kangaroo behind is faster so it can catch up
    if (x2 - x1) % (v1 - v2) == 0 and (x1 < x2 and v1 > v2 or x1 > x2 and v2 > v1):
        return "YES"
    
    return "NO"
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    x1 = int(first_multiple_input[0])
    v1 = int(first_multiple_input[1])
    x2 = int(first_multiple_input[2])
    v2 = int(first_multiple_input[3])
    result = kangaroo(x1, v1, x2, v2)
    fptr.write(result + '\n')
    fptr.close()

# Viral Advertising
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#
def viralAdvertising(n):
    in_people = 5
    cumulative_likes = 0
    for day in range(1, n + 1):
        liked = in_people // 2
        cumulative_likes += liked
        in_people = liked * 3
    return cumulative_likes

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input().strip())
    result = viralAdvertising(n)
    fptr.write(str(result) + '\n')
    fptr.close()

# Recursive Digit Sum
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k
#
def superDigit(n, k):
    digit_sum = sum(int(digit) for digit in n)
    total_sum = digit_sum * k
    while total_sum >= 10:
        total_sum = sum(int(digit) for digit in str(total_sum))
    
    return total_sum
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    n = first_multiple_input[0]
    k = int(first_multiple_input[1])
    result = superDigit(n, k)
    fptr.write(str(result) + '\n')
    fptr.close()

# Insertion Sort - Part 1
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#
def insertionSort1(n, arr):
    unsorted_value = arr[-1]
    
    i = n - 2 # we start by the first value next to our unsorted and move to the left
    while i >= 0 and arr[i] > unsorted_value:
        arr[i + 1] = arr[i]
        print(" ".join(map(str, arr)))
        i -= 1
# Place unsorted value in correct position
    arr[i + 1] = unsorted_value
    print(" ".join(map(str, arr)))
if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort1(n, arr)

# Insertion Sort - Part 2
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'insertionSort2' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#
def insertionSort2(n, arr):
    for i in range(1, n):
        unsorted_value = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > unsorted_value:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = unsorted_value
        print(" ".join(map(str, arr)))

if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort2(n, arr)

