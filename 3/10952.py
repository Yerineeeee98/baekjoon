# import sys

# a = 1
# b = 1

# while a != 0:
#     while b != 0:
    
#         a, b  = map(int, sys.stdin.readline().split())

#         print(a+b)

import sys

while True:
    a, b  = map(int, sys.stdin.readline().split())
    
    if a == 0 and b == 0 :
            break
    print(a+b)