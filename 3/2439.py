n = int(input())

for i in range(n):
    i = ' ' * (n-i-1) + '*' * (i+1)
    
    print(i)