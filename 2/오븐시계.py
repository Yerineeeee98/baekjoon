A, B = map(int, input().split())
C = int(input())

if B + C < 60:
    print(A, B + C)

else:
    if A + (B + C) // 60 >= 24:
        print(A + (B + C) // 60 - 24 , (B + C) % 60 )
    
    elif (B + C) // 60 >=1 :
        print((A + (B + C) // 60), (B + C) % 60)
        
        
# 더 간단한 표현식        
A, B = map(int, input().split())
C = int(input())

total_minutes = B + C # 나는 일일히 B + C를 했는데 변수 하나로 처리
A += total_minutes // 60 # if로 나누지 않고 A를 한번에 처리
B = total_minutes % 60 # B를 일일히 적지 않고 한번에 처리
A %= 24  # 24시가 넘어가면 0시로 돌아가게 처리

print(A, B)
    
    