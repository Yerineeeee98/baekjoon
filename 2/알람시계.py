H, M = map(int, input().split())

if M - 45 >= 0:
    print(H, M - 45)
    
# elif H - 1 < 0:
#     print(23, M +15)

# elif M - 45 < 0:
#     print(H -1, M +15)
    
# 이 부분을
else:
    if H == 0:
        print(23, M + 15)
    else:
        print(H - 1, M + 15)
# 로 처리 가능 else문 안에 if문이 중북될수있고 