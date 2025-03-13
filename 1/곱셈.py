# 내가 푼 방식 너무 어렵게 풀음
# 그냥 간단하게 한자리씩 곱한다 생각하면 됐는데
# a = int(input())
# b = int(input())

# print(a * (b % 10))
# print(a * ((b % 100) // 10))
# print(a * (b // 100))
# print(a * b)

a = int(input())
b = input()  # 문자열로 받기!

print(a * int(b[2]))  # 1의 자리
print(a * int(b[1]))  # 10의 자리
print(a * int(b[0]))  # 100의 자리
print(a * int(b))     # 전체 곱셈