# 윤년 = 1
# 윤년이 아님 = 0
# 윤년 = 연도가 4의배수, 100의 배수가 아님

y = int(input())

if (y % 4 == 0 and y % 100 !=0) or y % 400 == 0:
    print('1')

else:
    print('0')