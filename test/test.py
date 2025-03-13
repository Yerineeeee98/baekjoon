import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

test_mask_path = "/mnt/data/20241110_150755_stem_0.png"
test_image = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)

#닫힘 연산 (Closing) 적용하여 끊어진 윤곽선 채움
kernel = np.ones((5, 5), np.uint8)
closed_mask = cv2.morphologyEx(test_image, cv2.MORPH_CLOSE, kernel, iterations=4)

# 노이즈 제거 (Opening : 열림 연산)
denoised_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iteratiyons=2)

contours, hierarchy = cv2.findContours(denoised_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 윤곽선 내부를 채울 마스크 생성
mask_filled = np.zeros_like(test_image)
if len(contours) > 0:
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:  # 바깥쪽 윤곽선만 유지
# 윤곽선 추출
            cv2.drawContours(mask_filled, [contours[i]], -1, 255, thickness=cv2.FILLED)

# 객체 내부가 255(흰색), 배경이 0(검은색)인지 확인 후 반전 ( 골격화 함수 때문 )
if np.mean(mask_filled) > 128:  # 배경이 255면 반전
    mask_filled = cv2.bitwise_not(mask_filled)

# 이진화 적용 (객체 내부만 골격화)
binary_mask = (mask_filled > 0).astype(np.uint8)  # 객체가 1, 배경이 0 ## ?(mask_filled > 0).

# Skeletonization 수행
skeleton_corrected = skeletonize(binary_mask) * 255  # 객체 내부만 골격화됨

# Skeleton 위의 모든 점 좌표 호출
skeleton_points = np.column_stack(np.where(skeleton_corrected > 0))

# 균일한 랜덤 샘플링 적용 (스켈레톤 위에서 300개 점 선택)
num_random_points = min(300, len(skeleton_points))
# 아래 다시 해보기
selected_points_random_uniform = skeleton_points[np.random.choice(len(skeleton_points), num_random_points, replace=False)]

 
# 윤곽선과 점 사이의 최단 거리 계산 (각 점별 참조 윤곽선 확인)
formatted_contours = [contour.astype(np.float32) for contour in contours]
distances_random_uniform = []
point_assigned_contours = []
for y, x in selected_points_random_uniform:
    # enumerate 찾아보기 전체 다시 한번 읽어보기
    point_distances = [(cv2.pointPolygonTest(contour, (float(x), float(y)), True), idx) for idx, contour in enumerate(formatted_contours)]
    min_dist, assigned_contour = min(point_distances, key=lambda t: abs(t[0]))  # 최소 거리 값 선택
    distances_random_uniform.append(abs(min_dist))  # 절대값 적용하여 음수 거리 제거
    point_assigned_contours.append(assigned_contour)  # 점이 참조한 윤곽선 인덱스 저장

distances_random_uniform = np.array(distances_random_uniform)

# 필터링 (74% 기준: 평균 거리의 74% 이하 제거)
mean_distance_random_uniform = np.mean(distances_random_uniform)
threshold_random_uniform = mean_distance_random_uniform * 0.74

# 기준보다 작은 거리의 점을 제거
filtered_points_random_uniform = selected_points_random_uniform[distances_random_uniform >= threshold_random_uniform]
removed_points = selected_points_random_uniform[distances_random_uniform < threshold_random_uniform]

# 필터링 후 평균 거리 다시 계산
new_mean_distance = np.mean(distances_random_uniform[distances_random_uniform >= threshold_random_uniform])

#  Skeleton 위에 필터링된 점과 제거된 점 시각화 (점마다 참조한 윤곽선 포함)
skeleton_with_removed_points = cv2.cvtColor(skeleton_corrected.astype(np.uint8), cv2.COLOR_GRAY2BGR)
cv2.drawContours(skeleton_with_removed_points, contours, -1, (0, 255, 0), 1)

# 필터링 후 남은 점 (74% 이상 거리 유지된 점)

# 아래 찾아보기 gpt
for i, (y, x) in enumerate(filtered_points_random_uniform):
    color = (255, point_assigned_contours[i] * 50 % 255, 0)
    cv2.circle(skeleton_with_removed_points, (x, y), 4, color, -1)

#  **필터링되어 제거된 점 (74% 이하 거리의 점)**
# 아래 찾아보기
for i, (y, x) in enumerate(removed_points):
    color = (255, point_assigned_contours[i] * 50 % 255, 165)
    cv2.circle(skeleton_with_removed_points, (x, y), 4, color, -1)

# 결과
plt.figure(figsize=(6, 6))
plt.imshow(skeleton_with_removed_points)
plt.title("Skeleton with Removed & Filtered Points (74% Threshold)")
plt.axis("off")
plt.show()

# 최종 줄기 굵기(Stem Thickness)
print(f"필터링 후 평균 거리: {new_mean_distance:.2f} 픽셀")
print(f"추정된 줄기 굵기(Stem Thickness): {2 * new_mean_distance:.2f} 픽셀")


# 비교 항목	첫 번째 코드 (stem_thickness)	두 번째 코드
# 폐색(closing) 연산 횟수	iterations=2	iterations=4 (더 강하게 연결)
# 열림(opening) 연산	iterations=2	iteratiyons=2 (오타 있음 → 실행 오류 발생)
# 윤곽선 검출 방식	cv2.RETR_EXTERNAL (바깥 윤곽선만)	cv2.RETR_TREE (계층 구조 유지)
# 윤곽선 내부 채우기	contours[0] (가장 큰 윤곽선만 사용)	hierarchy 이용해 모든 바깥 윤곽선 채움
# 배경 반전 조건	255 - mask 단순 반전	np.mean(mask) > 128 기준으로 판단 후 반전
# 샘플링된 점 수	300개 점	300개 점
# 윤곽선과 거리 계산	cv2.pointPolygonTest(contour, (x, y), True)	동일한 방식이지만 각 점이 참조한 윤곽선 정보까지 저장
# 거리 필터링 (74%)	거리 74% 이하 점 제거	점마다 참조한 윤곽선도 고려
# 결과 시각화	스켈레톤과 점 표시	필터링 후 남은 점(녹색), 제거된 점(주황색) 시각화