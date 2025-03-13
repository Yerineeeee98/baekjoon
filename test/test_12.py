# 1. 이미지 로드 및 전처리 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

# 이미지 로드 (흑백 이미지로)
test_mask_path = "/mnt/data/20241110_150755_stem_0.png"
test_image = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)

#  2. 이미지 전처리 (노이즈 제거 & 윤곽선 찾기)
# 닫힘 연산 (끊어진 선 연결)
kernel = np.ones((5, 5), np.uint8)
closed_mask = cv2.morphologyEx(test_image, cv2.MORPH_CLOSE, kernel, iterations=4)

# 열림 연산 (노이즈 제거)
denoised_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=2)

# 윤곽선 찾기
contours, hierarchy = cv2.findContours(denoised_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#  3. 윤곽선 내부 채우기 & 이진화 - 객체 내부를 채워서 골격화 준비
mask_filled = np.zeros_like(test_image)
if len(contours) > 0:
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:  # 바깥쪽 윤곽선만
            cv2.drawContours(mask_filled, [contours[i]], -1, 255, thickness=cv2.FILLED)

# 배경 반전 (골격화 함수 때문)
if np.mean(mask_filled) > 128:
    mask_filled = cv2.bitwise_not(mask_filled)

# 이진화 적용 (객체가 1, 배경이 0)
binary_mask = (mask_filled > 0).astype(np.uint8)

#  4. Skeletonization (골격화) 및 랜덤 샘플링 - 줄기의 중심 골격 추출
# 골격화
skeleton_corrected = skeletonize(binary_mask) * 255

# Skeleton 위의 모든 점 좌표 호출
skeleton_points = np.column_stack(np.where(skeleton_corrected > 0))

# 균일한 랜덤 샘플링 (최대 300개 점)
num_random_points = min(300, len(skeleton_points))
selected_points_random_uniform = skeleton_points[np.random.choice(len(skeleton_points), num_random_points, replace=False)]

# 5. 점-윤곽선 거리 계산 & 필터링 - 골격 위의 점들과 윤곽선 사이의 거리 측정
# 평균 거리의 74% 이하인 점 제거 → 줄기의 외곽 잡음 제거
formatted_contours = [contour.astype(np.float32) for contour in contours]
distances_random_uniform = []
point_assigned_contours = []

for y, x in selected_points_random_uniform:
    point_distances = [(cv2.pointPolygonTest(contour, (float(x), float(y)), True), idx) for idx, contour in enumerate(formatted_contours)]
    min_dist, assigned_contour = min(point_distances, key=lambda t: abs(t[0]))
    distances_random_uniform.append(abs(min_dist))
    point_assigned_contours.append(assigned_contour)

distances_random_uniform = np.array(distances_random_uniform)

# 거리 필터링 (74% 기준)
mean_distance_random_uniform = np.mean(distances_random_uniform)
threshold_random_uniform = mean_distance_random_uniform * 0.74

filtered_points_random_uniform = selected_points_random_uniform[distances_random_uniform >= threshold_random_uniform]
removed_points = selected_points_random_uniform[distances_random_uniform < threshold_random_uniform]

#  6. 결과 시각화 & 두께 계산 - 최종 골격과 점을 시각화하고 줄기의 평균 두께 추정
# 필터링된 점, 제거된 점 시각화
skeleton_with_removed_points = cv2.cvtColor(skeleton_corrected.astype(np.uint8), cv2.COLOR_GRAY2BGR)
cv2.drawContours(skeleton_with_removed_points, contours, -1, (0, 255, 0), 1)

# 남은 점 (녹색) 시각화 - 
for i, (y, x) in enumerate(filtered_points_random_uniform):
    color = (255, point_assigned_contours[i] * 50 % 255, 0)
    cv2.circle(skeleton_with_removed_points, (x, y), 4, color, -1)

# 제거된 점 (주황색) 시각화
for i, (y, x) in enumerate(removed_points):
    color = (255, point_assigned_contours[i] * 50 % 255, 165)
    cv2.circle(skeleton_with_removed_points, (x, y), 4, color, -1)

# 시각화 출력
plt.figure(figsize=(6, 6))
plt.imshow(skeleton_with_removed_points)
plt.title("Skeleton with Removed & Filtered Points (74% Threshold)")
plt.axis("off")
plt.show()

# 최종 줄기 두께 계산
new_mean_distance = np.mean(distances_random_uniform[distances_random_uniform >= threshold_random_uniform])
print(f"필터링 후 평균 거리: {new_mean_distance:.2f} 픽셀")
print(f"추정된 줄기 굵기(Stem Thickness): {2 * new_mean_distance:.2f} 픽셀")
