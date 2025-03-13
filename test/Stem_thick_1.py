import cv2
import numpy as np #
import matplotlib.pyplot as plt # Matplotlib 라이브러리를 이용해서 그래프를 그리는 일반적인 방법
from skimage.morphology import skeletonize # 이미지에서 객체(예: 글자나 형상)의 뼈대를 추출하는 함수

def stem_thickness(mask_image):
    """
    주어진 마스크 이미지를 사용하여 줄기의 두께를 분석하는 함수.

    :param mask_image: 마스크 이미지 (numpy array, grayscale)
    :return: 줄기의 추정된 두께(픽셀)
    """

    # 원본 마스크 시각화
    plt.figure(figsize=(5, 5)) # 그래프 크기 설정 (5 * 5 인치)
    plt.imshow(mask_image, cmap="gray") # `mask_image`를 흑백(gray)으로 표시
    plt.title("Original Mask") # 그래프 제목
    plt.axis("off") # 축(눈금) 제거
    plt.show() # 그래프 출력력

    # 윤곽선 검출 (Contours Detection)
    # mask_image에서 윤곽선을 찾고, 면적을 기준으로 정렬하는 과정정
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    # contours: 찾은 윤곽선들의 리스트, _는 "이 값을 사용하지 않겠다"는 의미
    # cv2.findContours()는 (contours 리스트, hierarchy 정보) 두 개를 반환
    # cv2.RETR_EXTERNAL: 바깥쪽 윤곽선만 찾음 (중첩된 내부 윤곽선 무시)
    # cv2.CHAIN_APPROX_SIMPLE: 윤곽선 꼭짓점만 저장 (불필요한 점 제거)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 면적 기준 정렬
    # 찾은 윤곽선들을 기준으로 정렬하는 부분
    # cv2.contourArea(contour): 주어진 윤곽선의 **면적(픽셀 수)**을 계산함.
    # sorted(..., reverse=True): 가장 큰 면적을 가진 윤곽선이 첫 번째 요소가 되도록 정렬.
    
    
    
    # 윤곽선 결과 확인
    contour_mask = np.zeros_like(mask_image)
    cv2.drawContours(contour_mask, contours, -1, 255, thickness=1)

    plt.figure(figsize=(5, 5))
    plt.imshow(contour_mask, cmap="gray")
    plt.title("Contours Detection")
    plt.axis("off")
    plt.show()

    # 닫힘 연산 (Closing) 적용하여 끊어진 선 연결
    # 이미지 전처리 (Closing & Noise 제거) > 줄기가 좀 더 매끄럽게 보정정
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    # MORPH_CLOSE (닫힘 연산): 끊어진 선 연결 (노이즈 제거)
    # MORPH_OPEN (열림 연산): 작은 노이즈 제거
    
    
    plt.figure(figsize=(5, 5))
    plt.imshow(closed_mask, cmap="gray")
    plt.title("After Closing Operation (Swapped Order)")
    plt.axis("off")
    plt.show()

    # 윤곽선 내부를 채운 마스크 생성
    mask_filled = np.zeros_like(mask_image)
    if len(contours) > 0:
        main_contour = contours[0]  # 가장 큰 윤곽선 선택
        cv2.drawContours(mask_filled, [main_contour], -1, 255, thickness=cv2.FILLED)

    plt.figure(figsize=(5, 5))
    plt.imshow(mask_filled, cmap="gray")
    plt.title("Filled Contour Mask (Swapped Order)")
    plt.axis("off")
    plt.show()

    # 작은 노이즈 제거 (Opening 연산)
    denoised_mask = cv2.morphologyEx(mask_filled, cv2.MORPH_OPEN, kernel, iterations=2)

    plt.figure(figsize=(5, 5))
    plt.imshow(denoised_mask, cmap="gray")
    plt.title("After Noise Removal")
    plt.axis("off")
    plt.show()

    # 배경 반전 (Skeletonization 전처리)
    corrected_binary_mask = 255 - denoised_mask
    # denoised_mask는 **배경이 하얗고(255), 줄기가 검은색(0)**일 수도 있어서, 
    # 배경을 반전해서 줄기가 하얀색(255)이 되도록 만들어야 됨
    # denoised_mask의 픽셀 값이 0이면 → 255로 변경 (검정 → 흰색)
    # denoised_mask의 픽셀 값이 255이면 → 0으로 변경 (흰색 → 검정)

    plt.figure(figsize=(5, 5))
    plt.imshow(corrected_binary_mask, cmap="gray")
    plt.title("Background Inverted Mask")
    plt.axis("off")
    plt.show()

    # Skeletonization (뼈대 추출) 수행
    binary_mask = corrected_binary_mask // 255 # 픽셀 값이 0또는 1이됨(이진화)
    skeleton_corrected = skeletonize(binary_mask) * 255
    # skeletonize(): 줄기의 중심선을 뼈대(Skeleton)로 변환
    # skeletonize() 함수는 "흰색(255)"인 부분을 중심선(뼈대)으로 줄이는 연산.
    # 즉, 객체(줄기)가 하얀색(255)이어야 하고, 배경이 검은색(0)이어야 정상 동작
    
#     
# ❌ 반전 전 (denoised_mask)
# 🔲 배경: 255 (흰색)
# 🌱 줄기: 0 (검은색)

# ✅ 반전 후 (corrected_binary_mask)
# 🔲 배경: 0 (검은색)
# 🌱 줄기: 255 (흰색)

# ⚡ 이진화 후 (binary_mask)
# 🔲 배경: 0
# 🌱 줄기: 1

# 🔥 스켈레톤 생성 후 (skeleton_corrected)
# 🔲 배경: 0
# 🌱 줄기 뼈대: 255

# 배경을 반전하는 이유는 skeletonize()가 "흰색(255)"인 영역을 뼈대로 변환하기 때문

    
    
    plt.figure(figsize=(5, 5))
    plt.imshow(skeleton_corrected, cmap="gray")
    plt.title("Skeletonized Mask")
    plt.axis("off")
    plt.show()

    # 스켈레톤 위의 모든 점 좌표 가져오기
    skeleton_points = np.column_stack(np.where(skeleton_corrected > 0))

    # 균일한 랜덤 샘플링 적용 (스켈레톤 위에서 300개 점 선택)
    # 뼈대 위의 점들을 300개 샘플링
    # 샘플링된 점과 윤곽선 간 최단 거리를 계산
    num_random_points = min(300, len(skeleton_points))
    selected_points_random_uniform = skeleton_points[np.random.choice(len(skeleton_points), num_random_points, replace=False)]

    # 윤곽선과 점 사이의 최단 거리 계산
    formatted_contours = [contour.astype(np.float32) for contour in contours]
    distances_random_uniform = []
    for y, x in selected_points_random_uniform:
        min_dist = np.min([cv2.pointPolygonTest(contour, (float(x), float(y)), True) for contour in formatted_contours])
        distances_random_uniform.append(abs(min_dist))  # 절대값 적용하여 음수 거리 제거
    # cv2.pointPolygonTest(): 각 점과 윤곽선 사이의 최단 거리 계산
    # abs(min_dist): 거리값을 절대값으로 변환
    # > 줄기 중심선에서 윤곽선까지의 거리 리스트를 얻음
    
    distances_random_uniform = np.array(distances_random_uniform)

    # 필터링 (74% 기준: 평균 거리의 74% 이하 제거) (노이즈: 너무 작은 거리 값값 제거)
    mean_distance_random_uniform = np.mean(distances_random_uniform)
    threshold_random_uniform = mean_distance_random_uniform * 0.74
    # > 보다 정확한 줄기 두께를 계산 
    
    # 기준보다 작은 거리의 점을 제거
    filtered_points_random_uniform = selected_points_random_uniform[distances_random_uniform >= threshold_random_uniform]
    filtered_distances_random_uniform = distances_random_uniform[distances_random_uniform >= threshold_random_uniform]

    # 필터링 후 평균 거리 다시 계산
    new_mean_distance = np.mean(filtered_distances_random_uniform)

    # Skeleton 위에 필터링된 점 시각화
    skeleton_with_random_uniform_points = cv2.cvtColor(skeleton_corrected.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(skeleton_with_random_uniform_points, contours, -1, (0, 255, 0), 1)
    for y, x in filtered_points_random_uniform:
        cv2.circle(skeleton_with_random_uniform_points, (x, y), 4, (255, 0, 0), -1)

    plt.figure(figsize=(5, 5))
    plt.imshow(skeleton_with_random_uniform_points)
    plt.title("Skeleton with Filtered Points")
    plt.axis("off")
    plt.show()

    # 최종 줄기 굵기(Stem Thickness) 계산
    stem_thickness = 2 * new_mean_distance
    print(f"필터링 후 평균 거리: {new_mean_distance:.2f} 픽셀")
    print(f"추정된 줄기 굵기(Stem Thickness): {stem_thickness:.2f} 픽셀")

    return stem_thickness

# 이미지 로드
mask_path = "/mnt/data/20241110_150948_stem_14_mask.png"
mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# 줄기 두께 분석 실행
thickness_result = stem_thickness(mask_image)

# 결과 출력
print(f"🌱 최종 줄기 굵기: {thickness_result:.2f} 픽셀")
