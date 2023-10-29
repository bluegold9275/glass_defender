import cv2
import numpy as np

# 이미지 로드
image = cv2.imread('desk_image.jpg')

# 그레이스케일로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 노이즈 감소를 위해 가우시안 블러 적용
gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 엣지 검출
edges = cv2.Canny(gray_blurred, 50, 150)

# 허프 변환을 사용하여 선 검출
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# 검출된 선을 그림과 함께 원본 이미지에 저장
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 검출된 선 좌표를 파일에 저장
with open('edge_coordinates.txt', 'w') as file:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        file.write(f'({x1}, {y1}) - ({x2}, {y2})\n')

# 이미지를 화면에 표시
cv2.imshow('Desk Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()