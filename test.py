import numpy as np
import cv2
from processing.processor import Processor

markerPath = 'resources/test1/marker.png'
targetPath = 'resources/test1/test-1.png'

markerImg = cv2.imread(markerPath, cv2.IMREAD_GRAYSCALE) 
targetImage = cv2.imread(targetPath, cv2.IMREAD_GRAYSCALE) 

processor = Processor()
processor.loadMarker(markerImg)

H = processor.calcHomography(targetImage)

# Выводим результат, чтобы убедиться, что всё верно
# Чтобы вывести результат, нам необходимо точки прямоугольника привести к однородным координатам и умножить на матрицу гомографии
height, width = markerImg.shape
resultImage = cv2.cvtColor(targetImage, cv2.COLOR_GRAY2BGR)
pointsRect = [
  [ 0, 0 ],
  [ 0, height ],
  [ width, height ],
  [ width, 0 ]
]
# Приводим к однородным координатам
pointsRect = list(map(lambda p: p + [1], pointsRect))

# Мы перемножаем матрицы, чтобы получить точки на конечном изображении 
targetPoints = np.matmul(pointsRect, H)
# Преобразование из однородных координат в обычные точки
targetPoints = list(map(lambda p: np.divide(p[:2], p[2]).astype(int).tolist(), targetPoints))
print(targetPoints)

for i in range(0, len(targetPoints)):
  j = i+1 if (i < len(targetPoints)-1) else 0
  cv2.line(resultImage, targetPoints[i], targetPoints[j], (0,0,255), 2)

cv2.imshow('img', resultImage)
cv2.waitKey(0)