import numpy as np
import cv2
from processing.processor import Processor

markerPath = 'resources/test1/marker.png'
targetPath = 'resources/test1/test-1.png'

markerImg = cv2.imread(markerPath, cv2.IMREAD_GRAYSCALE) 
targetImage = cv2.imread(targetPath, cv2.IMREAD_GRAYSCALE) 

processor = Processor()
processor.loadMarker(markerImg)

H = processor.getHomography(targetImage)

# View results
height, width = markerImg.shape
resultImage = cv2.cvtColor(targetImage, cv2.COLOR_GRAY2BGR)
resultImage = cv2.line(resultImage, (0, 0), (100, 100), (0,0,255), 2)
print(targetImage)
cv2.imshow('img', targetImage)
cv2.waitKey(0)


