import cv2


class Processor:
  
  def __init__(self):
    self.orb = cv2.ORB_create(500)
    

  def loadMarker (self, img):
    self.markerImage = img
    self.markerKeypoints, self.markerDescriptors = self.orb.detectAndCompute(img, None)
    print(f'Marker image loaded. Keypoints count: {len(self.markerKeypoints)}')

  def calcHomography(self, img):

    print("Calculating homography...")

    return [ 
      [ 1, 0, 0 ],
      [ 0, 1, 0 ],
      [ 0, 0, 1 ]
    ]

  