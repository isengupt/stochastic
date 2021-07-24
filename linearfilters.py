import sys
import cv2 as cv
import numpy as np


class LinearFilter:
  def __init__(self,path):
        self.path = path

  def loadImage(self):
        self.src = cv.imread(cv.samples.findFile(self.path), cv.IMREAD_COLOR)
   
        if self.src is None:
            print('Image was not retrieved')

        else:
            print(self.src)

  def applyFilter(self):

      self.ddepth = -1
      ind = 0
      while True:
                
                kernel_size = 3 + 2 * (ind % 5)
                kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
                kernel /= (kernel_size * kernel_size)
                
                dst = cv.filter2D(self.src, self.ddepth, kernel)
                
                cv.imshow(window_name, dst)
                c = cv.waitKey(500)
                if c == 27:
                    break
                ind += 1
      return 0

      

window_name = 'filter2D'
l1 = LinearFilter('./image.png')
l1.loadImage()
l1.applyFilter()
