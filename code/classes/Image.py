import numpy as np
import cv2
from matplotlib import pyplot as plt

class Image():
  def __init__(self, image, name=None, metadata=None):
    self.image = image
    self.contours = []
    self.selected_contours = []
    self.name = name
    self.metadata = metadata
    
  def show_image(self, image = None):
    imshow = image if not (image is None) else self.image
    plt.figure(figsize=(30,12))
    plt.imshow(imshow, cmap='gray')
    plt.tick_params(bottom=False, labelbottom=False,
                    left=False, labelleft=False)
    
  def extend_contours(self, c, sc):
    self.contours.extend(c)
    self.selected_contours.extend(sc)
    
  def add_contour(self, c):
    self.contours.append(c)
    
  def add_sel_contour(self, sc):
    self.selected_contours.append(sc)
    
  def show_image_histogram(self, image = None):
    image = image if not (image is None) else self.image
    plt.figure(figsize=(30, 8))
    plt.subplot(1,2,1)
    plt.imshow(image, cmap="gray")
    plt.tick_params(bottom=False, labelbottom=False,
                    left=False, labelleft=False)

    nbins = 64
    h, bin_edges = np.histogram(image.ravel(), nbins, (0, 255))
    w = 256./nbins

    bin_centers = bin_edges[1:] - (w/2)
    plt.subplot(1,2,2)
    plt.bar(bin_centers, h, width=w)
    
  def crop_region(self, x, y, w, h):
    return self.image[y:y+h, x:x+w][:]

  def crop_plate(self, path=None, save=False):
    x, y, w, h = map(int, self.metadata['position_plate'].split(" "))
    cropped = self.crop_region(x, y, w, h)
    if (save and not path is None):
      cv2.imwrite(path+self.name+'.png', cropped)

  
  def draw_real_plate(self):
    if (self.metadata is None):
      raise Exception('This image\' metadata was not attached to the object')
      
    cpimage = np.array(self.image)
    cpimage = cv2.cvtColor(cpimage, cv2.COLOR_GRAY2BGR)
    x, y, w, h = map(int, self.metadata['position_plate'].split(" "))
    cv2.rectangle(cpimage, (x, y), (x+w, y+h), (0, 0, 255), 2)
    self.show_image(cpimage)
    
  def score(self, full=False):
    def intersection_size(x1, x2, w1, w2):
      x1_comeca = x1 <= x2
      indep = (x1 > x2 and x1 > x2+w2) or (x2 > x1 and x2 > x1+w1)
      res = 0

      if (indep): return res

      if (x1_comeca):
          if (x2 + w2 < x1 + w1):
              res = w1 - w2
          else:
              res = (x1+w1) - x2
      else:
          if (x2 + w2 > x1 > w1):
              res = w1
          else:
              res = (x2 + w2) - x1

      return abs(res)
    
    x, y, w, h = map(int, self.metadata['position_plate'].split(" "))
    true_area = w * h
    max_intersect = 0

    contours = self.selected_contours.copy()
    if (full): contours.extend(self.contours.copy())
    
    for sel in contours:
      x1, y1, w1, h1 = sel
      if ((w1 * h1) <= 2.5 * true_area):
        intersect_x = intersection_size(x, x1, w, w1)
        intersect_y = intersection_size(y, y1, h, h1)
        i_area = intersect_x * intersect_y
        max_intersect = i_area if i_area > max_intersect else max_intersect
        
    return max_intersect
