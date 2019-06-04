import cv2
from classes.Image import Image

class FilterSequence():
  def __init__(self, image=None, name=None, metadata=None):
    self.trueImage = image
    self.image = Image(image, name, metadata)
    self.filter_map = {
        'equalize': cv2.equalizeHist, 'average': cv2.blur,
        'gaussian': cv2.GaussianBlur, 'median': cv2.medianBlur,
        'bilateral': cv2.bilateralFilter, 'laplacian': cv2.Laplacian,
        'prewitt': cv2.filter2D, 'canny': cv2.Canny,
        'morph': cv2.morphologyEx, 'threshold': cv2.threshold,
        'adaptive': cv2.adaptiveThreshold, 'sobel': cv2.Sobel
    }
    self.filter_sequence = []
    
  def apply_filter(self, i):
    filt, config = self.filter_sequence[i]
    func = self.filter_map[filt]
    imn = 'src' if filt != 'canny' else 'image'
    config[imn] = self.image.image
    try:
      filtered = func(**config)
    except TypeError as e:
      print(filt)
      raise Exception(e)
      
    if (filt == 'threshold'):
      filtered = filtered[1]
    
    if (filt == 'equalize'):
      self.trueImage = filtered
    
    return filtered
  
  def apply_and_draw(self, rectangle=(17,13)):
    image = self.apply_all()
    img = image.image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, rectangle)
    dilated = cv2.dilate(img, kernel, 1)

    _, contours1, _ = cv2.findContours(dilated,
                                              cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)

    imgc1 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(imgc1, contours1, -1, (0, 255, 0), 2)
    image.show_image(imgc1)
    
  def apply_and_draw2(self, rectangle=(17,13), show=True, ratio=(0.3, 20), area=(2000, 23000)):
    image = self.apply_all()
    img = image.image
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, rectangle)
    dilated = cv2.dilate(img, kernel, 1)
    _, contours1, _ = cv2.findContours(dilated,
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
    
    color = cv2.cvtColor(self.trueImage, cv2.COLOR_GRAY2BGR)

    for cnt in contours1:
      epsilon = 0.05 * cv2.arcLength(cnt, True)
      approx = cv2.approxPolyDP(cnt,epsilon,True)

      if (len(approx) == 4):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(color, (x,y), (x+w, y+h), (255, 0, 0), 2)
        ar = 1.0 * h / 2

        if (ar >= ratio[0] and ar <= ratio[1] and h * w >= area[0] and h * w <= area[1]):
          self.image.add_sel_contour((x, y, w, h))
          cv2.rectangle(color, (x,y),(x+w, y+h), (0, 255, 0), 2)
        else:
          self.image.add_contour((x, y, w, h))
    if (show):
      image.show_image(color)
  
  def add_filter(self, name, config):
    self.filter_sequence.append((name, config))
    
  def plot_all_filters(self, hist=False):
    image = self.image
    
    if (hist):
      image.show_image_histogram()
    else:
      image.show_image()
    
    for i in range(len(self.filter_sequence)):
      filtered = self.apply_filter(i)
      if (hist):
        image.show_image_histogram(filtered)
      else:
        image.show_image(filtered)
        
  def apply_all(self):
    image = self.image
    
    for i in range(len(self.filter_sequence)):
      filtered = self.apply_filter(i)
      image = Image(filtered)
      
    return image
      
  def apply_in_sequence(self, hist=False):
    image = self.image
    
    if (hist):
      image.show_image_histogram()
    else:
      image.show_image()
    
    for i in range(len(self.filter_sequence)):
      filterred = self.apply_filter(i)
      image = Image(filterred)
      if (hist):
        image.show_image_histogram()
      else:
        image.show_image()
    return image
