import cv2, itertools
from classes.Image import Image
from classes.FilterSequence import FilterSequence

class PipelineStream():
  def __init__(self, images, names, metadata):
    self.images = []
    for i in range(len(images)):
      temp_m = metadata['training'][names[i]]
      temp = Image(images[i], name=names[i], metadata=temp_m)
      self.images.append(temp)
    self.filters = []
    
  def score_filters(self):
    # full_score = (index, score, rectangle)
    full_score = (-1, 0, ())
    
    # 16
    rectangles = [(x, y) for x in range(11, 18, 2) for y in range(5, 12, 2) if x > y]
    
    # 20
    areas = [(a_min * 100, a_max * 1000) for a_min in range(17, 24, 2) for a_max in range(17, 26, 2)]
    
    # 25
    ratios = [(r_min/100, r_max) for r_min in range(1, 151, 45) for r_max in range(7, 64, 27)]
    
    pair = [self.filters, rectangles, areas, ratios] # 16 * 20 * 25 = 8000
    
    l = len(self.filters) * len(rectangles) * len(areas) * len(ratios) * 8
    print("Executing %d combinations" % l)
    i = 0
    
    for filt, rec, ar, rat in itertools.product(*pair):
      total = 0
      for j in range(len(self.images)):
        img = self.images[j]
        filt.image = img
        filt.apply_and_draw2(rectangle=rec, show=False, area=ar, ratio=rat)
        score = filt.image.score()
        total += score
        
        i += 1
        if (i % 500 == 0): print("Current Progress: %d" % i)
        if (j > 7): break
      
      if (full_score[1] < total):
        full_score = (filt, total, rec, ar, rat)
        
    return full_score
  
  def add_filter(self, filt):
    self.filters.append(filt)
    
  def extend_filters(self, filters):
    self.filters.extend(filters)