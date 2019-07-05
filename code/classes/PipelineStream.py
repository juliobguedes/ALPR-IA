import cv2, itertools
import numpy as np
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
    
  def score_filters(self, iou=False):
    # full_score = (index, score, rectangle)
    full_score = (-1, 0, ())
    
    # 16
    rectangles = [(x, y) for x in range(11, 93, 2) for y in range(9, 35, 2) if x > y]
    
    # 20
    areas = [(1400, 1500)]
    
    # 25
    ratios = [(3, 5)]
    
    pair = [self.filters, rectangles, areas, ratios] # 16 * 20 * 25 = 8000
    
    l = len(self.filters) * len(rectangles) * len(areas) * len(ratios) * len(self.images)
    print("Executing %d combinations" % l)
    i = 0
    
    for filt, rec, ar, rat in itertools.product(*pair):
      total = 0 if not iou else []
      for j in range(len(self.images)):
        img = self.images[j]
        filt.image = img
        filt.apply_and_draw2(rectangle=rec, show=False, area=ar, ratio=rat)
        score = filt.image.score(full=True) if not iou else filt.image.score_iou(full=True)
        if (score != 0):
            print(img.name, score)
        if (not iou):
            total += score
        else:
            total.append(score)
        
        i += 1
        if (i % 500 == 0):
          parcial = total if not iou else np.average(total)
          print("\nCurrent Progress: %d" % i)
          print("CurrentScore: %.2f" % parcial)
          print("MaxScore: %.2f" % full_score[1])
      
        if not iou:
            if (full_score[1] < total):
                full_score = (filt, total, rec, ar, rat)
        else:
            if (full_score[1] < np.average(total)):
                full_score = (filt, np.average(total), rec, ar, rat)
        
        
    return full_score
  
  def add_filter(self, filt):
    self.filters.append(filt)
    
  def extend_filters(self, filters):
    self.filters.extend(filters)