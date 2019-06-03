#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import cv2
# import tensorflow as tf
# from tensorflow import keras

import matplotlib as mpl
from matplotlib import pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

import os, json, itertools


# In[2]:

from ..classes.Image import Image
from ..classes.FilterSequence import FilterSequence
from ..classes.PipelineStream import PipelineStream

# from google.colab import files, drive
# drive.mount('/content/drive')
filepath = u'/content/drive/My Drive/IA/Projeto/sample_data/'


# In[ ]:


metadata = json.load(open('metadata_db.json', 'r'))


# In[ ]:


arq_imagens = [png for png in os.listdir(filepath) if png.endswith(".png")]
ext_imagens = [nn.replace(".png", "") for nn in arq_imagens]
imagens_originais = []
imagens_grayscale = []

for i in range(len(arq_imagens)):
  aimg = arq_imagens[i]
  img = cv2.imread(filepath + aimg)
  imagens_originais.append(img)
  imagens_grayscale.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))


# In[ ]:


plt.figure(figsize=(30,12))
plt.imshow(imagens_grayscale[0], cmap="gray")
plt.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)


# ### Filtering
# (TCC)

# In[ ]:


teste = imagens_grayscale[1]

plt.figure(figsize=(20, 10))
plt.subplot(2,3,1)
plt.title("Original")
plt.imshow(teste, cmap="gray")

nbins = 64
h, bin_edges = np.histogram(teste.ravel(), nbins, (0, 255))
w = 256./nbins

bin_centers = bin_edges[1:] - (w/2)
plt.subplot(2,3,4)
plt.title("Histograma Original")
plt.bar(bin_centers, h, width=w)

###################################################################

plt.subplot(2,3,2)
teste_media = cv2.blur(teste, (5,5))
plt.title("Filtro de média")
plt.imshow(teste_media, cmap="gray")

nbins = 64
h, bin_edges = np.histogram(teste_media.ravel(), nbins, (0, 255))
w = 256 / nbins

bin_centers = bin_edges[1:] - (w/2)
plt.subplot(2,3,5)
plt.title("Histograma de média")
plt.bar(bin_centers, h, width=w)

###################################################################

plt.subplot(2,3,3)
teste_eql = cv2.equalizeHist(teste_media, (5,5))
plt.title("Imagem Equalizada")
plt.imshow(teste_eql, cmap="gray")

nbins = 64
h, bin_edges = np.histogram(teste_eql.ravel(), nbins, (0, 255))
w = 256 / nbins

bin_centers = bin_edges[1:] - (w/2)
plt.subplot(2,3,6)
plt.title("Histograma Equalizado")
plt.bar(bin_centers, h, width=w)


# Testando outros blurs

# In[ ]:


plt.figure(figsize=(30,12))
plt.imshow(teste, cmap="gray")
plt.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)


# Embaçando imagens?

# laplaciano

# In[ ]:


laplacian = cv2.Laplacian(teste_eql, cv2.CV_64F)
laplacian = cv2.normalize(laplacian, None, alpha = 0, beta = 255,
                          norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)


# prewitt

# In[ ]:


kernelx = np.array([[1,1,1], [0,0,0], [-1, -1, -1]])
kernely = np.array([[-1,0,1], [-1,0,1], [-1, 0, 1]])
img_prewittx = cv2.filter2D(teste_eql, -1, kernelx)
img_prewitty = cv2.filter2D(teste_eql, -1, kernely)


# sobel

# In[ ]:


sobelx = cv2.Sobel(teste_eql, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(teste_eql, cv2.CV_64F, 0, 1, ksize=3)
t1, t2 = np.uint8(np.absolute(sobelx)), np.uint8(np.absolute(sobely))
true_sobel = cv2.bitwise_or(t1, t2)
sobel_mag = (sobelx**2 + sobely**2) ** 0.5
sobel_ang = np.rad2deg(np.angle(sobelx+sobely*1j))
sobel_ang = (sobel_ang >= 0) * sobel_ang + (sobel_ang < 0) * (sobel_ang + 180)
sobel_ang = (sobel_ang < 180) * sobel_ang


# In[ ]:


sobelxn = cv2.normalize(sobelx, None, alpha = 0, beta = 255,
                          norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)
sobelyn = cv2.normalize(sobely, None, alpha = 0, beta = 255,
                          norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)
sobel_magn = cv2.normalize(sobel_mag, None, alpha = 0, beta = 255,
                          norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)
sobel_angn = cv2.normalize(sobel_ang, None, alpha = 0, beta = 255,
                          norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)


# canny e gradiente morfologico

# In[ ]:


canny = cv2.Canny(teste_eql, 100, 300)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
morphg = cv2.morphologyEx(teste_eql, cv2.MORPH_GRADIENT, kernel)


# plots

# In[ ]:


plt.figure(figsize=(30, 12))

plt.subplot(3,3,1)
plt.imshow(laplacian, cmap="gray")

plt.subplot(3,3,2)
plt.imshow(img_prewittx, cmap="gray")

plt.subplot(3,3,3)
plt.imshow(img_prewitty, cmap="gray")

plt.subplot(3,3,4)
plt.imshow(sobelxn, cmap="gray")

plt.subplot(3,3,5)
plt.imshow(sobelyn, cmap="gray")

plt.subplot(3,3,6)
plt.imshow(sobel_magn, cmap="gray")

plt.subplot(3,3,7)
plt.imshow(sobel_angn, cmap="gray")

plt.subplot(3,3,8)
plt.imshow(canny, cmap="gray")

plt.subplot(3,3,9)
plt.imshow(morphg, cmap="gray")


# In[ ]:


plt.figure(figsize=(30,12))
plt.imshow(canny, cmap="gray")


# ### Threshold
# 
# empirico

# In[ ]:


plt.figure(figsize=(30,45))

for i in range(9):
  limiar = int(255*(i+1)/10.0)
  plt.subplot(9, 3, i*3+1)
  _, xthr = cv2.threshold(sobelxn, limiar, 255, cv2.THRESH_BINARY)
  plt.imshow(xthr, cmap="gray")
  plt.subplot(9, 3, (i*3)+2)
  _, ythr = cv2.threshold(sobelxn, limiar, 255, cv2.THRESH_BINARY)
  plt.imshow(ythr, cmap="gray")
  plt.subplot(9, 3, (i*3)+3)
  _, mthr = cv2.threshold(sobelxn, limiar, 255, cv2.THRESH_BINARY)
  plt.imshow(mthr, cmap="gray")


# global

# In[ ]:


_, xthrg = cv2.threshold(sobelxn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, ythrg = cv2.threshold(sobelyn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, mthrg = cv2.threshold(sobel_magn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, true_thr = cv2.threshold(true_sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(30,12))
plt.subplot(2,2,1)
plt.imshow(xthrg, cmap="gray")
plt.subplot(2,2,2)
plt.imshow(ythrg, cmap="gray")
plt.subplot(2,2,3)
plt.imshow(mthrg, cmap="gray")
plt.subplot(2,2,4)
plt.imshow(true_thr, cmap="gray")


# otsu

# In[ ]:


xthr = cv2.adaptiveThreshold(sobelxn, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
ythr = cv2.adaptiveThreshold(sobelyn, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
mthr = cv2.adaptiveThreshold(sobel_magn, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)

plt.figure(figsize=(30,12))
plt.subplot(1,3,1)
plt.imshow(xthr, cmap="gray")
plt.subplot(1,3,2)
plt.imshow(ythr, cmap="gray")
plt.subplot(1,3,3)
plt.imshow(mthr, cmap="gray")


# GAUSSIAN?

# In[ ]:


xthr = cv2.adaptiveThreshold(sobelxn, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0)
ythr = cv2.adaptiveThreshold(sobelyn, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0)
mthr = cv2.adaptiveThreshold(sobel_magn, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0)

plt.figure(figsize=(30,12))
plt.subplot(1,3,1)
plt.imshow(xthr, cmap="gray")
plt.subplot(1,3,2)
plt.imshow(ythr, cmap="gray")
plt.subplot(1,3,3)
plt.imshow(mthr, cmap="gray")


# contornos

# In[ ]:


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,8))
morphDx = cv2.dilate(canny, kernel, 1)

_, contours1, hierarch = cv2.findContours(morphDx, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img_cont1 = cv2.cvtColor(teste_eql, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img_cont1, contours1, -1, (0, 255, 0), 3)
plt.figure(figsize=(30,12))
plt.imshow(img_cont1, cmap="gray")


# In[ ]:


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,11))
morphDy = cv2.dilate(ythr, kernel, 1)

_, contours2, hierarch = cv2.findContours(morphDy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img_cont2 = cv2.cvtColor(teste, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img_cont2, contours2, -1, (0, 255, 0), 3)
plt.figure(figsize=(30,12))
plt.imshow(img_cont2, cmap="gray")


# In[ ]:


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8,9))
morphDm = cv2.dilate(mthr, kernel, 1)

_, contours3, hierarch = cv2.findContours(morphDm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img_cont3 = cv2.cvtColor(teste, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img_cont3, contours3, -1, (0, 255, 0), 3)
plt.figure(figsize=(30,12))
plt.imshow(img_cont3, cmap="gray")


# In[ ]:


color = cv2.cvtColor(teste_eql, cv2.COLOR_GRAY2BGR)

for cnt in contours1:
  epsilon = 0.05 * cv2.arcLength(cnt, True)
  approx = cv2.approxPolyDP(cnt,epsilon,True)
  
  if (len(approx) == 4):
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(color, (x,y), (x+w, y+h), (255, 0, 0), 2)
    ar = 1.0 * h / 2

    if (ar >= 0.3 and ar <= 20 and h * w >= 2000 and h * w <= 23000):
      print (ar, h, w, h*w)
      cv2.rectangle(color, (x,y),(x+w, y+h), (0, 255, 0), 8)
      
plt.figure(figsize=(30,12))
plt.imshow(color, cmap="gray")


# abordagem 2

# In[ ]:


minArea = 2700
maxArea = 20000

thrs = []
axthrs = []
fxthrs = []
coloreds = []

plt.figure(figsize=(30, 12))

n = len(imagens_grayscale)

for i in range(n):
  imgi = imagens_grayscale[i]
  imgi_eq = cv2.equalizeHist(imgi)
  
  kernelTh = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
  morphTh = cv2.morphologyEx(imgi_eq, cv2.MORPH_TOPHAT, kernelTh)
  _, thr = cv2.threshold(morphTh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  
  thrs.append(thr)
  
  kernelAx = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11))
  morphAx = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernelAx)
  _, axthr = cv2.threshold(morphAx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  
  axthrs.append(axthr)
  
  kernelFx = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 1))
  morphFx = cv2.morphologyEx(morphAx, cv2.MORPH_CLOSE, kernelFx)
  _, fxthr = cv2.threshold(morphFx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  
  fxthrs.append(fxthr)

  _, contours4, hierarch = cv2.findContours(morphFx, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  color = cv2.cvtColor(imgi_eq, cv2.COLOR_GRAY2BGR)
  
  coloreds.append(color)
  
  for cnt in contours4:
    epsilon = 0.05 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    if (len(approx) == 4):
      x,y,w,h = cv2.boundingRect(cnt)
      ar = 1.0 * h / w
      
      if (ar >= 0.3 and ar <= 0.55 and h*w > minArea and h*w < maxArea):
        cv2.rectangle(color, (x,y), (x+w, y+h), (255, 0, 0), 2)
  plt.subplot(2, 4, i+1)
  plt.imshow(color, cmap="gray")


# In[ ]:


plt.figure(figsize=(30,12))
plt.imshow(coloreds[6], cmap="gray")


# ### Segmentação
# 
# [video](https://www.youtube.com/watch?v=STnoJ3YCWus)
# 
# 

# In[ ]:


import skimage
from skimage import filters, feature


# In[ ]:


testex = imagens_grayscale[1]
denoised = filters.median(testex, selem=np.ones((5,5)))
f, (ax0, ax1) = plt.subplots(1, 2, figsize=(15,5))
ax0.imshow(testex)
ax1.imshow(denoised)


# In[ ]:


edges = feature.canny(testex, sigma=3)
plt.figure(figsize=(30,12))
plt.imshow(edges)


# In[ ]:


from scipy.ndimage import distance_transform_edt
dt = distance_transform_edt(~edges)

plt.figure(figsize=(30,12))
plt.imshow(dt)


# In[ ]:


local_max = feature.peak_local_max(dt, indices=False, min_distance=1)
plt.figure(figsize=(30,12))
plt.imshow(local_max, cmap="gray")


# In[ ]:





# ### Criando Pipeline

# In[ ]:



# In[ ]:



# ### Usando Pipeline

# In[ ]:


ind = 9
real_seq = FilterSequence(
    imagens_grayscale[ind],
    name=ext_imagens[ind],
    metadata=metadata['training'][ext_imagens[ind]]
)
real_seq.add_filter('gaussian', { 'ksize': (3,3), 'sigmaX': 8 })
real_seq.add_filter('equalize', {  })
# real_seq.add_filter('laplacian', { 'ddepth': cv2.CV_64F, 'ksize': 5 })
# real_seq.add_filter('sobel', {
#     'ddepth': cv2.CV_64F, 'ksize': 5, 'dx': 1, 'dy': 0
# })
real_seq.add_filter('canny', {
    'threshold1': 100, 'threshold2': 200, 'apertureSize': 3
})
# real_seq.add_filter('threshold', {
#     'thresh': 0, 'maxval': 255, 'type': cv2.THRESH_BINARY + cv2.THRESH_OTSU
# })
# real_seq.add_filter('adaptive', {
#     'maxValue': 255, 'adaptiveMethod': cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     'thresholdType': cv2.THRESH_BINARY, 'blockSize': 15, 'C': 2
# })


# In[ ]:


real_seq.apply_and_draw2(rectangle=(13, 9))
# real_seq.apply_and_draw(rectangle=(17,5))


# In[ ]:


real_seq.image.draw_real_plate()


# In[65]:


real_seq.image.score()


# In[ ]:




# In[ ]:


f1 = FilterSequence(
    imagens_grayscale[ind],
    name=ext_imagens[ind],
    metadata=metadata['training'][ext_imagens[ind]])
f1.add_filter('gaussian', { 'ksize': (3,3), 'sigmaX': 0 })
f1.add_filter('equalize', {  })
f1.add_filter('canny', { 'threshold1': 100, 'threshold2': 200, 'apertureSize': 3 })
f1.apply_and_draw2()


# In[ ]:


f2 = FilterSequence(
    imagens_grayscale[ind],
    name=ext_imagens[ind],
    metadata=metadata['training'][ext_imagens[ind]])
f2.add_filter('gaussian', { 'ksize': (5,5), 'sigmaX': 0 })
f2.add_filter('equalize', {  })
f2.add_filter('canny', { 'threshold1': 100, 'threshold2': 200, 'apertureSize': 3 })
f2.apply_and_draw2()


# In[ ]:


f3 = FilterSequence(
    imagens_grayscale[ind],
    name=ext_imagens[ind],
    metadata=metadata['training'][ext_imagens[ind]])
f3.add_filter('gaussian', { 'ksize': (3,3), 'sigmaX': 5 })
f3.add_filter('equalize', {  })
f3.add_filter('canny', { 'threshold1': 100, 'threshold2': 200, 'apertureSize': 3 })
f3.apply_and_draw2()


# In[ ]:


f4 = FilterSequence(
    imagens_grayscale[ind],
    name=ext_imagens[ind],
    metadata=metadata['training'][ext_imagens[ind]])
f4.add_filter('gaussian', { 'ksize': (5,5), 'sigmaX': 5 })
f4.add_filter('equalize', {  })
f4.add_filter('canny', { 'threshold1': 100, 'threshold2': 200, 'apertureSize': 3 })
f4.apply_and_draw2()


# In[ ]:


f5 = FilterSequence(
    imagens_grayscale[ind],
    name=ext_imagens[ind],
    metadata=metadata['training'][ext_imagens[ind]])
f5.add_filter('average', { 'ksize': (3,3) })
f5.add_filter('equalize', {  })
f5.add_filter('canny', { 'threshold1': 100, 'threshold2': 200, 'apertureSize': 3 })
f5.apply_and_draw2()


# In[ ]:


f6 = FilterSequence(
    imagens_grayscale[ind],
    name=ext_imagens[ind],
    metadata=metadata['training'][ext_imagens[ind]])
f6.add_filter('average', { 'ksize': (5,5) })
f6.add_filter('equalize', {  })
f6.add_filter('canny', { 'threshold1': 100, 'threshold2': 200, 'apertureSize': 3 })
f6.apply_and_draw2()


# In[ ]:


f7 = FilterSequence(
    imagens_grayscale[ind],
    name=ext_imagens[ind],
    metadata=metadata['training'][ext_imagens[ind]])
f7.add_filter('average', { 'ksize': (7,7) })
f7.add_filter('equalize', {  })
f7.add_filter('canny', { 'threshold1': 100, 'threshold2': 200, 'apertureSize': 3 })
f7.apply_and_draw2()


# In[ ]:


f8 = FilterSequence(
    imagens_grayscale[ind],
    name=ext_imagens[ind],
    metadata=metadata['training'][ext_imagens[ind]])
f8.add_filter('average', { 'ksize': (7,7) })
f8.add_filter('equalize', {  })
k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
f8.add_filter('morph', { 'op': cv2.MORPH_GRADIENT, 'kernel': k1 })
f8.apply_and_draw2()


# In[ ]:


f9 = FilterSequence(
    imagens_grayscale[ind],
    name=ext_imagens[ind],
    metadata=metadata['training'][ext_imagens[ind]])
f9.add_filter('average', { 'ksize': (3,3) })
f9.add_filter('equalize', {  })
k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
f9.add_filter('morph', { 'op': cv2.MORPH_GRADIENT, 'kernel': k2 })
f9.apply_and_draw2()


# In[ ]:


fA = FilterSequence(
    imagens_grayscale[ind],
    name=ext_imagens[ind],
    metadata=metadata['training'][ext_imagens[ind]])
fA.add_filter('average', { 'ksize': (5,5) })
fA.add_filter('equalize', {  })
k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
fA.add_filter('morph', { 'op': cv2.MORPH_GRADIENT, 'kernel': k3 })
fA.apply_and_draw2()


# In[ ]:


fB = FilterSequence(
    imagens_grayscale[ind],
    name=ext_imagens[ind],
    metadata=metadata['training'][ext_imagens[ind]])
fB.add_filter('average', { 'ksize': (7,7) })
fB.add_filter('equalize', {  })
k4 = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
fB.add_filter('morph', { 'op': cv2.MORPH_GRADIENT, 'kernel': k4 })
fB.apply_and_draw2()


# In[ ]:


fC = FilterSequence(
    imagens_grayscale[ind],
    name=ext_imagens[ind],
    metadata=metadata['training'][ext_imagens[ind]])
fC.add_filter('gaussian', { 'ksize': (5,5), 'sigmaX': 5 })
fC.add_filter('equalize', {  })
k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
fC.add_filter('morph', { 'op': cv2.MORPH_GRADIENT, 'kernel': k5 })
fC.apply_and_draw2()


# In[ ]:


fD = FilterSequence(
    imagens_grayscale[ind],
    name=ext_imagens[ind],
    metadata=metadata['training'][ext_imagens[ind]])
fD.add_filter('gaussian', { 'ksize': (3,3), 'sigmaX': 0 })
fD.add_filter('equalize', {  })
fD.add_filter('canny', { 'threshold1': 100, 'threshold2': 300, 'apertureSize': 3 })
fD.apply_and_draw2()


# In[ ]:


fE = FilterSequence(
    imagens_grayscale[ind],
    name=ext_imagens[ind],
    metadata=metadata['training'][ext_imagens[ind]])
fE.add_filter('gaussian', { 'ksize': (3,3), 'sigmaX': 0 })
fE.add_filter('equalize', {  })
fE.add_filter('canny', { 'threshold1': 200, 'threshold2': 300, 'apertureSize': 3 })
fE.apply_and_draw2()


# In[ ]:


fE = FilterSequence(
    imagens_grayscale[ind],
    name=ext_imagens[ind],
    metadata=metadata['training'][ext_imagens[ind]])
fE.add_filter('gaussian', { 'ksize': (3,3), 'sigmaX': 0 })
fE.add_filter('equalize', {  })
fE.add_filter('canny', { 'threshold1': 100, 'threshold2': 300, 'apertureSize': 5 })
fE.apply_and_draw2()


# In[ ]:


fF = FilterSequence(
    imagens_grayscale[ind],
    name=ext_imagens[ind],
    metadata=metadata['training'][ext_imagens[ind]])
fF.add_filter('gaussian', { 'ksize': (3,3), 'sigmaX': 0 })
fF.add_filter('equalize', {  })
fF.add_filter('canny', { 'threshold1': 50, 'threshold2': 200, 'apertureSize': 3 })
fF.apply_and_draw2()


# In[ ]:


f10 = FilterSequence(
    imagens_grayscale[ind],
    name=ext_imagens[ind],
    metadata=metadata['training'][ext_imagens[ind]])
f10.add_filter('gaussian', { 'ksize': (3,3), 'sigmaX': 0 })
f10.add_filter('equalize', {  })
f10.add_filter('canny', { 'threshold1': 50, 'threshold2': 300, 'apertureSize': 5 })
f10.apply_and_draw2()


# In[ ]:


f11 = FilterSequence(
    imagens_grayscale[ind],
    name=ext_imagens[ind],
    metadata=metadata['training'][ext_imagens[ind]])
f11.add_filter('gaussian', { 'ksize': (5,5), 'sigmaX': 0 })
f11.add_filter('equalize', {  })
f11.add_filter('canny', { 'threshold1': 100, 'threshold2': 300, 'apertureSize': 3 })
f11.apply_and_draw2()


# In[ ]:


f12 = FilterSequence(
    imagens_grayscale[ind],
    name=ext_imagens[ind],
    metadata=metadata['training'][ext_imagens[ind]])
f12.add_filter('gaussian', { 'ksize': (3,3), 'sigmaX': 0 })
f12.add_filter('equalize', {  })
f12.add_filter('canny', { 'threshold1': 150, 'threshold2': 250, 'apertureSize': 3 })
f12.apply_and_draw2()


# In[ ]:


f13 = FilterSequence(
    imagens_grayscale[ind],
    name=ext_imagens[ind],
    metadata=metadata['training'][ext_imagens[ind]])
f13.add_filter('gaussian', { 'ksize': (3,3), 'sigmaX': 0 })
f13.add_filter('equalize', {  })
f13.add_filter('sobel', { 'ddepth': cv2.CV_64F, 'dx': 1, 'dy': 0, 'ksize': 3 })
f13.add_filter('threshold', {
    'thresh': 0, 'maxval': 255, 'type': cv2.THRESH_BINARY + cv2.THRESH_OTSU
})
f13.apply_and_draw2()


# In[ ]:


f14 = FilterSequence(
    imagens_grayscale[ind],
    name=ext_imagens[ind],
    metadata=metadata['training'][ext_imagens[ind]])
f14.add_filter('gaussian', { 'ksize': (3,3), 'sigmaX': 0 })
f14.add_filter('equalize', {  })
f14.add_filter('sobel', { 'ddepth': cv2.CV_64F, 'dx': 1, 'dy': 0, 'ksize': 3 })
f14.add_filter('adaptive', {
    'maxValue': 255, 'adaptiveMethod': cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    'thresholdType': cv2.THRESH_BINARY, 'blockSize': 15, 'C': 2
})
f14.apply_and_draw2()


# In[ ]:


f15 = FilterSequence(
    imagens_grayscale[ind],
    name=ext_imagens[ind],
    metadata=metadata['training'][ext_imagens[ind]])
f15.add_filter('gaussian', { 'ksize': (5,5), 'sigmaX': 0 })
f15.add_filter('equalize', {  })
f15.add_filter('sobel', { 'ddepth': cv2.CV_64F, 'dx': 1, 'dy': 0, 'ksize': 3 })
f15.add_filter('threshold', {
    'thresh': 0, 'maxval': 255, 'type': cv2.THRESH_BINARY + cv2.THRESH_OTSU
})
f15.apply_and_draw2()


# In[ ]:


f16 = FilterSequence(
    imagens_grayscale[ind],
    name=ext_imagens[ind],
    metadata=metadata['training'][ext_imagens[ind]])
f16.add_filter('gaussian', { 'ksize': (3,3), 'sigmaX': 0 })
f16.add_filter('equalize', {  })
f16.add_filter('sobel', { 'ddepth': cv2.CV_64F, 'dx': 1, 'dy': 0, 'ksize': 3 })
f16.add_filter('adaptive', {
    'maxValue': 255, 'adaptiveMethod': cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    'thresholdType': cv2.THRESH_BINARY, 'blockSize': 15, 'C': 2
})
f16.apply_and_draw2()


# In[ ]:


pipe = PipelineStream(imagens_grayscale, ext_imagens, metadata)


# In[ ]:


pipe.extend_filters([f1, f2, f3, f4, f5, f6, f7, f8, f9, fA, fB, fC, fD, fE, fF, f10, f11, f12, f13, f14, f15, f16])


# In[ ]:


# simplified
pipe.extend_filters([f1])


# In[143]:


hold = pipe.score_filters()


# In[144]:


hold[1:]


# In[ ]:


pipe2 = PipelineStream(imagens_grayscale, ext_imagens, metadata)
pipe2.extend_filters([f1, f3, f5, f7, f9, fB, fD, fF, f11, f13, f15])
hold2 = pipe2.score_filters()


# In[1]:


hold2[1:]


# In[ ]:



hold[0].filter_sequence


# In[ ]:


hold[1]


# In[ ]:


best_filter = hold[0]


# In[ ]:


indice = 11
img_name = ext_imagens[indice]
best_filter.image = Image(imagens_grayscale[indice], name=img_name, metadata=metadata['training'][img_name])
best_filter.apply_and_draw2()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




