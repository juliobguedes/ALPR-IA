#!/usr/bin/env python
# coding: utf-8

# # Amostra de Placas
# 
# Cria um conjunto de imagens apenas com as placas dos veículos

# ## Imports

# In[ ]:


import cv2
# import tensorflow as tf
# from tensorflow import keras

import matplotlib as mpl
from matplotlib import pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd

import os, json, itertools, random

from ..classes.Image import Image


# In[2]:


# from google.colab import files, drive
# drive.mount('/content/drive')
filepath = u'../placas/'
outputpath = u'/content/drive/My Drive/IA/Projeto/sample_plates/'
cropadas = '../cropadas/'


# ## File Import

# In[3]:


metadata = json.load(open('metadata_db.json', 'r'))


# ## Classes

# In[ ]:


# ## Script

# In[ ]:


arq_imagens = [png for png in os.listdir(filepath) if png.endswith(".png")]
ext_imagens = [nn.replace(".png", "") for nn in arq_imagens]

for aimg, exti in zip(arq_imagens, ext_imagens):
  img = cv2.imread(filepath + aimg)
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  rimg = Image(gray, exti, metadata['training'][exti])
  print(exti)
  rimg.crop_plate(path = cropadas, save = True)


# In[ ]:




