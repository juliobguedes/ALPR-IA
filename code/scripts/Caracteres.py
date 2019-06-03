#!/usr/bin/env python
# coding: utf-8

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
from ..classes.FilterSequence import FilterSequence
from ..classes.PipelineStream import PipelineStream


# In[ ]:


# from google.colab import files, drive
# drive.mount('/content/drive')
path = u'../placas/'
metadata = json.load(open('metadata_db.json', 'r'))

# In[ ]:

arquivos = os.listdir(path)
amostra = random.sample(arquivos, 50)
placas_gray = []

for img_nome in amostra:
    sem_ext = img_nome.replace(".png", "")

    colored = cv2.imread(path+img_nome)
    gray = cv2.cvtColor(colored, cv2.COLOR_BGR2GRAY)
    placas_gray.append((gray, sem_ext))

cv2.imshow(placas_gray[0][0])
cv2.waitKey(0)