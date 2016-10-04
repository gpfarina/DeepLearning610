import numpy as np
import glob
from sklearn.decomposition import PCA
from PIL import Image
from numpy import *
import PIL
import os

def loadpngs():
    trainData = []
    trainLabels = []
    maxwidth=325
    maxheight=150
    
    for filename in glob.glob('/home/gpietro/PhD/examsUB/610-DL/Handwritten-and-data/Cursive/images/*.png'):
        str=os.path.basename(filename)
        tag=str.split('_')[0]
        tag=int(''.join(i for i in tag if i.isdigit()))
        img=Image.open(filename).convert('RGBA')
        arr = np.array(img.resize((maxwidth,maxheight), PIL.Image.ANTIALIAS))
        trainData.append(arr)
        trainLabels.append(tag)
        img.close()
    return((np.asarray(trainData), np.asarray(trainLabels)))



