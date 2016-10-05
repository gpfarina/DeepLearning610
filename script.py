import numpy as np
import glob
from sklearn.decomposition import PCA
from PIL import Image
from numpy import *
import os
import matplotlib.pyplot as plt

def loadpngs():
    trainData = []
    trainLabels = []
    
    for filename in glob.glob('/home/gpietro/PhD/examsUB/610-DL/Handwritten-and-data/Cursive/imagesMod/*.png'):
        str=os.path.basename(filename)
        tag=str.split('_')[0]
        tag=int(''.join(i for i in tag if i.isdigit()))
        img=Image.open(filename).convert('L')#black and white
        arr = np.array(img)
        trainData.append(arr)
        trainLabels.append(tag)
        img.close()
    return((np.asarray(trainData), np.asarray(trainLabels)))

def pcaRed(img):#redDim):
    mu = np.mean(img, axis=0)
    pca = PCA()
    pca.fit(img)
    #Xhat = (np.dot(pca.transform(img)[:,:redDim], pca.components_[:redDim,:]))+pca.mean_
    return(pca)#,Xhat)
    
