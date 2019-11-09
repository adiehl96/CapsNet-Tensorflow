nonfaceIdFile = './nonfaceIds'
nonfacefolder = './data/coco/train2017/'
facefolder    = './data/faceData/*'
cocoAnn       = './data/coco/annotations/instances_train2017.json'
import sys
sys.path.append('./pythonapi/PythonAPI')

import pickle
import numpy as np
import skimage as ski
import skimage.io as io
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import glob

# coco = COCO(cocoAnn)
# with open('nonfaceIds', 'rb') as fp:
#     nonfaceids = pickle.load(fp)

#imgs = [coco.loadImgs(x)[] for x in nonfaceids]

# img = coco.loadImgs(nonfaceids[:1])[0]
# I = io.imread(nonfacefolder + img['file_name'])
# I = I[0:86,0:86]
# images = [I]
# i = 0
# for imgId in nonfaceids[1:]:
#     img = coco.loadImgs([imgId])[0]
#     I = io.imread(nonfacefolder + img['file_name'])
#     I = I[0:86,0:86]
#     # print(I.shape)
#     if(I.shape == (86,86,3)):
#         images = np.concatenate((images, [I]))
#     i = i +1
#     if (i%1000 == 0):
#         print(str(i) + " from " + str(len(nonfaceids)))
#     if (i == 30000):
#         break;
# with open('outfile', 'wb') as fp:
#     pickle.dump(images, fp)

# imagelocations = glob.glob('./data/faceData/*/*/*/*')
# I = io.imread(imagelocations[0])
# I = I[0:86,0:86]
# images = [I]
# i = 0
# for imgloc in imagelocations[1:]:
#     I = io.imread(imgloc)
#     I = I[0:86, 0:86]
#     i = i+1
#     if (I.shape == (86, 86, 3)):
#         images = np.concatenate((images, [I]))
#     if (i % 1000 == 0):
#         print(str(i) + " from " + str(len(imagelocations)))

facecategories = np.zeros((27682))
nonfacecategories = np.ones((29893))
imgset = np.concatenate((facecategories, nonfacecategories))

with open('categories', 'wb') as fp:
    pickle.dump(imgset, fp)

print(imgset.shape)