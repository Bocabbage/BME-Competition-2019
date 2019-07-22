import torch.utils.data as data
import torch

from scipy.ndimage import imread
import os
import os.path
import glob
import cv2
import numpy as np
from PIL import ImageEnhance
from PIL import Image

from torchvision import transforms

def make_dataset(root, train=True):

  dataset = []

  if train:
    dir = os.path.join(root, 'train')

    for fGT in glob.glob(os.path.join(dir, '*_mask.tif')):
      fName = os.path.basename(fGT)
      fImg = fName[:-9] + '.tif'

      dataset.append( [os.path.join(dir, fImg), os.path.join(dir, fName)] )

  return dataset

# class kaggle2016nerve(data.Dataset):
#   """
#   Read dataset of kaggle ultrasound nerve segmentation dataset
#   https://www.kaggle.com/c/ultrasound-nerve-segmentation
#   """

#   def __init__(self, root,transform=None, train=True):
#     self.train = train

#     # we cropped the image
#     self.nRow = 400
#     self.nCol = 560

#     if self.train:
#       self.train_set_path = make_dataset(root, train)

#   def __getitem__(self, idx):
#     if self.train:
#       img_path, gt_path = self.train_set_path[idx]

#       img = imread(img_path)
#       img = img[0:self.nRow, 0:self.nCol]
#       img = np.atleast_3d(img).transpose(2, 0, 1).astype(np.float32)
#       img = (img - img.min()) / (img.max() - img.min())
#       img = torch.from_numpy(img).float()

#       gt = imread(gt_path)[0:self.nRow, 0:self.nCol]
#       gt = np.atleast_3d(gt).transpose(2, 0, 1)
#       gt = gt / 255.0
#       gt = torch.from_numpy(gt).float()

#       return img, gt

#   def __len__(self):
#     if self.train:
#       return 5635
#     else:
#       return 5508


def make_datadir(root, train=True):

  dataset = []

  if train:
    dir_case = os.path.join(root, 'case')
    dir_mask = os.path.join(root, 'mask')

    for fGT in os.listdir(dir_mask):
      #fName = os.path.basename(fGT) # *_mask.bmp
      fImg = fGT[:-9] + '.tif'

      dataset.append( [os.path.join(dir_case, fImg), os.path.join(dir_mask, fGT)] )

  return dataset

from cv2 import getRotationMatrix2D,warpAffine
def rotate(image,angle,center = None,scale=1.0):
    (h,w) = image.shape

    if center is None:
        center = (w/2,h/2)
    M = getRotationMatrix2D(center, angle, scale)
    rotated = warpAffine(image, M,(w,h))
    return rotated

class SpinalSagitalMRI(data.Dataset):

  def __init__(self, root, cuda=False,transform=None, train=True):
    self.train = train
    self.cuda=cuda
    # self.nRow = 500
    # self.nCol = 300

    
    self.train_set_path = make_datadir(root, train)

  def __getitem__(self, idx):
    if self.train:
      img_path, gt_path = self.train_set_path[idx%len(self.train_set_path)]
      # img is an ndarray
      img = imread(img_path)
      '''
      img = img*255
      img= Image.fromarray(img).convert('L')
      enhancer = ImageEnhance.Sharpness(img)
      img= enhancer.enhance(3.0)
      img=np.array(img)
      img =img.astype('float32')
      
      img = img /255.0
      '''
      img = img[:, 2:466]
      #img = img + np.random.normal(0, 0.03, img.shape)   
      gt = imread(gt_path)
      gt = gt[:, 2:466]
      
      if len(self.train_set_path) <= idx < 2*len(self.train_set_path):
        # img = np.flip(img,1)
        # gt = np.flip(gt,1)
        img = rotate(img, angle=-5)
        gt = rotate(gt, angle=-5)     
        #img = img + np.random.normal(0, 0.05, img.shape)   
      elif 2*len(self.train_set_path)<= idx < 3*len(self.train_set_path):
        img = rotate(img, angle=5)
        #img = img + np.random.normal(0, 0.05, img.shape)
        gt = rotate(gt, angle=5)

      #img=(img*255).astype('uint8')
      #img=cv2.equalizeHist(img)
      #img= img.astype('float32')
      #img = (img - img.min()) / (img.max() - img.min())
      img = np.atleast_3d(img).transpose(2, 0, 1).astype(np.float32)
      #img = img + np.random.normal(0, 0.05, img.shape)
      #img = (img - img.min()) / (img.max() - img.min())
      #img = np.expand_dims(img,0)
      img = torch.from_numpy(img).float()

      gt = np.atleast_3d(gt).transpose(2, 0, 1)
      #gt = np.expand_dims(gt,0)
      gt = gt / 255.0
      gt = torch.from_numpy(gt).float()
    if self.cuda:
      return img,gt
    else:
      return img, gt

  def __len__(self):
    return len(self.train_set_path)
    #return 3

class SpinalSagitalMRIeval(data.Dataset):

  def __init__(self, root, cuda=False,transform=None, train=True):
    self.train = train
    self.cuda=cuda
    # self.nRow = 500
    # self.nCol = 300

    
    self.train_set_path = make_datadir(root, train)

  def __getitem__(self, idx):
    if self.train:
      img_path, gt_path = self.train_set_path[idx%len(self.train_set_path)]
      # img is an ndarray
      img = imread(img_path)
      '''
      img = img*255
      img= Image.fromarray(img).convert('L')
      enhancer = ImageEnhance.Sharpness(img)
      img= enhancer.enhance(3.0)
      img=np.array(img)
      img =img.astype('float32')
      
      img = img /255.0
      '''
      img = img[:, 2:466]
      #img = img + np.random.normal(0, 0.03, img.shape)   
      gt = imread(gt_path)
      gt = gt[:, 2:466]
      
      if len(self.train_set_path) <= idx < 2*len(self.train_set_path):
        # img = np.flip(img,1)
        # gt = np.flip(gt,1)
        img = rotate(img, angle=-5)
        gt = rotate(gt, angle=-5)     
        #img = img + np.random.normal(0, 0.05, img.shape)   
      elif 2*len(self.train_set_path)<= idx < 3*len(self.train_set_path):
        img = rotate(img, angle=5)
        #img = img + np.random.normal(0, 0.05, img.shape)
        gt = rotate(gt, angle=5)

      #img=(img*255).astype('uint8')
      #img=cv2.equalizeHist(img)
      #img= img.astype('float32')
      #img = (img - img.min()) / (img.max() - img.min())
      img = np.atleast_3d(img).transpose(2, 0, 1).astype(np.float32)
      #img = img + np.random.normal(0, 0.05, img.shape)
      #img = (img - img.min()) / (img.max() - img.min())
      #img = np.expand_dims(img,0)
      img = torch.from_numpy(img).float()

      gt = np.atleast_3d(gt).transpose(2, 0, 1)
      #gt = np.expand_dims(gt,0)
      gt = gt / 255.0
      gt = torch.from_numpy(gt).float()
    if self.cuda:
      return img,gt
    else:
      return img, gt

  def __len__(self):
    return len(self.train_set_path)
    #return 3
