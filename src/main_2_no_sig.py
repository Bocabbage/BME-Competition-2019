from dataset_ori import *
from model import Net
from network import *
import argparse
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.tensor
import torch.backends.cudnn as cudnn
import cv2
from torch.autograd import Variable

from PIL import Image
import nibabel as nib

from torch.autograd import Variable
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('dataroot', help='path to dataset of kaggle ultrasound nerve segmentation')
# parser.add_argument('dataroot', default='data', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--mode', type=str, default='train',help='select mode:\'predict\'/\'train\' ')
parser.add_argument('--start_epoch', type=int, default=0, help='number of epoch to start')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.9, help='momentum1 in Adam')        # momentum1 in Adam
parser.add_argument('--beta2', type=float, default=0.999, help='momentum2 in Adam')      # momentum2 in Adam
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--useBN', action='store_true', help='enalbes batch normalization')
parser.add_argument('--output_name', default='checkpoint___.tar', type=str, help='output checkpoint filename')

args = parser.parse_args()
print(args)

def Normalize(img):
  return (img-np.min(img))/(np.max(img)-np.min(img))

############## dataset processing
#dataset = kaggle2016nerve(args.dataroot)
if args.mode == 'train':
  dataset = SpinalSagitalMRI(args.dataroot,args.cuda,train=(args.mode=='train'))
  train_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             num_workers=args.workers, shuffle=False,pin_memory=True)

############## create model
#model = Net(args.useBN)
model = thin_AttU_Net()
if args.cuda:
  model.cuda()
  cudnn.benchmark = True
optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
############## resume
if args.mode == 'train':
  if args.resume:
    if os.path.isfile(args.resume):
      print("=> loading checkpoint '{}'".format(args.resume))

      if args.cuda == False:
        checkpoint = torch.load(args.resume, map_location={'cuda:0':'cpu'})
      else:
        checkpoint = torch.load(args.resume)

      args.start_epoch = checkpoint['epoch']

      model.load_state_dict(checkpoint['state_dict'])
      #optimizer.load_state_dict(checkpoint['optim_dict'])
      print("=> loaded checkpoint (epoch {}, loss {})"
          .format(checkpoint['epoch'], checkpoint['loss']) )
    else:
      print("=> no checkpoint found at '{}'".format(args.resume))


def save_checkpoint(state, filename=args.output_name):
  torch.save(state, filename)

############ just check test (visualization)

def showImg(img, binary=True, fName=''):
  """
  show image from given numpy image
  """
  img = img[0,0,:,:]

  if binary:
    img = img > 0.5

  img = Image.fromarray(np.uint8(img*255), mode='L')

  if fName:
    img.save('asset/'+fName+'.png')
  else:
    img.show()
  
with torch.no_grad():
  model.eval()
  #train_loader.batch_size=1
  DSC = 0
  PPV = 0
  Sensitivity = 0
  loss_fn = nn.MSELoss()
  if args.cuda:
    loss_fn = loss_fn.cuda()

  loss_sum = 0
  for i, (x,y) in enumerate(train_loader):
    x, y_true = Variable(x), Variable(y)
    if args.cuda:
      x = x.cuda()
      y_true = y_true.cuda()

    y_pred = model(Variable(x))
    loss = loss_fn(y_pred,y_true)
    loss_sum += loss.item()

    gt = y.numpy()
    pre = y_pred.data.cpu().numpy()
    nonBin=pre
    pre = pre > 0.666 #0.599 for surface 0.673 for middle 0.666 for core
    #pre=(pre*255).astype('uint8')
    #pre[0,0,:,:]=cv2.Canny(pre[0,0,:,:],0,255,cv2.THRESH_OTSU|cv2.THRESH_BINARY)
    #pre[0,0,:,:]=cv2.Canny(pre[0,0,:,:],20,150)
    #pre=pre >40
    
    showImg(x.cpu().numpy(), binary=False, fName=str(i)+'_ori')
    showImg(pre, binary=False, fName=str(i)+'_pre')
    showImg(nonBin, binary=False, fName=str(i)+'_pre_nonBin')
    showImg(gt, binary=False,fName=str(i)+'_grt')
    
    
      # DSC = 2TP/(FP+2TP+FN) #
      # PPV = TP/(TP+FP) #
      # Sensitivity = TP/(TP+FN) #
    for nums in range(pre.shape[0]):
      TP = sum(sum(sum((pre[nums,:,:,:]==1) & (gt[nums,:,:,:]==1))))
      FP = sum(sum(sum((pre[nums,:,:,:]==1) & (gt[nums,:,:,:]==0))))
      FN = sum(sum(sum((pre[nums,:,:,:]==0) & (gt[nums,:,:,:]==1))))
      DSC += 2*TP / (FP+2*TP+FN)
      PPV += TP / (TP+FP)
      Sensitivity += TP / (TP+FN)
  DSC = DSC / (len(train_loader)*pre.shape[0])
  PPV = PPV / (len(train_loader)*pre.shape[0])
  Sensitivity = Sensitivity / (len(train_loader)*pre.shape[0])
  loss_sum = loss_sum/len(train_loader)
  print('loss: {}'.format(loss_sum))
  print('PPV: {}'.format(PPV))
  print('Sensitivity: {}'.format(Sensitivity))
  print('DSC: {}'.format(DSC))

############## predict

if args.mode == 'predict':

  TypeI = []
  TypeII = []
  TypeIII = []

  def Load_NII_File(filename):
    img = nib.load(filename)
    img_array = img.get_fdata()
    img_array = np.squeeze(img_array)
    return img_array  

  def Get_Image_List(filename):
    if not os.path.isfile(filename):
      print("No such File!")
    else:
      img_array = Load_NII_File(filename)
      for i in range(0, img_array.shape[2]):
        img = Normalize(np.transpose(img_array[:,:,i]))
        image = Image.fromarray(img)
        #image = image.convert('L')
        if not img.shape[0]==880:
          image = image.resize([880,880])
        image = image.crop([208,0,880-208,880])
        if img_array.shape[2]==12:
          if (i < 2) or (i > 9):
            TypeI.append(image)
          elif (i < 4) or (i > 7):
            TypeII.append(image)
          else:
            TypeIII.append(image)
        else:
          if (i < 3) or (i > 11):
            TypeI.append(image)
          elif (i < 5) or (i > 9):
            TypeII.append(image)
          else:
            TypeIII.append(image)
          

  if os.path.isdir(args.dataroot):
    # .nii(s) in file
    for filename in os.listdir(args.dataroot):
      in_p = os.path.join(args.dataroot, filename)
      if not os.path.isfile(in_p):
        continue
      Get_Image_List(in_p)
  else:
    Get_Image_List(args.dataroot)

  model.eval()

  checkpointI = 'E:\\temp\\BME2019Competition\\UniChannel.tar'
  checkpointII = 'E:\\temp\\BME2019Competition\\UniChannel.tar'
  checkpointIII = 'E:\\temp\\BME2019Competition\\UniChannel.tar'
  Checkpoint = [checkpointI,checkpointII,checkpointIII]
  #print(Checkpoint[0])
  TypeList = [TypeI,TypeII,TypeIII]

  ######### Predict #########
  for Type in range(3):
    if os.path.isfile(Checkpoint[Type]):
      print("=> loading checkpoint '{}'".format(Checkpoint[Type]))

      if args.cuda == False:
        checkpoint = torch.load(Checkpoint[Type], map_location={'cuda:0':'cpu'})
      else:
        checkpoint = torch.load(Checkpoint[Type])

      # args.start_epoch = checkpoint['epoch']
      model.load_state_dict(checkpoint['state_dict'])
      # optimizer.load_state_dict(checkpoint['optim_dict'])
      # print("=> loaded checkpoint (epoch {}, loss {})"
      #     .format(checkpoint['epoch'], checkpoint['loss']) )
    else:
      print("=> no checkpoint found at '{}'".format(Checkpoint[Type]))

    sample = 0
    for img in TypeList[Type]:
      if args.cuda:
        img = torch.from_numpy(np.array(img)).cuda()
        y_pred = model(Variable(img))
        #showImg(img.cpu().numpy(), binary=False, fName='ori_'+str(i))
        showImg(y_pred.data.cpu().numpy(), binary=False, fName='pred_'+str(Type)+'_'+str(sample))
        #showImg(y.cpu().numpy(), fName='gt_'+str(i))
      else:
        img = torch.from_numpy(np.array(img)).float().unsqueeze(0).unsqueeze(0)
        y_pred = model(Variable(img))
        #showImg(img.cpu().numpy(), binary=False, fName='ori_'+str(i))
        showImg(y_pred.data.cpu().numpy(), binary=False, fName='pred_'+str(Type)+'_'+str(sample))
        #showImg(y.cpu().numpy(), fName='gt_'+str(i))
      sample += 1


