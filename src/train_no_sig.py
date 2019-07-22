from dataset import *
from model import Net
from network import *
import argparse
import torch.optim as optim
import torch as F
import torch.nn as nn
import torch.tensor
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

from torch.autograd import Variable
import shutil

import csv
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('dataroot', help='path to dataset of kaggle ultrasound nerve segmentation')
# parser.add_argument('dataroot', default='data', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
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

############## dataset processing
#dataset = kaggle2016nerve(args.dataroot)
dataset = SpinalSagitalMRI(args.dataroot,args.cuda)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                           num_workers=args.workers, shuffle=True,pin_memory=True)
evalset = SpinalSagitalMRIeval('../Desktop/test/',args.cuda)
eval_loader = torch.utils.data.DataLoader(evalset, batch_size=1,
                                           num_workers=args.workers, shuffle=False,pin_memory=True)
############## create model
#model = Net(args.useBN)
model = thin_AttU_Net()
if args.cuda:
  model.cuda()
  cudnn.benchmark = True
#optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
optimizer = optim.Adam(model.parameters(),lr = args.lr, betas = [args.beta1, args.beta2])
#optimizer = optim.RMSprop(model.parameters())
############## resume
if args.resume:
  if os.path.isfile(args.resume):
    print("=> loading checkpoint '{}'".format(args.resume))

    if args.cuda == False:
      checkpoint = torch.load(args.resume, map_location={'cuda:0':'cpu'})
    else:
      checkpoint = torch.load(args.resume)

    args.start_epoch = checkpoint['epoch']

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optim_dict'])
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
    img.save('assets/'+fName+'.png')
  else:
    img.show()



############## training

model.train()
epoch = args.start_epoch
global best_loss
best_loss = 1000000000
global best_epoch
best_epoch = 0

def train(in_epoch):
  """
  training
  """
  global best_loss 
  global best_epoch 
  epoch=in_epoch
  loss_fn = nn.MSELoss()
  if args.cuda:
    loss_fn = loss_fn.cuda()

  loss_sum = 0
  model.train()
  for i, (x, y) in enumerate(train_loader):
    x, y_true = Variable(x), Variable(y)
    if args.cuda:
      x = x.cuda()
      y_true = y_true.cuda()

    for ii in range(1):
      y_pred = model(x)
      loss = loss_fn(y_pred,y_true)
      optimizer.zero_grad()
      loss.backward()
      loss_sum += loss.item()

      optimizer.step()
 
    if i % 10 == 0:
      print('batch no.: {}, loss: {}'.format(i, loss.item()))

  loss_sum = loss_sum/len(train_loader)
  print('this epoch: {}, epoch loss: {}'.format(epoch,loss_sum ))
  train_loss_sum = loss_sum
  showImg(x.cpu().numpy(), binary=False, fName='ori_'+str(i))
  showImg(y_pred.data.cpu().numpy(), binary=False, fName='pred_'+str(i))
  showImg(y.cpu().numpy(), fName='gt_'+str(i))

  save_checkpoint({
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'loss':loss_sum,
    'optim_dict': optimizer.state_dict()
  })
  
  with torch.no_grad():
    model.eval()
    loss_sum = 0
    for i, (x,y) in enumerate(eval_loader):
      x, y_true = Variable(x), Variable(y)
      if args.cuda:
        x = x.cuda()
        y_true = y_true.cuda()

      y_pred = model(Variable(x))
      loss = loss_fn(y_pred,y_true)
      loss_sum += loss.item()
      
  loss_sum = loss_sum/len(eval_loader)
  print('##this epoch: {}, eval_loss: {}'.format(epoch,loss_sum ))
  if loss_sum < best_loss: 
    best_loss = loss_sum
    best_epoch = epoch
    save_checkpoint({
      'epoch': epoch,
      'state_dict': model.state_dict(),
      'loss':loss_sum,
      'optim_dict': optimizer.state_dict()
    } , filename= args.output_name + '.best' )
  
  with open('./loss_log_'+args.output_name+'.csv','a') as f:
    f_csv = csv.writer(f)
    #f_csv.writerows(header)
    f_csv.writerow([epoch,train_loss_sum,loss_sum]) 
  print('!!Best epoch: {}, best eval_loss: {}'.format(best_epoch,best_loss ))
  epoch+=1
  return epoch,(epoch-1,train_loss_sum,loss_sum)


log_list=[]
#header=['epoch','train-loss','eval-loss']
for xx in range(args.niter):
  epoch,log=train(epoch)
  log_list.append(log)




fig = plt.figure(1)
plt.plot([Epoch[0] for Epoch in log_list],[Epoch[1] for Epoch in log_list] ,'r')
plt.plot([Epoch[0] for Epoch in log_list],[Epoch[2] for Epoch in log_list] ,'g')
plt.title("Epoch-Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(['train','eval'])
plt.show()




'''
model.eval()
#train_loader.batch_size=1

for i, (x,y) in enumerate(train_loader):
  if i >= 11:
    break

  if args.cuda:
    x = x.cuda()
    y_pred = model(Variable(x))
    showImg(x.cpu().numpy(), binary=False, fName='ori_'+str(i))
    showImg(y_pred.data.cpu().numpy(), binary=False, fName='pred_'+str(i))
    showImg(y.cpu().numpy(), fName='gt_'+str(i))
'''
