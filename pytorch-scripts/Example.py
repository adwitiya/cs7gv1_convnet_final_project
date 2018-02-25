#Demo for CS7-GV1

#general modules
from __future__ import print_function, division
import os
import argparse
import time
import copy
import numpy as np
import sys
import resource
#pytorch modules
import torch
import torch.nn as nn
from torchvision import datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
import torch.optim as optim
from torch.autograd import Variable
import pdb

#user defined modules
import Augmentation as ag
import Models
from Test import Test
parser = argparse.ArgumentParser(description='CS7-GV1 Final Project');

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

#add/remove arguments as required. It is useful when tuning hyperparameters from bash scripts
resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)
parser.add_argument('--aug', type=str, default = '', help='data augmentation strategy')
parser.add_argument('--datapath', type=str, default='', 
               help='root folder for data.It contains two sub-directories train and val')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')               
parser.add_argument('--pretrained', action='store_true',
                    help='use pretrained model')
parser.add_argument('--batch_size', type=int, default = 128,
                    help='batch size')
parser.add_argument('--model', type=str, default = None, help='Specify model to use for training.')
parser.add_argument('--optim', type=str, default = None, help='Specify Optimizeer')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=25,
                    help='upper epoch limit')
parser.add_argument('--tag', type=str, default=None,
                    help='unique_identifier used to save results')
args = parser.parse_args();

#print (sys.getrecursionlimit())
if not args.tag:
    print('Please specify tag...')
    exit()
print (args)

#Define augmentation strategy
augmentation_strategy = ag.Augmentation(args.aug);
data_transforms = augmentation_strategy.applyTransforms();
##

#Root directory
data_dir = args.datapath;
##

######### Data Loader ###########
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
             for x in ['train', 'val']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=args.batch_size,
                                               shuffle=True, num_workers=16) # set num_workers higher for more cores and faster data loading
             for x in ['train', 'val']}
                 
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes
#################################

#set GPU flag
use_gpu = args.cuda;
##

#Load model . Once you define your own model in Models.py, you can call it from here. 
if args.model == "ResNet18":
    current_model = Models.resnet18(args.pretrained)
    num_ftrs = current_model.fc.in_features
    current_model.fc = nn.Linear(num_ftrs, len(dset_classes));
    
elif args.model == 'Demo':
    current_model = Models.demo_model();
elif args.model == 'AlexNet':
    current_model = Models.AlexNet(args.pretrained)
elif args.model == 'adwitiya':
    current_model = Models.FusionNet_def();
elif args.model == 'VGG13':
    current_model = Models.vgg(args.pretrained)
elif args.model == 'Test':
    print('loading test model')
    current_model = torch.load(args.tag+'.model')
    num_ftrs = current_model.my_fc_3.in_features
    current_model.my_fc_3 = nn.Linear(num_ftrs, 20);
    print('loaded test model')		
else :
    print ("Model %s not found"%(args.model))
    exit();    


if use_gpu:
    current_model = current_model.cuda();
    
# uses a cross entropy loss as the loss function
#  #
criterion = nn.CrossEntropyLoss()

#uses stochastic gradient descent for learning
# http://pytorch.org/docs/master/optim.html
if args.optim == "Sgd":
	print((args.optim))
	optimizer_ft = optim.SGD(current_model.parameters(), lr=args.lr, momentum=0.9)
elif args.optim == "Adam":
	print((args.optim))
	optimizer_ft = optim.Adam(current_model.parameters(), lr=args.lr)
elif args.optim == "Adamax":
	print((args.optim))
	optimizer_ft = optim.Adamax(current_model.parameters(), lr=args.lr)
else :
    print ("Optimizer %s not found"%(args.optim))
    exit();   
#the learning rate condition. The ReduceLROnPlateau class reduces the learning rate by 'factor' after 'patience' epochs.
scheduler_ft = ReduceLROnPlateau(optimizer_ft, 'min', factor = 0.5,patience = 3, verbose = True)



def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=25):
    since = time.time()
    best_model = model
    best_acc = 0.0
    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            top1 = AverageMeter()
            top5 = AverageMeter()
            # Iterate over data.
            for count, data in enumerate(dset_loaders[phase]):
                # get the inputs
                inputs, labels = data
                inputs, target = data
                target = target.cuda(async =True)

                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(target.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(target)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            
		prec1, prec5 = accuracy(outputs.data,target, topk=(1,5))
		#print(prec1[0])		
            	top1.update(prec1[0], inputs.size(0))
            	top5.update(prec5[0], inputs.size(0))
                # statistics
                
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                #if count%10 == 0:
                #    print('Batch %d || Running Loss = %0.6f || Running Accuracy = %0.6f'%(count+1,running_loss/(args.batch_size*(count+1)),running_corrects/(args.batch_size*(count+1))))
                #print('Running Loss = %0.6f'%(running_loss/(args.batch_size*(count+1))))

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]
            print('Epoch %d || %s Loss: %.4f || Acc: %.4f||Top1: %.4f|| Top5: %.4f||'%(epoch,
                phase, epoch_loss, epoch_acc,top1.avg,top5.avg),end = ' || ')
            #pdb.set_trace();
            if phase == 'val':
                print ('\n', end='');
                lr_scheduler.step(epoch_loss);
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model


#comment the block below if you are not training 
######################
trained_model = train_model(current_model, criterion, optimizer_ft, scheduler_ft,
                      num_epochs=args.epochs);
with open(args.tag+'.model', 'wb') as f:
    torch.save(trained_model, f);
######################    
## uncomment the lines blow while testing.
'''trained_model = torch.load(args.tag+'.model');
testDataPath = 'test_data'
t = Test(args.aug,trained_model);
scores = t.testfromdir(testDataPath);
#pdb.set_trace();
np.savetxt(args.tag+'.txt', scores, fmt='%0.5f',delimiter=',')'''
