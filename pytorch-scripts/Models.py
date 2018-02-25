from torchvision import models
import torch.nn as nn
import pdb
import sys
import resource

class Demo_Model(nn.Module):
    def __init__(self, nClasses = 200):
        super(Demo_Model,self).__init__();
        
        self.conv_1 = nn.Conv2d(3,32,kernel_size=5,stride=1, padding = 2)
        self.relu_1 = nn.ReLU(True);
        self.batch_norm_1 = nn.BatchNorm2d(32);
        self.pool_1 = nn.MaxPool2d(kernel_size = 2, stride =2)
        
        self.conv_2 = nn.Conv2d(32, 32,kernel_size=5,stride=1, padding = 2)
        self.relu_2 = nn.ReLU(True);
        self.batch_norm_2 = nn.BatchNorm2d(32);
        '''self.pool_2 = nn.MaxPool2d(kernel_size = 2, stride =2)

        self.conv_3 = nn.Conv2d(32,32,kernel_size=5,stride=1, padding = 2)
        self.relu_3 = nn.ReLU(True);
        self.batch_norm_3 = nn.BatchNorm2d(32);'''

        self.fc_1 = nn.Linear(32768, 1024);
        #self.fc_1 = nn.Linear(8192, 200);
        self.relu_4 = nn.ReLU(True);
        self.batch_norm_4 = nn.BatchNorm1d(1024);
        self.dropout_1 = nn.Dropout(p = 0.5);
        self.fc_2 = nn.Linear(1024, nClasses);
        
    def forward(self,x):
        #pdb.set_trace();
        y = self.conv_1(x)
        y = self.relu_1(y)
        y = self.batch_norm_1(y)
        y = self.pool_1(y)
        
        y = self.conv_2(y)
        y = self.relu_2(y)
        y = self.batch_norm_2(y)
        '''y = self.pool_2(y)
        
        y = self.conv_3(y)
        y = self.relu_3(y)
        y = self.batch_norm_3(y)'''

        y = y.view(y.size(0), -1)
        y = self.fc_1(y)
        y = self.relu_4(y)
        y = self.batch_norm_4(y)
        y = self.dropout_1(y)
        y = self.fc_2(y)
        return(y)

class FusionNet(nn.Module):
    def __init__(self, nClasses = 200):
        super(adwitiya,self).__init__();
        sys.setrecursionlimit(15000)
	#Convolutional Layers
	self.my_conv_1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)        
	self.my_conv_2 = nn.Conv2d(64, 192, kernel_size=5, padding=2) 
        self.my_conv_3 = nn.Conv2d(192, 384, kernel_size=3, padding=1) 
	self.my_conv_4 = nn.Conv2d(384, 256, kernel_size=3, padding=1) 
	self.my_conv_5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
	
	#Relu layers
	self.my_relu_1 = nn.ReLU(inplace=True)
	self.my_relu_2 = nn.ReLU(inplace=True)
	self.my_relu_3 = nn.ReLU(inplace=True)
	self.my_relu_4 = nn.ReLU(inplace=True)
	self.my_relu_5 = nn.ReLU(inplace=True)
	self.my_relu_6 = nn.ReLU(inplace=True)
	self.my_relu_7 = nn.ReLU(inplace=True)

	# Max pooling layers
	self.my_max_pool_1 = nn.MaxPool2d(kernel_size=3, stride=2)
	self.my_max_pool_2 = nn.MaxPool2d(kernel_size=3, stride=2)
	self.my_max_pool_3 = nn.MaxPool2d(kernel_size=3, stride=2)
	
	#Bacth Norm Layers
	self.my_batch_norm_1 = nn.BatchNorm2d(64)
	self.my_batch_norm_2 = nn.BatchNorm2d(192)
	self.my_batch_norm_3 = nn.BatchNorm2d(384)
	self.my_batch_norm_4 = nn.BatchNorm2d(256)
	self.my_batch_norm_5 = nn.BatchNorm2d(256)

	#One Fully connected Layer
	self.my_fc_1 = nn.Linear(256*6*6, 4096)
	self.my_fc_2 = nn.Linear(4096,4096)
	self.my_fc_3 = nn.Linear(4096,nClasses)

	#Dropout Layers
	self.my_dropout_1 = nn.Dropout()
	self.my_dropout_2 = nn.Dropout()
	

	
    def forward(self,x):

        y = self.my_conv_1(x)
        y = self.my_relu_1(y)
        y = self.my_batch_norm_1(y)
        y = self.my_max_pool_1(y)
        
	y = self.my_conv_2(y)
        y = self.my_relu_2(y)
        y = self.my_batch_norm_2(y)
        y = self.my_max_pool_2(y)

	y = self.my_conv_3(y)
        y = self.my_relu_3(y)
        y = self.my_batch_norm_3(y)

	y = self.my_conv_4(y)
        y = self.my_relu_4(y)
        y = self.my_batch_norm_4(y)

	y = self.my_conv_5(y)
        y = self.my_relu_5(y)
        y = self.my_batch_norm_5(y)
	y = self.my_max_pool_3(y)
        y = y.view(y.size(0), -1)
        y = self.my_dropout_1(y)
	y = self.my_fc_1(y)
	y = self.my_relu_6(y)
	y = self.my_dropout_2(y)
	y = self.my_fc_2(y)
	y = self.my_relu_7(y)
	y = self.my_fc_3(y)
		
        return(y)
        
def resnet18(pretrained = True):
    return models.resnet18(pretrained)

def AlexNet(pretrained = True):
    return models.alexnet(pretrained)
    
def demo_model():
    return Demo_Model();
def vgg(pretrained = True):
    return models.vgg13(pretrained)
def FusionNet_def():
    return FusionNet();

