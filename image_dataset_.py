import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10 

from PIL import Image

def to_RGB(image:Image)->Image:
  if image.mode == 'RGB':return image
  image.load() # required for png.split()
  background = Image.new("RGB", image.size, (255, 255, 255))
  background.paste(image, mask=image.split()[3]) # 3 is the alpha channel
  
  file_name = 'tmp.jpg'
  background.save(file_name, 'JPEG', quality=80)
  return cv2.open(file_name)
  #return Image.open(file_name)


class MyAddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
class MyConvertRGB2YCrCb(object):
    def __init__(self, R=0, G=0, B= 0):
        self.R = R
        self.G = G
        self.B = B
        
    def __call__(self, tensor):
        Y = 0.299 * self.R+ 0.587 * self.G+ 0.114 * self.B
        Cb = -0.168736 * self.R - 0.331264 * self.G + 0.5 * self.B
        Cr = 0.5 * self.R - 0.418688 * self.G - 0.081312 * self.B    
        return Y, Cb, Cr
    
    def __repr__(self):
        return self.__class__.__name__ + '(R={0}, G={1}, B = {2})'.format(self.R, self.G, self.B) 

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, data_num,train_=True, transform1 = None, transform2 = None,train = True):
                
        self.transform1 = transform1
        self.transform2 = transform2
        self.ts = torchvision.transforms.ToPILImage()
        self.ts2 = transform=transforms.ToTensor()
        self.train = train_
        
        self.data_dir = './'
        self.data_num = data_num
        self.data = []
        self.label = []

        # download
        CIFAR10(self.data_dir, train=True, download=True)
        #CIFAR10(self.data_dir, train=False, download=True)
        self.data =CIFAR10(self.data_dir, train=self.train, transform=self.ts2)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx][0]
        out_label_ =  self.data[idx][1]
        out_label = torch.from_numpy(np.array(out_label_)).long()
        
        if self.transform1:
            out_data1 = self.transform1(out_data)
        if self.transform2:
            out_data2 = self.transform2(out_data)
        return out_data, out_data1, out_data2, out_label

    
ts = torchvision.transforms.ToPILImage()    
dims = (32, 32) 
mean, std =[0.5,0.5,0.5], [0.25,0.25,0.25]
trans2 = torchvision.transforms.Compose([
    #torchvision.transforms.Normalize(mean, std),
    #torchvision.transforms.Resize(dims)
])
trans1 =  torchvision.transforms.Compose([
    torchvision.transforms.Normalize(mean, std),
    MyAddGaussianNoise(0., 0.5),
    #torchvision.transforms.Resize(dims),
    torchvision.transforms.Grayscale()
])

dataset = ImageDataset(32, transform1=trans1, transform2=trans2)
testloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=0)
for out_data, out_data1, out_data2,out_label in testloader:
    for i in range(len(out_label)):
        image =  out_data[i]
        image1 = out_data1[i]
        image_ = to_RGB(ts(image)) #jpeg
        orgYCrCb = cv2.cvtColor(np.float32(image_), cv2.COLOR_BGR2YCR_CB)
        Y, Cr,Cb = cv2.split(orgYCrCb)
        print(out_label[i])
        plt.imshow(Y, cmap = "gray")
        plt.title('Y')
        print(type(orgYCrCb))
        plt.pause(1)
        plt.imshow(Cr)
        plt.title('Cr')
        plt.pause(1)
        plt.imshow(Cb)
        plt.title('Cb')
        plt.pause(1)
        #img_bgr = cv2.cvtColor(orgYCrCb, cv2.COLOR_YCrCb2GBR)
        #cv2.imshow('Y+Cr+Cb',Y+Cr+Cb)
        CC = cv2.merge((np.array(Y/255.),np.array(Cr/255.),np.array(Cb/255.)))
        #YCC = cv2.merge(Y,CC)
        plt.imshow(CC)
        plt.title('Y+Cr+Cb')
        plt.pause(1)
        
        plt.close()        