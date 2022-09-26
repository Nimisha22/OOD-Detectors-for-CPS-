# 
from asyncio.unix_events import BaseChildWatcher
from code import interact
from email.mime import image
from multiprocessing.connection import wait
from time import process_time_ns
import torch
import numpy 
import torchvision
import cv2
import PIL
from PIL import Image
import os
from bvae import BetaVae
import xlsxwriter

images = []
ce = []
kl = []

def ce_loss(image1, frame, out1, batch):
    print(image1)
    ce_loss = torch.nn.functional.binary_cross_entropy(frame,out1,reduction='sum')
    print("CE Loss = ")
    ce.append(ce_loss)
    print(ce_loss)


def kl_loss(image1,mu, logvar):
    kl_loss = torch.mul(
                    input=torch.sum(mu.pow(2) + logvar.exp() - logvar - 1),
                    other=0.5)
    print(f"KL Loss for {image1}")
    kl.append(kl_loss)
    print(kl_loss)


def excel():
    workbook = xlsxwriter.Workbook('losses_1.xlsx')
    worksheet = workbook.add_worksheet()

    ce_round = []
    for num in ce:
        ce_round.append(int(num))

    #kl_round = []
    #for num in ce:
    #    kl_round.append(int(num))
    
    my_dict = {'Image': images,
           'CE_Loss': ce_round,
           'KL_Loss': kl,
           }

    
    col_num = 0
    for key, value in my_dict.items():
        worksheet.write(0, col_num, key)
        worksheet.write_column(1, col_num, value)
        col_num += 1

    workbook.close()


def inference(image1, frame: torch.Tensor) -> numpy.ndarray:
        """Make an inference on the encoder.
        
        Args:
            frame - raw input data to encoder
            
        Returns:
            A (1 x n_latent) vector of the frame's KL-divergence for each
            latent dimension.
        """
        with torch.no_grad():
            encoder = torch.load('/home/n2202864a/Downloads/code/bvae_n30_b1.0__128x128.pt')
            encoder.batch =1
            batch=1
            #print(encoder.batch)
            out, mu, logvar = encoder(frame.unsqueeze(0).to('cuda'))
            out = out.detach().cpu()
           
            #0.5 * (numpy.power(mu, 2) + numpy.exp(logvar) - logvar - 1)
            out=out*255
            #print(out.shape)
            out=out.squeeze()

            out1=out
            #print(out.shape)
            out=out.cpu().detach().numpy()
            #print(out.shape)
            out=out.transpose(1, 2, 0)
            #print(out.shape)
            out=cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            #cv2.imwrite('output.png',out)
            ce_loss(image1, frame, out1, batch)
            kl_loss(image1, mu, logvar)


def preprocess(image1, frame, image_array: numpy.ndarray) -> torch.Tensor:
        """Perform preprocing on an input frame.  This includes color space
        conversion and resize operations.
        
        Args:
            frame - input image taken from video.
        
        Returns:
            The preprocessed video frame.
        """
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torchvision.transforms.functional.to_tensor(frame)
        frame = torchvision.transforms.functional.resize(frame,(128,128), Image.BILINEAR)
        '''if n_chan == 1:
            frame = torchvision.transforms.functional.rgb_to_grayscale(frame)'''
        inference(image1, frame)           
            

def input():
    for pn in ['ood', 'No_Fog', 'Light_Fog', 'Heavy_Fog']:
        path=f"/home/n2202864a/Downloads/test1/{pn}"
        for image1 in os.listdir(path):
            images.append(image1)
            print(images)
            frame = cv2.imread(f"/home/n2202864a/Downloads/test1/{pn}/{image1}")
            #image = PIL. Image. open(image)
            image_array = numpy.array(frame)
            #print(shape(frame))
            #cv2.imshow('src',frame)
            #cv2.waitKey(0)
            preprocess(image1, frame ,image_array)   
            excel()
   

if __name__ == '__main__':
    input()
    
    

from asyncio.unix_events import BaseChildWatcher
from code import interact
from email.mime import image
from multiprocessing.connection import wait
from time import process_time_ns
import torch
import numpy 
import torchvision
import cv2
import PIL
from PIL import Image
import os
from bvae import BetaVae
import xlsxwriter

images = []
ce = []
kl = []

def ce_loss(image1, frame, out1, batch):
    print(image1)
    ce_loss = torch.nn.functional.binary_cross_entropy(frame,out1,reduction='sum')
    print("CE Loss = ")
    ce.append(ce_loss)
    print(ce_loss)


def kl_loss(image1,mu, logvar):
    kl_loss = torch.mul(
                    input=torch.sum(mu.pow(2) + logvar.exp() - logvar - 1),
                    other=0.5)
    print(f"KL Loss for {image1}")
    kl.append(kl_loss)
    print(kl_loss)


def excel():
    workbook = xlsxwriter.Workbook('losses_1.xlsx')
    worksheet = workbook.add_worksheet()

    ce_round = []
    for num in ce:
        ce_round.append(int(num))

    #kl_round = []
    #for num in ce:
    #    kl_round.append(int(num))
    
    my_dict = {'Image': images,
           'CE_Loss': ce_round,
           'KL_Loss': kl,
           }

    
    col_num = 0
    for key, value in my_dict.items():
        worksheet.write(0, col_num, key)
        worksheet.write_column(1, col_num, value)
        col_num += 1

    workbook.close()


def inference(image1, frame: torch.Tensor) -> numpy.ndarray:
        """Make an inference on the encoder.
        
        Args:
            frame - raw input data to encoder
            
        Returns:
            A (1 x n_latent) vector of the frame's KL-divergence for each
            latent dimension.
        """
        with torch.no_grad():
            encoder = torch.load('/home/n2202864a/Downloads/code/bvae_n30_b1.0__128x128.pt')
            encoder.batch =1
            batch=1
            #print(encoder.batch)
            out, mu, logvar = encoder(frame.unsqueeze(0).to('cuda'))
            out = out.detach().cpu()
           
            #0.5 * (numpy.power(mu, 2) + numpy.exp(logvar) - logvar - 1)
            out=out*255
            #print(out.shape)
            out=out.squeeze()

            out1=out
            #print(out.shape)
            out=out.cpu().detach().numpy()
            #print(out.shape)
            out=out.transpose(1, 2, 0)
            #print(out.shape)
            out=cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            #cv2.imwrite('output.png',out)
            ce_loss(image1, frame, out1, batch)
            kl_loss(image1, mu, logvar)


def preprocess(image1, frame, image_array: numpy.ndarray) -> torch.Tensor:
        """Perform preprocing on an input frame.  This includes color space
        conversion and resize operations.
        
        Args:
            frame - input image taken from video.
        
        Returns:
            The preprocessed video frame.
        """
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torchvision.transforms.functional.to_tensor(frame)
        frame = torchvision.transforms.functional.resize(frame,(128,128), Image.BILINEAR)
        '''if n_chan == 1:
            frame = torchvision.transforms.functional.rgb_to_grayscale(frame)'''
        inference(image1, frame)           
            

def input():
    for pn in ['ood', 'No_Fog', 'Light_Fog', 'Heavy_Fog']:
        path=f"/home/n2202864a/Downloads/test1/{pn}"
        for image1 in os.listdir(path):
            images.append(image1)
            print(images)
            frame = cv2.imread(f"/home/n2202864a/Downloads/test1/{pn}/{image1}")
            #image = PIL. Image. open(image)
            image_array = numpy.array(frame)
            #print(shape(frame))
            #cv2.imshow('src',frame)
            #cv2.waitKey(0)
            preprocess(image1, frame ,image_array)   
            excel()
   

if __name__ == '__main__':
    input()
    
    
