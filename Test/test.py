import torch
import numpy 
import torchvision
import cv2
from PIL import Image



def inference(frame: torch.Tensor) -> numpy.ndarray:
        """Make an inference on the encoder.
        
        Args:
            frame - raw input data to encoder
            
        Returns:
            A (1 x n_latent) vector of the frame's KL-divergence for each
            latent dimension.
        """
        with torch.no_grad():
            encoder = torch.load('/home/n2202864a/Downloads/code/bvae_n30_b2.0__128x128.pt')
            encoder.batch = 1
            print(encoder.batch)
            out, mu, logvar = encoder(frame.unsqueeze(0).to('cuda'))
            mu = mu.detach(numpy()).cpu()
            logvar = logvar.detach().cpu().numpy()
            out = out * 255
            print(out.shape)
            out = out.squeeze()
            print(out.shape)
            out = out.cpu().detach().numpy()
            print(out.shape)
            out = out.transpose(1, 2, 0)
            print(out.shape)
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            
            output = cv2.imshow('dst', out)
            cv2.imwrite('output.png',out)
            cv2.waitKey(0)


def preprocess(frame, image_array: numpy.ndarray) -> torch.Tensor:
        """Perform preprocessing on an input frame.  
        This includes color space conversion and resize operations.
        
        Args:
            frame - input image taken from video.
        
        Returns:
            The preprocessed video frame.
        """
       
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torchvision.transforms.functional.to_tensor(frame)
        frame = torchvision.transforms.functional.resize(frame,(128,128), Image.BILINEAR)
        inference(frame)
            

def input():
    path="/home/n2202864a/Downloads/test1/original.png"
    frame = cv2.imread(path)
    print(frame)
    image_array = numpy.array(frame)

    cv2.imshow('src',frame)
    cv2.waitKey(0)
    preprocess(frame, image_array)


if __name__ == '__main__':
    input()
    
    
