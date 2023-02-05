import itertools
import argparse
import math
import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F 
import pickle
import PIL
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, UNet2DConditionModel, LMSDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from huggingface_hub import notebook_login
import requests
import glob
from io import BytesIO

from random import sample
import scipy
import torchvision
import pdb
from torch.utils.data import DataLoader 
### Own modules ###
from utils.utils           import save_image, create_random_ascii, load_config_files,freeze_params,manage_cuda,my_collate
from utils.inference_utils import InferenceStableDifussion_ClIPEncoder


device = manage_cuda(2) # Set CUDA

imagenet_templates_small = [
    "a photo of a",
    "a rendering of a",
    "a cropped photo of the",
    "the photo of a",
    "a photo of a severe",
    "a close-up photo of a",
    "a bright photo of the",
    "a cropped photo of a",
    "a photo of",
    "a close-up photo of the",
    "a rendition of the",
    "a photo of the weird"
]

### Parse Arguments ###
parser = argparse.ArgumentParser(description="Performs diffusion steps for StableDifussion Pipe (either finetunned or normal pipe)")
parser.add_argument('--prompt',            help = "prompt for image conversion", type=str,    default =  "Dermatological skin disease" )
parser.add_argument('--b_start',           help = "beta start for scheduler",    type=float,  default = 0.00085)
parser.add_argument('--b_end',             help = "beta end   for scheduler",    type=float,  default = 0.012)
parser.add_argument('--guidance_scale',    help = "guidence scale of inference", type=float,  default = 7.5)
parser.add_argument('--T'    ,             help = "Difussion Steps",             type=int,    default = 20)
parser.add_argument('--height',            help = "Image Height",                type=int,    default = 512)
parser.add_argument('--width',             help = "Image Width",                 type=int,    default = 512)
parser.add_argument('--verbose',           help = "Verbose",                     type=bool,   default = False)
parser.add_argument('--epochs',            help = "epochs use for training",     type=int,    default = 1)
parser.add_argument('--loading_dir',       help = "Directory with saved weights",type=str,    default = "Pipelines/original_encoder_d_epoch_20_pqk")
  

### Load Paramaeters from Config files ####
PathToConfigFile     = os.path.join("config","config.json")
hyperparameters_path = os.path.join("config","hyperparameters.json")
token_path           = os.path.join("config","token.json")
hyperparameters, TOKEN, MODEL,REVISION,patch_data_path ,csv_path ,transforms,max_patches_total, quick  = load_config_files(PathToConfigFile ,hyperparameters_path, token_path)
############################################

def main():
    #### Parse Options ####
    arguments                   =  parser.parse_args()
    prompt                      =  arguments.prompt          # prompt  for text to image generation 
    b_start                     =  arguments.b_start      
    b_end                       =  arguments.b_end
    T                           =  arguments.T
    height                      =  arguments.height
    width                       =  arguments.width
    guidance_scale              =  arguments.guidance_scale
    verbose                     =  arguments.verbose
    number_of_epochs            =  arguments.epochs          # Number of epochs to train      
    loading_dir                 =  arguments.loading_dir  
    train_batch_size            = hyperparameters["train_batch_size"]
    gradient_accumulation_steps = hyperparameters["gradient_accumulation_steps"]
    learning_rate               = hyperparameters["learning_rate"]
    max_train_steps             = hyperparameters["max_train_steps"]
    output_dir                  = hyperparameters["output_dir"]

    ########################################################  TEXT ENCODER ##################################################################################
    tokenizer     = CLIPTokenizer.from_pretrained(loading_dir,  subfolder="tokenizer",use_auth_token=TOKEN)                           # Load tokenizer for CLIP model
    text_encoder  = CLIPTextModel.from_pretrained(loading_dir,  subfolder="text_encoder", use_auth_token=TOKEN)                       # Load Text Encoder 
    vae           = AutoencoderKL.from_pretrained(loading_dir,  subfolder="vae"         , use_auth_token=TOKEN,fp16=True)             # Image's Lattent Encoder - Decoder  (encoder used for training, Decoder for inferance)
    unet          = UNet2DConditionModel.from_pretrained(MODEL , subfolder="unet" , use_auth_token=TOKEN,fp16=True)                   # Unet (used to predict noise at timestep t)#
    scheduler     = LMSDiscreteScheduler(beta_start=b_start, beta_end=b_end,beta_schedule='scaled_linear', num_train_timesteps=1000)  # Scheduler 
    unet          = unet.to(device)                    # Move to GPU
    vae           = vae.to(device)                     # Move to GPU
    text_encoder  = text_encoder.to(device)            # Move to GPU
    diffuser_clip = InferenceStableDifussion_ClIPEncoder(vae,unet,text_encoder,scheduler,tokenizer,device=device)
    vae.eval(), unet.eval(), text_encoder.eval()       # Set Models For Evaluation

    
    path_to_directory_s =  os.path.join(loading_dir,"SYNTHETIC") # Create path for synthetic images
    print("Models Instanciated")
    try:
        os.mkdir(path_to_directory_s)
        print(f"Path Instanciated: {path_to_directory_s}")
    except:
        print("Model has been previously used for inferance")
    
    print(f"H:{device} height: {height}")
    for epoch in range(number_of_epochs):                                           # For number of epochs 
        image         = diffuser_clip.prompt_to_img(sample(imagenet_templates_small,1)[0] +" " + prompt, height, width , T)[0]   # Prompt to image 
        #pdb.set_trace()
        save_image(image,path_to_directory_s,prompt="monkeyPox")                         # Save IMAGE 
    

        

    





if __name__ == "__main__":
    main()
