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
from utils.utils            import save_image, create_random_ascii, load_config_files,freeze_params,manage_cuda,my_collate

import  sys 
sys.path.append('..')
from data import dataset 

os.environ['TRANSFORMERS_CACHE'] = '/labs/gevaertlab/data/bmed/code/Alejandro/cache' ## Set Where to laod downloaded models





### Manage Cuda ###
device = manage_cuda()
device = "cuda:2"
### Parse Arguments ###
parser = argparse.ArgumentParser(description="Performs diffusion steps for StableDifussion Pipe (either finetunned or normal pipe)")
parser.add_argument('--prompt',            help = "prompt for image conversion", type=str,    default = "H&E Stained Image TCGA-KIRP" )
parser.add_argument('--b_start',           help = "beta start for scheduler",    type=float,  default = 0.00085)
parser.add_argument('--b_end',             help = "beta end   for scheduler",    type=float,  default = 0.012)
parser.add_argument('--guidance_scale',    help = "guidence scale of inference", type=float,  default = 7.5)
parser.add_argument('--T'    ,             help = "Difussion Steps",             type=int,    default = 20)
parser.add_argument('--height',            help = "Image Height",                type=int,    default = 512)
parser.add_argument('--width',             help = "Image Width",                 type=int,    default = 512)
parser.add_argument('--verbose',           help = "Verbose",                     type=bool,   default = True)
parser.add_argument('--quick',             help = "quick option",                type=bool,   default = True)
parser.add_argument('--max_patches_total', help = "patches per image",           type=int,    default = 10)
parser.add_argument('--only_tune_encoder', help = "Wether to only tune encoder", type=bool,   default = True)
parser.add_argument('--epochs',            help = "epochs use for training",     type=int,    default = 3)
parser.add_argument('--loading_dir',       help = "Directory to load  weights",  type=str,    default = "None")
parser.add_argument('--difusse_on_end',    help = "Perform prediction on end",   type=bool,   default = False)

#"Pipelines/original_encoder_d_epoch_0_RmL"

### Load Paramaeters from Config files ####
PathToConfigFile     = os.path.join("config","config.json")
hyperparameters_path = os.path.join("config","hyperparameters.json")
token_path           = os.path.join("config","token.json")
hyperparameters, TOKEN, MODEL,REVISION,patch_data_path ,csv_path ,transforms,max_patches_total, quick  = load_config_files(PathToConfigFile ,hyperparameters_path, token_path)
############################################

def main():
    ################# Parse Options ##################
    arguments         =  parser.parse_args()         # Get arguments 
    prompt            =  arguments.prompt            # prompt = "H&E Stained Image <vector> \in 200" 
    b_start           =  arguments.b_start           # Beta start
    b_end             =  arguments.b_end             # Beta end
    T                 =  arguments.T                 # Diffusion Steps
    height            =  arguments.height            # Height of image (512)
    width             =  arguments.width             # Width  of image (512)
    guidance_scale    =  arguments.guidance_scale    # Guidance scale for diffusion
    verbose           =  arguments.verbose           # Argument to be excplit 
    max_patches_total =  arguments.max_patches_total # maximum number of patches to use
    quick             =  arguments.quick             # if True, use only 10 patches
    only_tune_encoder =  arguments.only_tune_encoder # fine tune encoder   
    number_of_epochs  =  arguments.epochs            # Number of epochs to train   
    loading_dir       =  arguments.loading_dir       # log dir to continue trainning   
    difusse_on_end    =  arguments.difusse_on_end    # run prediction at the end   
    save_every_n_epocns         = 20                 # epoxhs to save 
    loss_list                   = []
    train_batch_size            = hyperparameters["train_batch_size"]
    gradient_accumulation_steps = hyperparameters["gradient_accumulation_steps"]
    learning_rate               = hyperparameters["learning_rate"]
    max_train_steps             = hyperparameters["max_train_steps"]
    output_dir                  = hyperparameters["output_dir"]


    ################ Instaciatiate Dataset ########################
    if verbose:
        print("Instaciating dataset")
        
    augmentation       = [torchvision.transforms.ToTensor(),torchvision.transforms.Resize(( height, width)),torchvision.transforms.RandomRotation((-180,180))]
    transforms         = torchvision.transforms.Compose(augmentation)
    pixel_test_dataset = dataset.PixelTextDataset(image_size=512 ,font ="arial.ttf")
    #pdb.set_trace()
    

    ########################################################  TEXT ENCODER ##################################################################################
    placeholder_token     = "text" #  "Monkey Pox Images" # {type:"string"}  We use angle brackets to differentiate a token from other words/tokens,to avoid collision.
    initializer_token     = "text" # "Dermatology"      #"Skin"     
    

    if loading_dir == "None":
        print("Fine Tunning Model from sracth")
        text_encoder     = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder", use_auth_token=TOKEN) # instanciate text encoder 
        tokenizer        = CLIPTokenizer.from_pretrained(MODEL ,subfolder="tokenizer",use_auth_token=TOKEN)  # Load tokenizer for CLIP model
        num_added_tokens = tokenizer.add_tokens(placeholder_token)                                           # Add placeholder token  from new instnace
        token_ids        = tokenizer.encode(initializer_token, add_special_tokens=False)                     # Convert the initializer_token, placeholder_token to 
        ### Check if initializer_token is a single token or a sequence of tokens ###
        if len(token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")

        initializer_token_id = token_ids[0]                                          # Get token Ids
        placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)    # Covner place holder
        ### Moifiy Text Encoder ###
        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds                       = text_encoder.get_input_embeddings().weight.data
        token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]
    else:
        try:
            print(f"Loading: {loading_dir} ")
            with open( os.path.join(loading_dir,"placeholders.json"), 'r') as handler:
                token_ids = json.load(handler)
                initializer_token_id = token_ids["initializer_token_id"]    # 8890
                placeholder_token_id = token_ids["placeholder_token_id"]   # 49408
            #initializer_token_id = 8890
            #placeholder_token_id =9408

            tokenizer    = CLIPTokenizer.from_pretrained(loading_dir,  subfolder="tokenizer",use_auth_token=TOKEN)     # Load tokenizer for CLIP  fine tuned model
            text_encoder = CLIPTextModel.from_pretrained(loading_dir,  subfolder="text_encoder", use_auth_token=TOKEN) # Load Text Encoder 
        except:
            print(f"Model with path {loading_dir} not loadable")

    print(f"Initalizer token Id: {initializer_token_id}")
    print(f"Placeholder token Id: {placeholder_token_id}")
  
        
    vae             = AutoencoderKL.from_pretrained(MODEL ,        subfolder="vae"         , use_auth_token=TOKEN,fp16=True)  # Image's Lattent Encoder - Decoder 
    unet            = UNet2DConditionModel.from_pretrained(MODEL , subfolder="unet" , use_auth_token=TOKEN,fp16=True)
    noise_scheduler = DDPMScheduler(beta_start=b_start, beta_end=b_end , beta_schedule="scaled_linear", num_train_timesteps=1000, tensor_format="pt")
    ### MOVE MODELS TO GPU ###
    unet         = unet.to(device)          # Move to GPU
    vae          = vae.to(device)           # Move to GPU
    text_encoder = text_encoder.to(device)  # Move to GPU

    if verbose:
        print("Models have been stanciated")

    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )

    #### Freeze vae and unet and some paramaters from text encoder ####
    freeze_params(vae.parameters())       #  Freeze Variational Autoencoder
    if only_tune_encoder:
        freeze_params(unet.parameters()) #  Freeze Unet
    freeze_params(params_to_freeze)      #  Freeze all parameters except for the token embeddings in text encoder

    #Initialize the optimizer
    if only_tune_encoder:
        params     = text_encoder.get_input_embeddings().parameters()
    else:
        params     =  list(text_encoder.get_input_embeddings().parameters()) + list(unet.parameters())
                    
       
    optimizer  = torch.optim.AdamW(params,  lr=learning_rate)
    vae.eval(), unet.train(), text_encoder.train()

    train_project_name = create_random_ascii(3)
    #unet.eval()
    for epoch in range(number_of_epochs):
        vae.eval(), unet.train(), text_encoder.train()
        print(f"epoch: {epoch}")
        for data_point in tqdm(pixel_test_dataset):
            if ( data_point["image"] == None):
                continue
            else:
                pass
                #image      = image/ 127.5 - 1.0
                image      = image.to(device)
                text_class = data_point["text"]
             
             
                text_encoded          = tokenizer( text_class,padding="max_length",truncation=True,max_length=tokenizer.model_max_length,return_tensors="pt").input_ids[0]
                text_encoded          = text_encoded.reshape(1,-1)
                text_encoded          = text_encoded
                encoder_hidden_states = text_encoder( text_encoded  )[0]
                
                # Convert images to latent space
                latents = vae.encode(image).latent_dist.sample().detach() # Get latents from image  latent =  Vae(Image)
                latents = latents * 0.18215                                               # Multiply latents by a factor

                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(latents.device)                     # Create a random image
                bsz   = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss       = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                loss.backward()
                
                ##
                # Get the index for tokens that we want to zero the grads for
                grads                              = text_encoder.get_input_embeddings().weight.grad
                index_grads_to_zero                = torch.arange(len(tokenizer)) != placeholder_token_id
                grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)
                ###
                optimizer.step()
                optimizer.zero_grad()
                loss_list.append( loss.item())

        vae.eval(), unet.eval(), text_encoder.eval()
        epoch_str = f"epoch_{epoch}_"


        if epoch  % save_every_n_epocns  == 0:
            ### Set Name according to pretrain or  train from scratch ###
            if loading_dir == "None":
                base_dir = os.path.join(f"Pipelines","original_encoder_d_" +  epoch_str  + train_project_name )
            
            else:
                base_dir = loading_dir+ "_"+ epoch_str + train_project_name 
            
            print(f"Saving Model in {base_dir}")
            tokenizer.save_pretrained(os.path.join(base_dir,"tokenizer"))
            text_encoder.save_pretrained(os.path.join(base_dir,"text_encoder"))
            vae.save_pretrained(os.path.join(base_dir,"vae"))
            unet.save_pretrained(os.path.join(base_dir,"unet"))

    ### Save PlaceHodlers ###
    holder_dict = {"initializer_token_id":initializer_token_id,"placeholder_token_id":placeholder_token_id }
    if loading_dir == "None":
        with open(os.path.join(base_dir,"placeholders.json"), 'w') as handler:
            json.dump(holder_dict,handler)
    else:
        with open( os.path.join(base_dir,"placeholders.json"), 'w') as handler:
            json.dump(holder_dict,handler)




    ### inferenace ####s
    if difusse_on_end:
        from utils.inference_utils import InferenceStableDifussion_ClIPEncoder                                                           # Import Inferance Class
        scheduler    = LMSDiscreteScheduler(beta_start=b_start, beta_end=b_end,beta_schedule='scaled_linear', num_train_timesteps=1000)  # Set faster scheduler
        print(f"Difussion for T={T} steps")
        diffuser_clip = InferenceStableDifussion_ClIPEncoder(vae,unet,text_encoder,scheduler,tokenizer) # Instanciate diffuser pipeline
        path_to_directory =  os.path.join(base_dir,"Outputs")
        os.mkdir(path_to_directory )
    
        for text_class_i,label_i in zip(text_class,label):
            prompt        = text_class  
            image         = diffuser_clip.prompt_to_img(prompt, height, width , T)[0]  # Prompt to image 
            save_image(image,path_to_directory,prompt=label_i + "_vec")                  # Save IMAGE 
        
        
            
        print(f"Images saved at {path_to_directory}")
    #pdb.set_trace()





if __name__ == "__main__":
    main()
    print("Code Run Succesfully!")
