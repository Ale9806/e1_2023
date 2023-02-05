import os 
import json
import torch 
import random
import string
import matplotlib.pyplot as plt 


def random_string_generator(str_size:int, allowed_chars:list) -> str:
    return ''.join(random.choice(allowed_chars) for x in range(str_size))


def create_random_ascii(str_size:int) -> str:
    chars = string.ascii_letters 
    return random_string_generator(str_size, chars)
   

def save_image(image,path_to_directory:str="SyntheticGeneratedHaE",prompt:str="Image",verbose:bool=True) -> None:
    """ Saves Images in specified directory (generating name based on prompt)
        parameters:
            path_to_directory<str>: path to save image
            prompt<str>:            prompt
        output:
            saves image 
    """
    
    random_run = create_random_ascii(3)
    save_name  = f'image_{prompt}_{random_run}.png'
    image.save(os.path.join("Synthetic",save_name))
   

  

def load_config_files(PathToConfigFile ,hyperparameters_path, token_path):
    with open(os.path.join(hyperparameters_path)) as f:
        hyperparameters = json.load(f)

    with open(os.path.join(token_path)) as f:
        parameters = json.load(f)
        TOKEN      =  parameters["TOKEN"]
        MODEL      = parameters["MODEL"]
        REVISION   = parameters["REVISION"]

    with open(PathToConfigFile) as handler:
        config            = json.load(handler)
        patch_data_path   = config["patch_data_path"] # path to subfolder with patches
        csv_path          = config["path_csv"]        # path to csv file with rna data and image names
        transforms        = config["transforms"]      # transforms to apply to the image
        max_patches_total = config["max_patches_total"]  # maximum number of patches to use
        quick             = config["quick"]              # if True, use only 10 patches

    return hyperparameters, TOKEN, MODEL,REVISION,patch_data_path ,csv_path ,transforms,max_patches_total, quick 


def freeze_params(params):
    """ Freeze pyrtorch weights to perform trainning"""
    for param in params:
        param.requires_grad = False


def manage_cuda(d=0):
    """" Returns Available device  (either cuda or cpu)"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.cuda.empty_cache()
    device    = f"cuda:{d}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    return device


def my_collate(batch):
    """ Removes Nans from dataset """
    batch = [x  for x in batch if x!= None]
    return torch.utils.data.dataloader.default_collate(batch)

