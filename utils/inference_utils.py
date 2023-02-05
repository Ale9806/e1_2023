import torch
from   torch import autocast
from tqdm.auto import tqdm
from   PIL import Image

class InferenceStableDifussion_ClIPEncoder:
    def __init__(self,vae,unet,text_encoder,scheduler,tokenizer,device:str="cuda"):
        self.device       = device
        self.tokenizer    = tokenizer      # Tokenizer for prompt 
        self.vae          = vae            # VAE for inference 
        self.unet         = unet           # Unet for noice removal
        self.text_encoder = text_encoder   # Text encoder for tokenizer embedding
        self.scheduler    = scheduler      # Scheduler for infrence 
        
        ### Make Sure models are in evaluation Mode ###
        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()



    def get_text_embeds(self,prompt:str) -> torch.tensor:
        """Tokenize text and get embeddings of both prompt and latten seed (to genereate random output)
        params:
            prompt<str>:                   Text to convert to image
        output: 
            text_embeddings<torch.tensor>: Text embeddings in tensor 
        """

        text_embeddings = self.get_text_embeds_utils(prompt)              # Get text embeddings

        ## Do the same for unconditional embeddings (laten seed) ##
        laten_seed        = [''] * len(prompt)                            # Create Latent Seed (this isd used to generate a new image every time)
        uncond_embeddings = self.get_text_embeds_utils(laten_seed )       # Get embeddings for Latent Seed

        ### GET LATENTS FROM UNET ###
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings]) # Concatenate text emmbeding and latent_seed embedding Cat for final embeddings

        return text_embeddings




    def get_text_embeds_utils(self,prompt:str) -> torch.tensor:
        """ Gets Text embeddings by  tokenizing promt  and feeding it to encode
        params:
            prompt<str>:                   Text to convert to image
        output: 
            text_embeddings<torch.tensor>: Text embeddings in tensor 
        """
        ### Tokenize Text ###
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,truncation=True, return_tensors='pt') # Tokenize text
    
        ### Embed  Tokenize prompt###
        with torch.no_grad():                                                             # Disable gradient tracking
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]  # Get text embeddings
        return text_embeddings



    def produce_latents(self,text_embeddings, height:int=512, width:int=512,num_inference_steps:int=50, guidance_scale:float=7.5, latents:bool=None) -> torch.tensor:

        """ Produces Final latten by denoissing inital laten seed at time t=T until time t=0 
            params:
                text_embeddings<torch.tensor>: tokenized promt embeddings
            output:
                latents <torch.tensor>:        latten at time t=0

        """
        if latents is None:                                                                                        # If no latents are provided
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8)) # Create random latents

        latents = latents.to(self.device)                  # Move latents to device
        self.scheduler.set_timesteps(num_inference_steps)  # Set Scheduler (to calculate Betas in order to apply Diffusion process in one step and not on a iteration basis
        latents = latents *self.scheduler.sigmas[0]        # Scale latents by the first sigma value (this is done to make sure the latents are in the same range as the model expects)

        #with autocast(self.device):                           
        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            sigma              = self.scheduler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, i, latents)['prev_sample']

        return latents




    def decode_img_latents(self,latents) -> Image:
        """ Get latent at t=0 and use vae to decoded into image space
            params:
                latents <torch.tensor>:        latten at time t=0
            output:
                pil_images

                
        """
        latents = 1 / 0.18215 * latents                  # Multiply by guidding constant

        with torch.no_grad():                                      # Run without keeping track of grads
            imgs = self.vae.decode(latents).sample                 # Decode latten and transform it into image 
            imgs = (imgs / 2 + 0.5).clamp(0, 1)                    # Clamp input into [0,1] values 
            imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy() # Detach images from device

        imgs = (imgs * 255).round().astype('uint8')                # set images to values from [0,255] and unit8 format
        pil_images = [Image.fromarray(image) for image in imgs]    # Unpack images into array 
        return pil_images



    #### Main Function of this class: calls above functions to perform inferance ##
    def prompt_to_img(self,prompts, height:int=512, width:int=512, num_inference_steps:int=50,guidance_scale:float=7.5, latents=None) -> Image:

        if isinstance(prompts, str): # If prompt is a string
            prompts = [prompts]      # Transform prompt to list

        # STAGE 1: Get text embeddings by   (prompt -(tokenizer)-> tokenized prompt -(text_encoder)-> text embeddings)
        text_embeds = self.get_text_embeds(prompts)    # Prompts -> text embeds
    
        # STAGE 2: Get img latents at t=0 by ( text embeddings -(unet + scheduler)-> lattent  repeat untill  t=0)
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents,num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)  # Text embeds -> img latents
    
        # STAGE 3: Decode latten to image space 
        imgs = self.decode_img_latents(latents) # Img latents -> imgs

        return imgs


class InferenceStableDifussion_CustomEncoder:
    def __init__(self,vae,unet,text_encoder,scheduler,device:str="cuda"):
        self.device       = device
        self.vae          = vae            # VAE for inference 
        self.unet         = unet           # Unet for noice removal
        self.text_encoder = text_encoder   # Text encoder for tokenizer embedding
        self.scheduler    = scheduler      # Scheduler for infrence 
        
        ### Make Sure models are in evaluation Mode ###
        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()



    def get_text_embeds(self,prompt:str) -> torch.tensor:
        """Tokenize text and get embeddings of both prompt and latten seed (to genereate random output)
        params:
            prompt<str>:                   Text to convert to image
        output: 
            text_embeddings<torch.tensor>: Text embeddings in tensor 
        """

        text_embeddings = self.get_text_embeds_utils(prompt)                       # Get text embeddings

        ## Do the same for unconditional embeddings (laten seed) ##
        uncond_embeddings = self.get_text_embeds_utils(torch.zeros_like(prompt) )  # Get embeddings for Latent Seed

        ### GET LATENTS FROM UNET ###
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])   # Concatenate text emmbeding and latent_seed embedding Cat for final embeddings
           
        return text_embeddings




    def get_text_embeds_utils(self,text_input) -> torch.tensor:
        """ Gets Text embeddings by  tokenizing promt  and feeding it to encode
        params:
            prompt<str>:                   Text to convert to image
        output: 
            text_embeddings<torch.tensor>: Text embeddings in tensor 
        """
      
        ### Embed  Tokenize prompt###
        with torch.no_grad():                                                            # Disable gradient tracking
            text_embeddings = self.text_encoder(text_input.to(self.device))    # Get text embeddings
        return text_embeddings



    def produce_latents(self,text_embeddings, height:int=512, width:int=512,num_inference_steps:int=50, guidance_scale:float=7.5, latents:bool=None) -> torch.tensor:

        """ Produces Final latten by denoissing inital laten seed at time t=T until time t=0 
            params:
                text_embeddings<torch.tensor>: tokenized promt embeddings
            output:
                latents <torch.tensor>:        latten at time t=0

        """
        if latents is None:                                                                                        # If no latents are provided
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8)) # Create random latents

        latents = latents.to(self.device)                  # Move latents to device
        self.scheduler.set_timesteps(num_inference_steps)  # Set Scheduler (to calculate Betas in order to apply Diffusion process in one step and not on a iteration basis
        latents = latents *self.scheduler.sigmas[0]        # Scale latents by the first sigma value (this is done to make sure the latents are in the same range as the model expects)

        with autocast(self.device):                           
            for i, t in tqdm(enumerate(self.scheduler.timesteps)):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)
                sigma              = self.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                    # perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, i, latents)['prev_sample']

        return latents




    def decode_img_latents(self,latents) -> Image:
        """ Get latent at t=0 and use vae to decoded into image space
            params:
                latents <torch.tensor>:        latten at time t=0
            output:
                pil_images

                
        """
        latents = 1 / 0.18215 * latents                  # Multiply by guidding constant

        with torch.no_grad():                                      # Run without keeping track of grads
            imgs = self.vae.decode(latents).sample                 # Decode latten and transform it into image 
            imgs = (imgs / 2 + 0.5).clamp(0, 1)                    # Clamp input into [0,1] values 
            imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy() # Detach images from device

        imgs = (imgs * 255).round().astype('uint8')                # set images to values from [0,255] and unit8 format
        pil_images = [Image.fromarray(image) for image in imgs]    # Unpack images into array 
        return pil_images



    #### Main Function of this class: calls above functions to perform inferance ##
    def prompt_to_img(self,prompts, height:int=512, width:int=512, num_inference_steps:int=50,guidance_scale:float=7.5, latents=None) -> Image:

        # STAGE 1: Get text embeddings by   (prompt -(tokenizer)-> tokenized prompt -(text_encoder)-> text embeddings)
        text_embeds = self.get_text_embeds(prompts)    # Prompts -> text embeds
    
        # STAGE 2: Get img latents at t=0 by ( text embeddings -(unet + scheduler)-> lattent  repeat untill  t=0)
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents,num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)  # Text embeds -> img latents
    
        # STAGE 3: Decode latten to image space 
        imgs = self.decode_img_latents(latents) # Img latents -> imgs

        return imgs