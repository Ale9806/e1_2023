import textwrap
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from datasets import load_dataset

class PixelTextDataset:
    
    def __init__(self,image_size:int=512,  font_size:int = 55, font:str='Arial', margin:int=20, offset:int =20, ) -> None:

        """
        image_size <int>: Size of the image to be generated
        font_size  <int>: Size of the font to be used
        font       <str>: Font to be used
        margin     <int>: Margin of the text from the left side of the image
        offset     <int>: Offset of the text from the top side of the image
        """
        self.text_ds    = load_dataset('lambdalabs/pokemon-blip-captions', split='train')
        self.size       = len(self.text_ds)
        self.image_size = image_size
        self.font       = font
        self.font_size  = font_size
        self.margin     = margin
        self.offset     = offset
    



    def __getitem__(self, idx) -> dict:        
        image   = Image.new('RGB', ( self.image_size  , self.image_size ))
        draw    = ImageDraw.Draw(image)
        font = ImageFont.truetype(self.font,self.font_size)
        text    = self.text_ds['text'][idx]


    
        for line in textwrap.wrap(text, width=15):
            draw.text((self.margin, self.offset), line, font=font, fill="white")
            self.offset += font.getsize(line)[1]
        
        return {'text': text, 'image': np.array(image)}
    
    def __len__(self):
        return self.size
