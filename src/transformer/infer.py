from src.transformer.options.test_options import TestOptions
from src.transformer.data import create_dataset
from src.transformer.models import create_model
import cv2
import torchvision.transforms.v2 as transforms
import os 
import torch
import matplotlib.pyplot as plt  
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from src.wct.utils import color_injection
class ContentStyleDataset:
    content_path = './data/contents'
    style_path = "./data/styles"

    def __init__(self): 
        self.size = [150, 300,500,700, 256]  
        self.content_size = 256
        self.fixed_size = 256                                                                                                         
        self.content_images_name = sorted([f for f in os.listdir(self.content_path)], key = lambda x: int(x[:-4].split('_')[1]))      
        self.style_images_name = sorted([f for f in os.listdir(self.style_path)], key = lambda x: int(x[:-4].split('_')[1]))          

    def __len__(self):
        return  50*50*4

    def __getitem__(self,idx):
        ishow = False # default image show 

        if isinstance(idx, tuple) and len(idx) == 2:
            content_ID, style_ID = idx
            size = None  

        elif isinstance(idx,tuple) and len(idx) == 3:
            content_ID , style_ID , size = idx
            assert size in self.size, "The size of the style resolution must belong to [150, 300, 500, 700]"

        elif isinstance(idx, tuple) and len(idx) == 4:
            content_ID, style_ID, size , ishow = idx 
            assert size in self.size, "The size of the style resolution must belong to [150, 300, 500, 700]"

        else:
            raise ValueError("Index be a tuple of (content_ID, style_ID) or (content_ID, style_ID , size) or (content_ID, style_ID, size, ishow )")

        assert (0 < content_ID <= 50  or 0 < style_ID <= 50 ), f"The content_ID and style_ID should in range from 1 to 50"  

        content_name = self.content_images_name[content_ID-1]
        style_name = self.style_images_name[style_ID-1]

        content = cv2.cvtColor(cv2.imread(self.content_path + "/" + content_name), cv2.COLOR_BGR2RGB)
        style = cv2.cvtColor(cv2.imread(self.style_path + "/" + style_name), cv2.COLOR_BGR2RGB)
        height, width = content.shape[:2]
        content_size = (width, height)


        content = cv2.resize(content , (self.content_size , self.content_size), interpolation= cv2.INTER_LINEAR)
        
        if self.size is not None:
            style = cv2.resize(style,(size, size), interpolation= cv2.INTER_LINEAR)

        if ishow :
            fig , axes = plt.subplots(1,2, figsize = (10,5))
            axes[0].imshow(content)
            axes[0].set_title(f"content image {content_ID}")
        
            axes[1].imshow(style)
            axes[1].set_title(f"Style image {style_ID}")

            plt.show()

        return torch.tensor(content).permute(2,0,1).float(), torch.tensor(style).permute(2,0,1).float() , content_size



def infer(content_id , style_id, retain_color):
    transform = transforms.Compose([                
        transforms.Normalize((0.5, 0.5, 0.5),      
                            (0.5, 0.5, 0.5))
    ])

    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)      # create a model given opt.model and other options
    dataset = ContentStyleDataset()
    model.setup(opt)       
    model.parallelize()

    content , style, content_size = dataset[content_id,style_id,256]
    content , style = transform(content/255).unsqueeze(0), transform(style/255).unsqueeze(0)
            
    model.set_input({'A': content, 'B': style, 'A_paths':'./datasets/testA/content','B_paths':'./datasets/testB/style'})  
    model.test()          
    visuals = model.get_current_visuals() 
        
    image = visuals['fake_B']

    img = image.squeeze(0)          
    img = img * 0.5 + 0.5                   
    img = img.permute(1, 2, 0).cpu().numpy()  

    img_uint8 = (img * 255).clip(0,255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    
    resize_img = pil_img.resize(content_size, Image.LANCZOS)
    content_dir =  "./data/contents/content_" + str(content_id)+".jpg"
    final_img = color_injection(content_dir,resize_img, retain_color)
    return final_img