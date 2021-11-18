import torch
from torch._C import device
from utils import load_checkpoint
from generator import Generator
from config import Config
import torch.optim as optim
import numpy as np
from PIL import Image



def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path, h, w,aspect_ratio=1.0):
    """Save a numpy image to the disk
    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    w1, h1, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h1, int(w1 * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h1 / aspect_ratio), w1), Image.BICUBIC)
    else:
        image_pil = image_pil.resize((h, w), Image.BICUBIC) 
    image_pil.save(image_path)


def load_image(image_path):
    """Load image 

    Args:
        image_path (str): image
    """
    c = Config()
    img = np.array(Image.open(image_path).convert("RGB"))
    w, h, _ = img.shape
    img = c.in_transform(image=img)
    img = torch.unsqueeze(img['image'], 0).to(c.device)
    return img, h, w


def inference(input_path: str, output_path: str):
    """ Inference using inputs

    Args:
        input_path (str): [description]
        output_path (str): [description]
    """
    c = Config()
    gen_a = Generator(img_channels=3, num_residuals=9).to(c.device)
    gen_b = Generator(img_channels=3, num_residuals=9).to(c.device)

    opt_gen = optim.Adam(
        list(gen_a.parameters()) + list(gen_b.parameters()),
        lr=c.lr,
        betas=(0.5, 0.999),
    )
    
    load_checkpoint(
            c.checkpoint_gen_a, gen_a, opt_gen, c.lr,
        )

    load_checkpoint(
            c.checkpoint_gen_b, gen_b, opt_gen, c.lr,
        )
    img,h ,w = load_image(input_path)
    out_b = gen_b(img)
    out_b = tensor2im(out_b)
    save_image(out_b, output_path, h, w)



if __name__ == "__main__":
    inference('CycleGAN/test_images/house.jpg', 'CycleGAN/test_images/output_house.jpg')

    