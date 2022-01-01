import torch
from torch import nn 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from models import ViT, MAE
import wandb
from config import Config

def train():
    """ Training function
    """
    conf = Config()
    v = ViT.ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048).to(device=conf.device)

    mae = MAE.MAE(
        encoder = v,
        masking_ratio = 0.75,   
        decoder_dim = 512,      
        decoder_depth = 6       
    ).to(device=conf.device)
    
    wandb.login(key=conf.wandb_key)
    wandb.init(project=conf.project)
    train_loader, val_loader = load_dataset(32)
    optimizer = torch.optim.AdamW(mae.parameters(), lr=3e-4)
    

    for epoch in range(1, 71):
        for image, label in train_loader:
            image, label = image.to(conf.device), label.to(conf.device)
            loss = mae(image)
            loss.backward()
            optimizer.step()
            wandb.log({'Reconstruction loss MSE': loss})
        
        for val_img, val_label in val_loader:
            val_img, val_label = val_img.to(conf.device), val_label.to(conf.device)
            out = mae(val_img)
            img_grid = torchvision.utils.make_grid(out)
            wandb.log({"Examples": img_grid})
    torch.save(v.state_dict(),conf.model_save_path)







def load_dataset(batch_size):
    """Utility function to load the CIFAR10 dataset

    Args:
        batch_size (int): batch size

    Returns:
        [DataLoader]: dataloaders for train,val and test set
    """
    torch.manual_seed(43)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256,256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    val_size = 5000
    train_size = len(dataset) - val_size 

    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                     download=True, transform=transform)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                         shuffle=False, num_workers=2)

    return train_loader, val_loader



if __name__== "__main__":
    train()