import torch
import torch.nn as nn
import numpy as np
import wandb
import torchvision
from dataset import get_dataloaders_cifar10
from model import DCGAN
from config import Config
from seed_setter import set_all_seeds, set_deterministic
import time

def train():
    """Main function for training DCGAN
    """
    c = Config()
    loss_fn = nn.functional.binary_cross_entropy_with_logits

    fixed_noise = torch.randn(64, c.latent_dim, 1, 1, device=c.device)
    model = DCGAN().to(c.device)
    set_all_seeds(c.random_seed)
    set_deterministic()
    optim_gen = torch.optim.Adam(model.generator.parameters(),
                             betas=(0.5, 0.999),
                             lr=c.generator_lr)

    optim_discr = torch.optim.Adam(model.discriminator.parameters(),
                               betas=(0.5, 0.999),
                               lr=c.discriminator_lr)

    start_time = time.time() 
    custom_transforms = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop((160, 160)),
        torchvision.transforms.Resize([c.img_size[0], c.img_size[1]]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])     
    train_loader, valid_loader, test_loader = get_dataloaders_cifar10(c.batch_size, train_transforms=custom_transforms, test_transforms=custom_transforms, num_workers=4)
    wandb.login(key=c.wandb_key)
    wandb.init(project=c.project,config={'batch_size': c.batch_size,"optimizer": "Adam", "loss_fn":"Binary crossentropy with logits","gen_lr":c.generator_lr, "discr_lr":c.discriminator_lr, "device": c.device})

    for epoch in range(c.num_epochs):
        model.train()  
        for batch_idx, (features, _) in enumerate(train_loader):

            batch_size = features.size(0)

            real_images = features.to(c.device)
            real_labels = torch.ones(batch_size, device=c.device)

            noise = torch.randn(batch_size, c.latent_dim, 1, 1, device=c.device)

            fake_images = model.generator_forward(noise)
            fake_labels = torch.zeros(batch_size, device=c.device)
            flipped_fake_labels = real_labels

            optim_discr.zero_grad()

            discr_pred_real = model.discriminator_forward(real_images).view(-1)                 
            real_loss = loss_fn(discr_pred_real, real_labels)

            discr_pred_fake = model.discriminator_forward(fake_images.detach()).view(-1)
            fake_loss = loss_fn(discr_pred_fake, fake_labels)

            discr_loss = 0.5 * (real_loss + fake_loss)
            discr_loss.backward()

            optim_discr.step()

            optim_gen.zero_grad()

            discr_pred_fake = model.discriminator_forward(fake_images).view(-1)
            gener_loss = loss_fn(discr_pred_fake, flipped_fake_labels)
            gener_loss.backward()

            optim_gen.step()

            wandb.log({
                'train_generator_loss_per_batch': gener_loss.item(),
                "train_discriminator_loss_per_batch": discr_loss.item(),
                })

            predicted_labels_real = torch.where(discr_pred_real.detach() > 0., 1., 0.)
            predicted_labels_fake = torch.where(discr_pred_fake.detach() > 0., 1., 0.)
            acc_real = (predicted_labels_real == real_labels).float().mean()*100.
            acc_fake = (predicted_labels_fake == fake_labels).float().mean()*100.

            wandb.log({
                "train_discriminator_real_acc_per_batch": acc_real.item(),
                "train_discriminator_fake_acc_per_batch": acc_fake.item()
            })
            logging_interval = 100

            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Gen/Dis Loss: %.4f/%.4f' 
                       % (epoch+1, c.num_epochs, batch_idx, 
                          len(train_loader), gener_loss.item(), discr_loss.item()))
            with torch.no_grad():
                fake_images = model.generator_forward(fixed_noise).detach().cpu()
                image_array = torchvision.utils.make_grid(fake_images, padding=2, normalize=True)
                
            images = wandb.Image(image_array, caption=f'Images generated from noise per epoch, epoch:{epoch}')
            wandb.log({f"Generated images at epoch {epoch}": images})
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

if __name__ == "__main__":
    train()