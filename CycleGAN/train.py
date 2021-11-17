from albumentations.pytorch import transforms
import torch
from torch.utils.data.dataset import Dataset
from dataset import PixelDataset
from utils import save_checkpoint, load_checkpoint, seeder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from config import Config
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator
import wandb


def train(disc_a : nn.Module, disc_b : nn.Module, gen_a : nn.Module, gen_b : nn.Module, opt_dis: torch.optim, opt_gen : torch.optim,  loader: Dataset):
    """ Train step for CycleGAN

    Args:
        disc_a (nn.Module): Discriminator A
        disc_b (nn.Module): Discriminator B
        gen_a (nn.Module): Generator A 
        gen_b (nn.Module): Generator B
        opt_dis_a (nn.Module): Optimizer for Discriminator A & B
        opt_gen_a (torch.optim): Optimizer for Generator A & B
        loader (Dataset) : Loader for Pixerart dataset
    """
    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    c = Config()

    for idx, (a, b) in enumerate(loader):
        a = a.to(c.device)
        b = b.to(b.device)

        # Train discriminator a
        fake_a = gen_a(b)
        d_a_real = disc_a(a)
        d_a_fake = disc_a(fake_a.detach())
        d_a_real_loss = mse(d_a_real, torch.ones_like(d_a_real))
        d_a_fake_loss = mse(d_a_fake, torch.zeros_like(d_a_fake))
        d_a_loss = d_a_real_loss + d_a_fake_loss

        # Train discriminator b 
        fake_b = gen_b(a)
        d_b_real = disc_b(b)
        d_b_fake = disc_b(fake_b.detach())
        d_b_real_loss = mse(d_b_real, torch.ones_like(d_b_real))
        d_b_fake_loss = mse(d_b_fake, torch.zeros_like(d_b_fake))
        d_b_loss = d_b_real_loss + d_b_fake_loss

        d_loss = (d_a_loss + d_b_loss) / 2

        opt_dis.zero_grad()
        d_loss.backward()
        opt_dis.step()


        # Train generator a and b
        d_a_fake = disc_a(fake_a)
        d_b_fake = disc_b(fake_b)
        loss_g_a = mse(d_a_fake, torch.ones_like(d_a_fake))
        loss_g_b = mse(d_b_fake, torch.ones_like(d_b_fake))

        # Cycle loss
        cycle_a = gen_a(fake_b)
        cycle_b = gen_b(fake_a)
        cycle_a_loss = l1(a, cycle_a)
        cycle_b_loss = l1(b, cycle_b)

        # Identity loss
        identity_a = gen_a(a)
        identity_b = gen_b(b)
        identity_a_loss = l1(a, identity_a)
        identity_b_loss = l1(b, identity_b)

        G_loss = (
            loss_g_a + loss_g_b + 
            cycle_a_loss * c.lambda_cycle +
            cycle_b_loss * c.lambda_cycle +
            identity_a_loss * c.lambda_identity +
            identity_b_loss * c.lambda_identity
        )
        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()
    wandb.log({
        "G_loss" : G_loss.item(),
        "D_loss" : d_loss.item()
    })
    




def main():
    """Function to initiate training
    """
    c = Config()
    wandb.init(name=c.project)
    disc_a = Discriminator(in_channels=3).to(c.device)
    disc_b = Discriminator(in_channels=3).to(c.device)
    gen_a = Generator(img_channels=3, num_residuals=9).to(c.device)
    gen_b = Generator(img_channels=3, num_residuals=9).to(c.device)

    opt_dis = optim.Adam(
        list(disc_a.parameters()) + list(disc_b.parameters()),
        lr=c.lr,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_a.parameters()) + list(gen_b.parameters()),
        lr=c.lr,
        betas=(0.5, 0.999),
    )

    dataset = PixelDataset(root_a = c.root_a, root_b=c.root_b ,transform=c.transforms)

    loader = DataLoader(
        dataset,
        batch_size=c.BATCH_SIZE,
        shuffle=True,
        num_workers=c.NUM_WORKERS,
    )
    for epoch in range(c.num_epochs):
        train(disc_a, disc_b, gen_a, gen_b, opt_dis, opt_gen, loader)

        if c.SAVE_MODEL:
            save_checkpoint(gen_a, opt_gen, filename=c.checkpoint_gen_a)
            save_checkpoint(gen_b, opt_gen, filename=c.checkpoint_gen_b)
            save_checkpoint(disc_a, opt_dis, filename=c.checkpoint_dis_a)
            save_checkpoint(disc_b, opt_dis, filename=c.checkpoint_dis_b)


if __name__ == "__main__":
    main()