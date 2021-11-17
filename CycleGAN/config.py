import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Config:
    def __init__(self):
        self.device = "cpu"
        self.root_a = 'CycleGAN/training_dataset/trainA'
        self.root_b = "CycleGAN/training_dataset/trainB"
        self.batch_size = 1
        self.lr = 1e-5
        self.lambda_identity = 0.0
        self.lambda_cycle = 10
        self.num_workers = 2
        self.num_epochs = 50
        self.load_model = True
        self.save_model = True
        self.checkpoint_gen_a = "CycleGAN/results/gena.pth.tar"
        self.checkpoint_gen_b = "CycleGAN/results/genb.pth.tar"
        self.checkpoint_dis_a = "CycleGAN/results/disa.pth.tar"
        self.checkpoint_dis_b = "CycleGAN/results/disb.pth.tar"
        self.pretrained_checkpoint_gen_a = 'CycleGAN/pretrained_weights/CycleGAN_weights/genh.pth.tar'
        self.pretrained_checkpoint_gen_b = 'CycleGAN/pretrained_weights/CycleGAN_weights/genz.pth.tar'
        self.pretrained_checkpoint_dis_a = 'CycleGAN/pretrained_weights/CycleGAN_weights/critich.pth.tar'
        self.pretrained_checkpoint_dis_b = 'CycleGAN/pretrained_weights/CycleGAN_weights/criticz.pth.tar'

        self.wandb_key = 'cc482a8baec19ffd11294cdda13fa28a935e644c'
        self.project = 'CycleGAN'

        self.transforms = A.Compose(
            [
                A.Resize(width=256, height=256),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
                ToTensorV2(),
            ],
            additional_targets={"image0": "image"},
        )
        self.in_transform = A.Compose(
            [
                A.Resize(width=256, height=256),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
                ToTensorV2(),
            ],)

