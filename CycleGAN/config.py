import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Config:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.root_a = '../input/pixel-dataset/trainA'
        self.root_b = "../input/pixel-dataset/trainB"
        self.batch_size = 1
        self.lr = 1e-5
        self.lambda_identity = 0.0
        self.lambda_cycle = 10
        self.num_workers = 2
        self.num_epochs = 50
        self.load_model = True
        self.save_model = True
        self.checkpoint_gen_a = "gena.pth.tar"
        self.checkpoint_gen_b = "genb.pth.tar"
        self.checkpoint_dis_a = "disa.pth.tar"
        self.checkpoint_dis_b = "disb.pth.tar"

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

