class Config:
    def __init__(self):
        self.device = 'cuda'
        self.random_seed = 42
        self.generator_lr = 0.0002
        self.discriminator_lr = 0.0002
        self.num_epochs = 50
        self.batch_size = 64
        self.img_size = [64, 64, 3]
        self.save_dir = './DCGAN/dcgan.pth'
        self.wandb_key = 'cc482a8baec19ffd11294cdda13fa28a935e644c'
        self.project = 'DCGAN'
        self.latent_dim = 100

