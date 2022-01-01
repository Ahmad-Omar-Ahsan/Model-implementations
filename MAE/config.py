class Config:
    def __init__(self):
        self.img_channels = 3
        self.num_classes = 1000
        self.criterion = 'l_smooth'
        self.device = 'cuda'
        self.batch_size = 64
        self.epoch = 50
        self.wandb_key = 'cc482a8baec19ffd11294cdda13fa28a935e644c'
        self.project = 'MAE'
        self.model_save_path = 'MAE_net.pth'
        self.lr = 3e-5