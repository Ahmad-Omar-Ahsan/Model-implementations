class Config:
    def __init__(self):
        """Config values
        """
        self.base_model = [
            # expand_ratio, channels, repeats, stride, kernel_size
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
        ]

        self.phi_values = {
            # tuple of: (phi_value, resolution, drop_rate)
            "b0": (0, 224, 0.2),  # alpha, beta, gamma, depth = alpha ** phi
            "b1": (0.5, 240, 0.2),
            "b2": (1, 260, 0.3),
            "b3": (2, 300, 0.3),
            "b4": (3, 380, 0.4),
            "b5": (4, 456, 0.4),
            "b6": (5, 528, 0.5),
            "b7": (6, 600, 0.5),
        }
        self.num_classes=10
        self.criterion = 'l_smooth'
        self.device = 'cuda'
        self.batch_size = 64
        self.epoch = 50
        self.wandb_key = 'cc482a8baec19ffd11294cdda13fa28a935e644c'
        self.project = 'EfficientNet'

