class Config:
    def __init__(self):
        self.model = {
            "image_size": (32, 32),
            "patch_size": (4, 4),
            "num_layers": 8,
            "num_classes": 10,
            "d_model": 192,
            "num_heads": 3,
            "mlp_dim": 256,
            "dropout": 0}

        self.optimizer = {
            "learning_rate": 0.001,
            "weight_decay" : 1e-4
        }
        self.save_dir = "ViT/"
        self.log_dir = "logs/"
        self.epochs = 1
        self.batch_size = 64