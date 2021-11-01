import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from model import EfficientNet
from config import Config
from torch.utils.data import random_split
from loss import LabelSmoothingLoss
import wandb





def pipeline():
    """Initiates training and testing pipeline for EfficientNet
    """
    conf = Config()
    net=EfficientNet("b0", conf.num_classes, conf.base_model, conf.phi_values).to(conf.device)
    train_loader, val_loader, test_loader = load_dataset(conf.batch_size)
    loss_fn = LabelSmoothingLoss(conf.num_classes)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-5)
    wandb.login(key=conf.wandb_key)
    wandb.init(project=conf.project,config={'batch_size': conf.batch_size,"optimizer": "AdamW", "loss_fn":"crossentropy with labelsmoothing","lr":3e-5})

    for epoch in range(1, conf.epoch+1):
        loss_train = 0.0
        loss_val = 0.0
        correct = 0
        total = 0
        val_correct = 0
        val_total = 0

        for image, label in train_loader:
            imgs, labels = image.to(conf.device), label.to(conf.device)

            outputs = net(imgs)

            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            total += labels.shape[0]
            correct += int((predicted == labels).sum())

            acc_train = float(correct/total)
        
        for val_imgs , val_labels in val_loader:
            val_img, val_label = val_imgs.to(conf.device), val_labels.to(conf.device)

            with torch.no_grad():
                val_output = net(val_img)
                loss = loss_fn(val_output, val_label)
                loss_val += loss.item()

                _, predicted = torch.max(val_output, dim=1)
                val_total += val_labels.shape[0]
                val_correct += int((predicted == val_label).sum())

                acc_val = float(val_correct/ val_total)
        wandb.log({"Train_loss": loss_train})
        wandb.log({"Val_loss": loss_val})
        wandb.log({"Train_accuracy": acc_train})
        wandb.log({"Val_accuracy": acc_val})
        print(f"Train_accuracy:{acc_train:.2f}, Train_loss:{loss_train:.2f}, Val_accuracy:{acc_val:.2f},Val_loss:{loss_val:.2f}")

    PATH = conf.model_save_path
    torch.save(net.state_dict(), PATH)
    with torch.no_grad():
        correct = 0
        total = 0
        for test_img, test_label in test_loader:
            imgs, labels = test_img.to(conf.device),test_label.to(conf.device)
            outputs = net(imgs)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.shape[0]
            correct += int((predicted == labels).sum())
        wandb.log({"Test_accuracy": float(correct/total)})




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
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    return train_loader, val_loader,test_loader

if __name__=="__main__":
    pipeline()