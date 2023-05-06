from ast import arg
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import MrcleanDataset
from config_follow import parse_option
import warnings

from models.networks import MultiSwinTrans, MultiViTrans, MultiViTransConv

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, weight=self.weight, reduction=self.reduction)
        return loss
    
def fix_random_seeds(seed=42):
    """
    Fix random seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably! '
                  'You may see unexpected behavior when restarting '
                  'from checkpoints.')
    
def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (volume, clinical_data, target) in enumerate(train_loader):
        volume, clinical_data, target = volume.to(device), clinical_data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(volume, clinical_data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        preds = output.argmax(dim=1)
        correct += preds.eq(target).sum().item()
        total += target.size(0)

    train_loss /= len(train_loader)
    train_acc = 100 * correct / total
    print(f'Train Loss: {train_loss:.4f} Train Accuracy: {train_acc:.2f}%')
    return train_loss, train_acc

def validate(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (volume, clinical_data, target) in enumerate(val_loader):
            volume, clinical_data, target = volume.to(device), clinical_data.to(device), target.to(device)
            output = model(volume, clinical_data)
            loss = loss_fn(output, target)
            val_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total += target.size(0)

    val_loss /= len(val_loader)
    val_acc = 100 * correct / total
    print(f'Validation Loss: {val_loss:.4f} Validation Accuracy: {val_acc:.2f}%')
    return val_loss, val_acc

def train_and_validate(model, train_loader, val_loader, optimizer, loss_fn, device, epochs):
    for epoch in range(1, epochs+1):
        print(f'Epoch {epoch}')
        train_loss, train_acc = train(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)
    return train_loss, train_acc, val_loss, val_acc

if __name__ == '__main__':
    
    args = parse_option()
    if args.seed is not None:
        fix_random_seeds(args.seed)
    args.device = 'cuda:'+str(args.gpu)


    # Define your model
    if args.arch == 'ViT':
        model = MultiViTrans(args, channel=256, out_class=args.num_classes, drop=args.dropout, attention=args.attention, follow=args.follow_time)
    elif args.arch == 'ViTConv':
        model = MultiViTransConv(args, channel=256, out_class=args.num_classes, drop=args.dropout, attention=args.attention, follow=args.follow_time)
    elif args.arch == 'SwinT':
        model = MultiSwinTrans(args, channel=256, out_class=args.num_classes, drop=args.dropout, attention=args.attention, follow=args.follow_time)
    else:
        print('ERROR, architecture not found!!!')
    
    
    model.to(args.device)

    # Define your optimizer
    optimizer = optim.Adam(model.parameters(), args.learning_rate,
                                     weight_decay=args.weight_decay)

    # Define your loss function
    loss_fn = FocalLoss()

    # Create your train and validation datasets and data loaders
    train_dataset = MrcleanDataset(args.data_root, args.train_csv, args, phase='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = MrcleanDataset(args.data_root, args.val_csv, args, phase='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Train and validate your model
    train_and_validate(model, train_loader, val_loader, optimizer, loss_fn, args.device, epochs=args.num_epochs)