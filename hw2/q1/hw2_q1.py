# -*- coding: utf-8 -*-


#https://github.com/MedMNIST/MedMNIST


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms

from medmnist import BloodMNIST, INFO

import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score


#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"


# Data Loading

data_flag = 'bloodmnist'
print(data_flag)
info = INFO[data_flag]
print(len(info['label']))
n_classes = len(info['label'])

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

batch_size = 64
epochs = 200

import time

# --------- Before Training ----------
total_start = time.time()

#Training Function

def train_epoch(loader, model, criterion, optimizer):
    total_loss = 0

    model.train()
    
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        y_hat = model(X)
        y = y.squeeze(1)
        loss = criterion(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

   

    return total_loss / len(loader)

#Evaluation Function

def evaluate(loader, model):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.squeeze().long()

            outputs = model(imgs)
            preds += outputs.argmax(dim=1).cpu().tolist()
            targets += labels.tolist()

    return accuracy_score(targets, preds)


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(range(epochs), plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')

train_dataset = BloodMNIST(split='train', transform=transform, download=True, size=28)
val_dataset   = BloodMNIST(split='val',   transform=transform, download=True, size=28)
test_dataset  = BloodMNIST(split='test',  transform=transform, download=True, size=28)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# initialize the model
"""
model = nn.Sequential( ## input 3 x 28 x 28
    nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1), ## 32 x 28 x 28
    nn.ReLU(),
    nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1), ## 64 x 28 x 28
    nn.ReLU(),
    nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1), ## 128 x 28 x 28
    nn.ReLU(),
    nn.Flatten(), ## 100352
    nn.Linear(100352, 256), ## 256
    nn.ReLU(),
    nn.Linear(256,8), ## 8
    #nn.Softmax()
)
#"""
#"""
model = nn.Sequential( ## input 3 x 28 x 28
    nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1), ## 32 x 28 x 28
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2), ## 32 x 14 x 14
    nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1), ## 64 x 14 x 14
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2), ## 64 x 7 x 7
    nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1), ## 128 x 7 x 7
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2), ## 128 x 3 x 3
    nn.Flatten(), ## 1152
    nn.Linear(1152, 256), ## 256
    nn.ReLU(),
    nn.Linear(256,8), ## 8
    # nn.Softmax(dim = 1)
)
#"""

model = model.to(device)
# get an optimizer
optim = torch.optim.Adam

optimizer = optim(model.parameters(), lr = 0.001)

# get a loss criterion
criterion = nn.CrossEntropyLoss()



# training loop

train_losses = []
val_accs = []
test_accs = []
for epoch in range(epochs):

    epoch_start = time.time()

    train_loss = train_epoch(train_loader, model, criterion, optimizer)
    val_acc = evaluate(val_loader, model)

    train_losses.append(train_loss)
    val_accs.append(val_acc)

    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start

    print(f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | "
        f"Time: {epoch_time:.2f} sec")

    #Test Accuracy
    test_acc = evaluate(test_loader, model)
    print("Test Accuracy:", test_acc)
    test_accs.append(test_acc)


#Save the model
torch.save(model.state_dict(), "bloodmnist_logits_cnn_maxpool_softmax_200.pth")
print("Model saved as bloodmnist_logits_cnn.pth")


# --------- After Training ----------
total_end = time.time()
total_time = total_end - total_start

print(f"\nTotal training time: {total_time/60:.2f} minutes "
      f"({total_time:.2f} seconds)")

print('Final Test acc: %.4f' % test_acc)

config = "{}-{}-{}-{}-{}".format(0.001, "Adam", "maxpool", "softmax","200")
#config = "{}".format(str(0.1))

plot(epochs, train_losses, ylabel='Loss', name='CNN-training-loss-{}'.format(config))
plot(epochs, val_accs, ylabel='Accuracy', name='CNN-validation-accuracy-{}'.format(config))
plot(epochs, test_accs, ylabel='Accuracy', name='CNN-test-accuracy-{}'.format(config))

##final test accuracy 
"""
normal - 0.9307

maxpool - 0.9369
"""