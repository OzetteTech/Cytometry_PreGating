import torch
import os
import random 
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from UNet_Model import UNET
from Utils_Train import *

def train(gate, learning_rate, device, batch_size, epoches, n_worker):
  # process data
  path_train = pd.read_csv(f'./Data_{gate}/Train_Test_Val/Train.csv')
  path_test = pd.read_csv(f"./Data_{gate}/Train_Test_Val/Test.csv")

  train_transforms = A.Compose(
      [
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        ToTensorV2(),
      ],
  )

  test_transforms = A.Compose(
      [
        ToTensorV2(),
      ],
  )

  model = UNET(in_channels = 1, out_channels = 1).to(device)
  loss_fn = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(model.parameters(), lr = learning_rate)

  train_loader, test_loader = get_loaders(path_train, path_test, batch_size, train_transforms, test_transforms, num_workers = n_worker, pin_memory = True)

  accuracy_list = []
  dice_score_list = []

  # train
  for epoch in range(epoches):
    train_epoch(train_loader, model, optimizer, loss_fn, device)

    # check accuracy
    accuracy, dice_score = check_accuracy(test_loader, model, device= device)
    accuracy_list.append(accuracy)
    dice_score_list.append(dice_score.cpu())

  PATH = os.path.join('./', gate+'_model.pt')
  torch.save(model.state_dict(), PATH)
  # pred_list, y_list, x_list, subj_list = predict_visualization(test_loader, model)
  testing_plot(accuracy_list, dice_score_list, gate)

if __name__ == "__main__":

  gate = 'gate2_cd45'

  # hyperparameters
  learning_rate = 1e-4
  # device = "cuda" if torch.cuda.is_available() else "cpu"
  device = "mps"
  batch_size = 8
  epoches = 20
  n_worker = 0

  train(gate, learning_rate, device, batch_size, epoches, n_worker)