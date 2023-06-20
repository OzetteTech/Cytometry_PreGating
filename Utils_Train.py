import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from Dataset import dataset

def save_checkpoint(state, filename = "my_checkpoint.pth.tar"):
  print("=> Saving checkpoint")
  torch.save(state, filename)

def load_checkpoint(checkpoing, model):
  print("=> Loading checkpoint")
  model.load_state_dict(checkpoing["state_dict"])

def get_loaders(path_train, path_test, batch_size, train_transform, test_transform, num_workers = 2, pin_memory = True):
  train_ds = dataset(path_train, train_transform)
  train_loader = DataLoader(train_ds, batch_size = batch_size, num_workers = num_workers, pin_memory = pin_memory, shuffle = True)
  test_ds = dataset(path_test, test_transform)
  test_loader = DataLoader(test_ds, batch_size = batch_size, num_workers = num_workers, pin_memory = pin_memory, shuffle = True)
  return train_loader, test_loader

def train_epoch(loader, model, optimizer, loss_fn, device):
  # loop = tqdm(loader)
  for batch_idx, (data, target, subj) in enumerate(loader):
    data = data.type(torch.float32)
    data = data.to(device = device)
    targets = target.float().unsqueeze(1).to(device = device)

    #forward
    # with torch.cuda.amp.autocast():
    predictions = model(data)
    loss = loss_fn(predictions, targets)         

    #backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def check_accuracy(loader, model, device="mps"):
  num_correct = 0
  num_pixels = 0
  dice_score = 0
  model.eval()

  with torch.no_grad():
    for x,y, subj in loader:
      x = x.type(torch.float32)
      x = x.to(device)
      y = y.type(torch.float32)
      y = y.to(device).unsqueeze(1)
      preds = torch.sigmoid(model(x))
      preds = (preds > 0.5).float()
      num_correct += (preds == y).sum()
      num_pixels += torch.numel(preds)
      dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

  print(
      f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
  )
  print(f"Dice score: {dice_score/len(loader)}")
  model.train()

  return num_correct/num_pixels*100, dice_score/len(loader)

def predict_visualization(loader, model, device="mps"):
  model.eval()
  preds_list = []
  y_list = []
  x_list = []
  subj_list = []
  for idx, (x,y,subj) in enumerate(loader):
    x = x.type(torch.float32)
    x = x.to(device=device)
    with torch.no_grad():
      preds = torch.sigmoid(model(x))
      preds = (preds > 0.5).float()
    preds_list.append(preds)
    y_list.append(y.unsqueeze(1))
    x_list.append(x.unsqueeze(1))
    subj_list.append(subj)

  return preds_list, y_list, x_list, subj_list

def testing_plot(accuracy_list, dice_score_list, gate):

  if not os.path.exists(f"./Figure_{gate}"):
    os.mkdir(f"./Figure_{gate}")

  xpoints = np.linspace(1,len(accuracy_list), len(accuracy_list))
  accuracy_list = [x.cpu().numpy() for x in accuracy_list]
  plt.plot(xpoints, accuracy_list)
  plt.title("Testing Accuracy During Traning")
  plt.savefig(f'./Figure_{gate}/Accuracy.png')

  xpoints = np.linspace(1,len(dice_score_list), len(dice_score_list))
  dice_score_list = [float(x.cpu().numpy()) for x in dice_score_list]

  plt.figure()
  plt.plot(xpoints, dice_score_list)
  plt.title("Testing Dice Score During Traning")
  plt.savefig(f'./Figure_{gate}/Dice_Score.png')

