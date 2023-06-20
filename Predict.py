import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from UNet_Model import UNET
from Dataset import dataset
from Utils_Predict import *
from Utils_Train import predict_visualization
import argparse


def val(x_axis, y_axis, gate, path_raw, num_workers, device, seq = False, gate_pre = None):
  # in sequential predicting, the path_raw is the path for prediction of last gate
  PATH = os.path.join(f'./{gate}_model.pt')
  model = UNET().to(device)
  model.load_state_dict(torch.load(PATH))
  model.eval()

  test_transforms = A.Compose(
      [
        ToTensorV2(),
      ],
  )

  path_val = pd.read_csv(f"./Data_{gate}/Train_Test_Val/Val.csv")
  val_ds = dataset(path_val, test_transforms)
  val_loader = DataLoader(val_ds, batch_size = path_val.shape[0], num_workers = num_workers, pin_memory = True)

  val_list, y_val_list, x_list, subj_list = predict_visualization(val_loader, model, device=device)

  val_df = pd.DataFrame(columns=['Subject', 'Accuracy', 'Recall', 'F1 Score'])
  if not os.path.exists(f"./Pred_Results_{gate}"):
    os.mkdir(f"./Pred_Results_{gate}")
  for i in range(path_val.shape[0]):
      data_df_pred, subj_path = mask_to_gate(y_val_list, val_list, x_list, subj_list, x_axis, y_axis, gate, path_raw, seq, gate_pre, worker = 0, idx = i)
      print(subj_path)

      accuracy, recall, f1 = evaluation(data_df_pred, gate)
      print(accuracy, recall, f1)

      entry = [subj_path, accuracy, recall, f1]
      val_df.loc[len(val_df)] = entry

  val_df.to_csv(os.path.join(f"./Figure_{gate}/Val_df.csv"))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    prog="test",
    description="cytometry autogating"
  )
  parser.add_argument("--g", default='gate2_cd45', help = 'gate')
  parser.add_argument("--x", default='Ir193Di___193Ir_DNA2', help = 'x axis measurement') 
  parser.add_argument("--y", default='Y89Di___89Y_CD45', help = 'y axis measurement')
  parser.add_argument("--d", default='cuda', help = 'device')
  args = parser.parse_args()
  gate = args.g
  y_axis = args.x
  x_axis = args.y
  device = args.d

  n_worker = 0

  path_raw = './Raw_Data/'
  # path_raw = '/Pred_Results_{gate}/Pred_Results_'
  val(x_axis, y_axis, gate, path_raw, n_worker, device)