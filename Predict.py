import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from UNet_Model import UNET
from Dataset import dataset
from Utils_Predict import *
from Utils_Train import predict_visualization


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

  val_list, y_val_list, x_list, subj_list = predict_visualization(val_loader, model)

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

  y_axis = 'Ir193Di___193Ir_DNA2' # x axis in plot
  x_axis = 'Y89Di___89Y_CD45' # y axis in plot  
  gate = 'gate2_cd45'

  device = "mps"
  n_worker = 0

  path_raw = './Raw_Data/'
  # path_raw = '/Pred_Results_{gate}/Pred_Results_'
  val(x_axis, y_axis, gate, path_raw, n_worker, device)