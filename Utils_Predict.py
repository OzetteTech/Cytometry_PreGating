import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score
from Data_Preprocessing import normalize

def get_pred_label(data_df, x_axis, y_axis, mask, gate):
    # data_df_selected = data_df[[x_axis, y_axis]]
    # currently the axis name refers to the normalized and the backup refers to the raw data!
    data_df_selected = data_df
    if x_axis != 'Event_length': 
      data_df_selected[x_axis+'_backup'] = data_df_selected[x_axis].copy()
      data_df_selected[x_axis] = normalize(data_df_selected, x_axis)*100
      data_df_selected[x_axis] = data_df_selected[x_axis].round(0).astype(int)
    if y_axis != 'Event_length': 
      data_df_selected[y_axis+'_backup'] = data_df_selected[y_axis].copy()
      data_df_selected[y_axis] = normalize(data_df_selected, y_axis)*100
      data_df_selected[y_axis] = data_df_selected[y_axis].round(0).astype(int)

    # data_df_selected = pd.concat([data_df_selected, data_df[[gate]]], axis = 1)

    index_x = np.linspace(0,100,101).round(0).astype(int).astype(str)
    index_y = np.linspace(0,100,101).round(0).astype(int).astype(str)
    index_x = index_y[::-1] # invert y axis
    df_plot = pd.DataFrame(mask.cpu().numpy().reshape(101,101), index_x, index_y)

    gate_pred = gate + '_pred'
    # data_df_selected[gate_pred] = [int(df_plot.loc[str(a), str(b)]) for (a, b) in zip(data_df_selected[x_axis], data_df_selected[y_axis])]
    pred_label_list = []
    for i in range(data_df_selected.shape[0]):
      # print(data_df_selected.columns)
      a = data_df_selected.loc[i, x_axis]
      b = data_df_selected.loc[i, y_axis]
      if a > 100 or b > 100: 
        # outlier - label as 0 and continue
        pred_label_list.append(0)
        continue
      pred_label = int(df_plot.loc[str(a), str(b)])
      true_label = data_df_selected.loc[i, gate]
      pred_label_list.append(pred_label)
    data_df_selected[gate_pred] = pred_label_list

    return data_df_selected


def mask_to_gate(y_list, pred_list, x_list, subj_list, x_axis, y_axis, gate, path_raw, seq, gate_pre, worker = 0, idx = 0):
  raw_img = x_list[worker][idx]
  mask_img = y_list[worker][idx]
  mask_pred = pred_list[worker][idx]
  subj_path = subj_list[worker][idx]

  # plot
  fig, axs = plt.subplots(1, 4, figsize=(15, 15))

  #plot raw:
  axs[0].imshow(raw_img.cpu().reshape(101,101))
  axs[0].set_title("Raw Density Plot")

  #plot true label:
  axs[1].imshow(mask_img.cpu().reshape(101,101))
  axs[1].set_title("True Mask")

  #plot prediction:
  axs[2].imshow(mask_pred.cpu().reshape(101,101))
  axs[2].set_title("Predicted Mask")

  # find path for raw tabular data
  substring = os.path.join(f"./Data_{gate}/Raw_Numpy/")
  subj_path = subj_path.split(substring)[1]
  substring = ".npy"
  subj_path = subj_path.split(substring)[0]

  raw_table = pd.read_csv(path_raw + subj_path)

  data_df_pred = get_pred_label(raw_table, x_axis, y_axis, mask_pred, gate)
  if seq:
    data_df_pred[gate + '_pred'] = data_df_pred.apply(lambda x: 1 if x[gate + '_pred']==1 and x[gate_pre + '_pred']==1 else 0, axis = 1)
  data_df_pred.to_csv(os.path.join(f'./Pred_Results_{gate}/Pred_Results_' + subj_path))


  data_df_masked = data_df_pred[data_df_pred[gate + '_pred']==1]

  df_plot = matrix_plot(data_df_masked, x_axis, y_axis)
  axs[3].imshow(df_plot)
  axs[3].set_title("Reconstructed Mask")

  plt.savefig(os.path.join(f"./Figure_{gate}/" + subj_path + ".png"))

  return data_df_pred, subj_path

def evaluation(data_df_pred, gate):

    accuracy = accuracy_score(data_df_pred[gate], data_df_pred[gate+'_pred'])
    recall = recall_score(data_df_pred[gate], data_df_pred[gate+'_pred'])
    f1 = f1_score(data_df_pred[gate], data_df_pred[gate+'_pred'])

    return accuracy, recall, f1

def matrix_plot(data_df, x_axis, y_axis, pad_number = 100):
    density = np.zeros((101,101))
    data_df_selected = data_df[[x_axis, y_axis]]
    data_df_selected = data_df_selected.round(2) # round to nearest 0.005
    data_df_selected_count = data_df_selected.groupby([x_axis, y_axis]).size().reset_index(name="count")

    coord = data_df_selected_count[[x_axis, y_axis]]
    # do not normalize event length
    coord = coord.to_numpy().round(0).astype(int).T
    coord[0] = 100 - coord[0] # invert position on plot
    coord = list(zip(coord[0], coord[1]))
    replace = data_df_selected_count[['count']].to_numpy()
    for index, value in zip(coord, replace):
        density[index] = 100 # value + pad_number # this is to make boundary more recognizable for visualization
    
    index_x = np.linspace(0,1,101).round(2)
    index_y = np.linspace(0,1,101).round(2)
    df_plot = pd.DataFrame(density, index_x, index_y)

    return df_plot
