import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sn
import argparse

def normalize(data, column):
    df_normalize = data[column]
    min = df_normalize.min()
    max = df_normalize.max()
    df_normalize = (df_normalize-min)/(max-min)
    return df_normalize

def matrix_plot(data_df_selected, x_axis, y_axis, pad_number = 100):
    density = np.zeros((101,101))

    data_df_selected = data_df_selected.round(0)
    data_df_selected_count = data_df_selected.groupby([x_axis, y_axis]).size().reset_index(name="count")

    coord = data_df_selected_count[[x_axis, y_axis]]
    coord = coord.to_numpy().round(0).astype(int).T
    coord[0] = 100 - coord[0] # invert position on plot
    coord = list(zip(coord[0], coord[1]))
    replace = data_df_selected_count[['count']].to_numpy()
    for index, value in zip(coord, replace):
        density[index] = value + pad_number # this is to make boundary more recognizable for visualization
    
    index_x = np.linspace(0,100,101).round(2)
    index_y = np.linspace(0,100,101).round(2)
    df_plot = pd.DataFrame(density, index_x, index_y)

    return df_plot

def export_matrix(file_name, x_axis, y_axis, gate):
    data_df = pd.read_csv(os.path.join("./Raw_Data/", file_name))

    cols = list(data_df)

    # data_df_selected = data_df[[x_axis, y_axis, gate]]
    data_df_selected = data_df[[x_axis, y_axis, cols[-1]]]
    if x_axis != "Event_length":
        data_df_selected[x_axis] = normalize(data_df_selected, x_axis)
        data_df_selected[x_axis] = data_df_selected[x_axis]*100
    if y_axis != "Event_length":
        data_df_selected[y_axis] = normalize(data_df_selected, y_axis)
        data_df_selected[y_axis] = data_df_selected[y_axis]*100

    df_plot = matrix_plot(data_df_selected, x_axis, y_axis, 0)
    sn.heatmap(df_plot, vmax = df_plot.max().max())
    plt.savefig(os.path.join(f'./Data_{gate}/Raw_PNG/', file_name+'.png'))
    np.save(os.path.join(f'./Data_{gate}/Raw_Numpy/', file_name+'.npy'), df_plot.to_numpy())
    plt.close()
    
    # data_df_masked_1 = data_df[data_df.gate1_ir==1]
    # data_df_masked_2 = data_df_selected[data_df_selected[gate]==1]
    data_df_masked_2 = data_df_selected[data_df_selected[cols[-1]]==1]
    df_plot = matrix_plot(data_df_masked_2, x_axis, y_axis, 0)
    df_plot = df_plot.applymap(lambda x: 1 if x != 0 else 0)
    sn.heatmap(df_plot, vmax = df_plot.max().max())
    plt.savefig(os.path.join(f'./Data_{gate}/Mask_PNG/', file_name+'.png'))
    np.save(os.path.join(f'./Data_{gate}/Mask_Numpy/', file_name+'.npy'), df_plot.to_numpy())
    plt.close()

def process_table(x_axis, y_axis, gate):    
    # assign directory
    directory = "./Raw_Data/"
    # iterate over files in
    # that directory
    count = 1
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            export_matrix(filename, x_axis, y_axis, gate)
            print(count)
            count+=1


def filter(path_list):
  path_ = []
  for path in path_list:
    if 'npy' in path:
      path_.append(path)
  return path_

def train_test_val_split(gate):
    imgs = list(sorted(os.listdir(f"./Data_{gate}/Raw_Numpy")))
    masks = list(sorted(os.listdir(f"./Data_{gate}/Mask_Numpy")))

    imgs_ = [f"./Data_{gate}/Raw_Numpy/"+x for x in imgs]
    masks_ = [f"./Data_{gate}/Mask_Numpy/"+x for x in masks]

    imgs = filter(imgs_)
    masks = filter(masks_)
    path = pd.DataFrame(list(zip(imgs, masks)), columns = ['Image','Mask'])

    num_sample = path.shape[0]
    idx_list = list(range(num_sample))
    random.seed(42)
    random.shuffle(idx_list)

    #set the number of train test validate
    train_number = num_sample * 2 // 3
    train_test_number = train_number + 1

    train_idx = idx_list[:train_number]
    test_idx = idx_list[train_number:train_test_number]
    val_idx = idx_list[train_test_number:]

    path_train = path.iloc[train_idx] 
    path_test = path.iloc[test_idx]
    path_val = path.iloc[val_idx]

    path_train.to_csv(f"./Data_{gate}/Train_Test_Val/Train.csv", index=False)
    path_test.to_csv(f"./Data_{gate}/Train_Test_Val/Test.csv", index=False)
    path_val.to_csv(f"./Data_{gate}/Train_Test_Val/Val.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="preprocessing",
        description="cytometry autogating"
    )
    parser.add_argument("--g", default='gate2_cd45', help = 'gate')
    parser.add_argument("--x", default='Ir193Di___193Ir_DNA2', help = 'x axis measurement') 
    parser.add_argument("--y", default='Y89Di___89Y_CD45', help = 'y axis measurement')
    args = parser.parse_args()
    gate = args.g
    y_axis = args.x
    x_axis = args.y

    if not os.path.exists("./Data"):
        os.mkdir(f"./Data_{gate}")
        os.mkdir(f"./Data_{gate}/Mask_Numpy")
        os.mkdir(f"./Data_{gate}/Mask_PNG")
        os.mkdir(f"./Data_{gate}/Raw_Numpy")
        os.mkdir(f"./Data_{gate}/Raw_PNG")
    process_table(x_axis, y_axis, gate)

    if not os.path.exists(f"./Data_{gate}/Train_Test_Val"):
        os.mkdir(f"./Data_{gate}/Train_Test_Val")
    train_test_val_split(gate)