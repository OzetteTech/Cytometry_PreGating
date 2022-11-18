import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import argparse

def plot_one(gate1, x_axis1, y_axis1, gate2, x_axis2, y_axis2, subject):

    substring = f"./Data_{gate2}/Raw_Numpy/"
    subj_path = subject.split(substring)[1]
    substring = ".npy"
    subj_path = subj_path.split(substring)[0]

    data_table = pd.read_csv(f'./Pred_Results_gate2_cd45/Pred_Results_{subj_path}')

    if x_axis1 != 'Event_length':
        x_axis1 = x_axis1 + '_backup'
    y_axis1 = y_axis1 + '_backup'
    x_axis2 = x_axis2 + '_backup'
    y_axis2 = y_axis2 + '_backup'

    gate1_pred = gate1 + '_pred'
    gate2_pred = gate2 + '_pred'

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    #plot raw:
    data = data_table.copy()
    x = data[x_axis1]
    y = data[y_axis1]
    axs[0, 0].plot(x, y, 'o', color='black', markersize = 3)
    data = data[data[gate1]==1]
    x = data[x_axis1]
    y = data[y_axis1]
    axs[0, 0].plot(x, y, 'o', color='blue', markersize = 3)
    axs[0, 0].set_title("Raw Gate 1 Plot")
    axs[0, 0].set_xlim([0, 10])

    data = data_table.copy()
    x = data[x_axis2]
    y = data[y_axis2]
    axs[0, 1].plot(x, y, 'o', color='black', markersize = 3)
    data = data[data[gate2]==1]
    x = data[x_axis2]
    y = data[y_axis2]
    axs[0, 1].plot(x, y, 'o', color='blue', markersize = 3)
    axs[0, 1].set_title("Raw Gate 2 Plot")
    axs[0, 1].set_xlim([0, 10])

    data = data_table.copy()
    data = data[data[gate1]==1]
    x = data[x_axis2]
    y = data[y_axis2]
    axs[0, 2].plot(x, y, 'o', color='black', markersize = 3)
    data = data[data[gate2]==1]
    x = data[x_axis2]
    y = data[y_axis2]
    axs[0, 2].plot(x, y, 'o', color='blue', markersize = 3)
    axs[0, 2].set_title("Raw Gate 2 Filtered by Gate 1 Plot")
    axs[0, 2].set_xlim([0, 10])

    data = data_table.copy()
    x = data[x_axis1]
    y = data[y_axis1]
    axs[1, 0].plot(x, y, 'o', color='black', markersize = 3)
    data = data[data[gate1_pred]==1]
    x = data[x_axis1]
    y = data[y_axis1]
    axs[1, 0].plot(x, y, 'o', color='blue', markersize = 3)
    axs[1, 0].set_title("Reconstructed Gate 1 Plot")
    axs[1, 0].set_xlim([0, 10])

    data = data_table.copy()
    x = data[x_axis2]
    y = data[y_axis2]
    axs[1, 1].plot(x, y, 'o', color='black', markersize = 3)
    data = data[data[gate2_pred]==1]
    x = data[x_axis2]
    y = data[y_axis2]
    axs[1, 1].plot(x, y, 'o', color='blue', markersize = 3)
    axs[1, 1].set_title("Reconstructed Gate 2 Plot")
    axs[1, 1].set_xlim([0, 10])

    data = data_table.copy()
    data = data[data[gate1_pred]==1]
    x = data[x_axis2]
    y = data[y_axis2]
    axs[1, 2].plot(x, y, 'o', color='black', markersize = 3)
    data = data[data[gate2_pred]==1]
    x = data[x_axis2]
    y = data[y_axis2]
    axs[1, 2].plot(x, y, 'o', color='blue', markersize = 3)
    axs[1, 2].set_title("Reconstructed Gate 2 Filtered by Gate 1 Plot")
    axs[1, 2].set_xlim([0, 10])

    plt.savefig('./Figure_{gate2}/Recon_{subject}.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="cytometry autogating")

    parser.add_argument("--g1", default='gate1_ir', help = 'gate')
    parser.add_argument("--x1", default='Ir191Di___191Ir_DNA1', help = 'x axis measurement') 
    parser.add_argument("--y1", default='Event_length', help = 'y axis measurement')
    parser.add_argument("--g2", default='gate2_cd45', help = 'gate')
    parser.add_argument("--x2", default='Ir193Di___193Ir_DNA2', help = 'x axis measurement') 
    parser.add_argument("--y2", default='Y89Di___89Y_CD45', help = 'y axis measurement')
    args = parser.parse_args()

    gate1 = args.g1
    y_axis1 = args.x1
    x_axis1 = args.y1
    gate2 = args.g2
    y_axis2 = args.x2
    x_axis2 = args.y2

    path_val = pd.read_csv(f"./Data_{gate2}/Train_Test_Val/Val.csv")
    # find path for raw tabular data
    for subject in path_val.Image:
        plot_one(gate1, x_axis1, y_axis1, gate2, x_axis2, y_axis2, subject)
