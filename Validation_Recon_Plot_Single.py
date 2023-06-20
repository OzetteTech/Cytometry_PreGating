import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import argparse

def plot_one(gate, x_axis, y_axis, subject):

    substring = f"./Data_{gate}/Raw_Numpy/"
    subject = subject.split(substring)[1]
    substring = ".npy"
    subject = subject.split(substring)[0]

    data_table = pd.read_csv(f'./Pred_Results_{gate}/Pred_Results_{subject}')

    if x_axis != 'Event_length':
        x_axis = x_axis + '_backup'
    if y_axis != 'Event_length':
        y_axis = y_axis + '_backup'

    gate_pred = gate + '_pred'

    fig, axs = plt.subplots(2, 1, figsize=(5, 10))

    #plot raw:
    data = data_table.copy()
    x = data[x_axis]
    y = data[y_axis]
    axs[0].plot(x, y, 'o', color='black', markersize = 3)
    data = data[data[gate]==1]
    x = data[x_axis]
    y = data[y_axis]
    axs[0].plot(x, y, 'o', color='blue', markersize = 3)
    axs[0].set_title("Raw Gate Plot")
    axs[0].set_xlim([0, 10])


    data = data_table.copy()
    x = data[x_axis]
    y = data[y_axis]
    axs[1].plot(x, y, 'o', color='black', markersize = 3)
    data = data[data[gate_pred]==1]
    x = data[x_axis]
    y = data[y_axis]
    axs[1].plot(x, y, 'o', color='blue', markersize = 3)
    axs[1].set_title("Reconstructed Gate Plot")
    axs[1].set_xlim([0, 10])

    plt.savefig(f'./Figure_{gate}/Recon_Single_{subject}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="cytometry autogating")

    parser.add_argument("--g", default='gate2_cd45', help = 'gate')
    parser.add_argument("--x", default='Ir193Di___193Ir_DNA2', help = 'x axis measurement') 
    parser.add_argument("--y", default='Y89Di___89Y_CD45', help = 'y axis measurement')
    args = parser.parse_args()

    gate = args.g
    x_axis = args.x
    y_axis = args.y

    path_val = pd.read_csv(f"./Data_{gate}/Train_Test_Val/Val.csv")
    # find path for raw tabular data
    for subject in path_val.Image:
        plot_one(gate, x_axis, y_axis, subject)
