from Data_Preprocessing import *
from Train import *
from Predict import *
import os
import argparse
parser = argparse.ArgumentParser(description="cytometry autogating")

parser.add_argument("--g", default='gate2_cd45', help = 'gate')
parser.add_argument("--x", default='Ir193Di___193Ir_DNA2', help = 'x axis measurement') 
parser.add_argument("--y", default='Y89Di___89Y_CD45', help = 'y axis measurement')
parser.add_argument("--d", default='mps', help="training device")
args = parser.parse_args()

# y_axis = 'Ir193Di___193Ir_DNA2' # x axis in plot
# x_axis = 'Y89Di___89Y_CD45' # y axis in plot  
# gate = 'gate2_cd45'

gate = args.g
y_axis = args.x
x_axis = args.y
device = args.d
n_worker = 0
learning_rate = 1e-4
batch_size = 8
epoches = 20
n_worker = 0

# 1. preprocess data
if not os.path.exists(f"./Data_{gate}"):
    os.mkdir(f"./Data_{gate}")
    os.mkdir(f"./Data_{gate}/Mask_Numpy")
    os.mkdir(f"./Data_{gate}/Mask_PNG")
    os.mkdir(f"./Data_{gate}/Raw_Numpy")
    os.mkdir(f"./Data_{gate}/Raw_PNG")
# process_table(x_axis, y_axis, gate)

if not os.path.exists(f"./Data_{gate}/Train_Test_Val"):
    os.mkdir(f"./Data_{gate}/Train_Test_Val")
# train_test_val_split(gate)

# 2. train
# train(gate, learning_rate, device, batch_size, epoches, n_worker)

# 3. predict
path_raw = './Raw_Data/'
val(x_axis, y_axis, gate, path_raw, n_worker, device)