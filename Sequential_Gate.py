from Data_Preprocessing import *
from Train import *
from Predict import *
import os
import argparse
parser = argparse.ArgumentParser(description="cytometry autogating")

parser.add_argument("--g1", default='gate1_ir', help = 'gate')
parser.add_argument("--x1", default='Ir191Di___191Ir_DNA1', help = 'x axis measurement') 
parser.add_argument("--y1", default='Event_length', help = 'y axis measurement')
parser.add_argument("--g2", default='gate2_cd45', help = 'gate')
parser.add_argument("--x2", default='Ir193Di___193Ir_DNA2', help = 'x axis measurement') 
parser.add_argument("--y2", default='Y89Di___89Y_CD45', help = 'y axis measurement')
parser.add_argument("--d", default='mps', help="training device")
args = parser.parse_args()

gate1 = args.g1
y_axis1 = args.x1
x_axis1 = args.y1
gate2 = args.g2
y_axis2 = args.x2
x_axis2 = args.y2
device = args.d

n_worker = 0
learning_rate = 1e-4
batch_size = 8
epoches = 20

###########################################################
# gate 1
###########################################################

# 1. preprocess data
if not os.path.exists("./Data"):
    os.mkdir(f"./Data_{gate1}")
    os.mkdir(f"./Data_{gate1}/Mask_Numpy")
    os.mkdir(f"./Data_{gate1}/Mask_PNG")
    os.mkdir(f"./Data_{gate1}/Raw_Numpy")
    os.mkdir(f"./Data_{gate1}/Raw_PNG")
process_table(x_axis, y_axis, gate1)

if not os.path.exists(f"./Data_{gate1}/Train_Test_Val"):
    os.mkdir(f"./Data_{gate1}/Train_Test_Val")
train_test_val_split(gate1)

# 2. train
train(gate1, learning_rate, device, batch_size, epoches, n_worker)

# 3. predict
path_raw = './Raw_Data/'
val(x_axis1, y_axis1, gate1, path_raw, n_worker, device, seq = False)

###########################################################
# gate 2
###########################################################

# 1. preprocess data
if not os.path.exists("./Data"):
    os.mkdir(f"./Data_{gate2}")
    os.mkdir(f"./Data_{gate2}/Mask_Numpy")
    os.mkdir(f"./Data_{gate2}/Mask_PNG")
    os.mkdir(f"./Data_{gate2}/Raw_Numpy")
    os.mkdir(f"./Data_{gate2}/Raw_PNG")
process_table(x_axis1, y_axis1, gate2)

if not os.path.exists(f"./Data_{gate2}/Train_Test_Val"):
    os.mkdir(f"./Data_{gate2}/Train_Test_Val")
train_test_val_split(gate2)

# 2. train
train(gate2, learning_rate, device, batch_size, epoches, n_worker)

# 3. predict
path_raw = f'./Pred_Results_{gate1}/Pred_Results_'
val(x_axis1, y_axis1, gate2, path_raw, n_worker, device, seq = True, gate_pre = gate1)