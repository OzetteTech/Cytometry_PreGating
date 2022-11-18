from Data_Preprocessing import *
from Train import *
from Predict import *
import os

y_axis = 'Ir193Di___193Ir_DNA2' # x axis in plot
x_axis = 'Y89Di___89Y_CD45' # y axis in plot  
gate = 'gate2_cd45'

device = "mps"
# device = "cuda" if torch.cuda.is_available() else "cpu"
n_worker = 0
learning_rate = 1e-4
batch_size = 8
epoches = 20
n_worker = 0

# 1. preprocess data
if not os.path.exists("./Data"):
    os.mkdir(f"./Data_{gate}")
    os.mkdir(f"./Data_{gate}/Mask_Numpy")
    os.mkdir(f"./Data_{gate}/Mask_PNG")
    os.mkdir(f"./Data_{gate}/Raw_Numpy")
    os.mkdir(f"./Data_{gate}/Raw_PNG")
process_table(x_axis, y_axis, gate)

if not os.path.exists(f"./Data_{gate}/Train_Test_Val"):
    os.mkdir(f"./Data_{gate}/Train_Test_Val")
train_test_val_split()

# 2. train
train(gate, learning_rate, device, batch_size, epoches, n_worker)

# 3. predict
path_raw = './Raw_Data/'
val(gate, path_raw, n_worker, device)