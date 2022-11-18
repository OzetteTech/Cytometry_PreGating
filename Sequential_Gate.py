from Data_Preprocessing import *
from Train import *
from Predict import *
import os

device = "mps"
# device = "cuda" if torch.cuda.is_available() else "cpu"
n_worker = 0
learning_rate = 1e-4
batch_size = 8
epoches = 20
n_worker = 0

###########################################################
# gate 1
###########################################################
y_axis = 'Ir191Di___191Ir_DNA1' # x axis in plot
x_axis = 'Event_length' # y axis in plot  
gate1 = 'gate1_ir'

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
val(x_axis, y_axis, gate1, path_raw, n_worker, device, seq = False)

###########################################################
# gate 2
###########################################################
y_axis = 'Ir193Di___193Ir_DNA2' # x axis in plot
x_axis = 'Y89Di___89Y_CD45' # y axis in plot  
gate2 = 'gate2_cd45'

# 1. preprocess data
if not os.path.exists("./Data"):
    os.mkdir(f"./Data_{gate2}")
    os.mkdir(f"./Data_{gate2}/Mask_Numpy")
    os.mkdir(f"./Data_{gate2}/Mask_PNG")
    os.mkdir(f"./Data_{gate2}/Raw_Numpy")
    os.mkdir(f"./Data_{gate2}/Raw_PNG")
process_table(x_axis, y_axis, gate2)

if not os.path.exists(f"./Data_{gate2}/Train_Test_Val"):
    os.mkdir(f"./Data_{gate2}/Train_Test_Val")
train_test_val_split(gate2)

# 2. train
train(gate2, learning_rate, device, batch_size, epoches, n_worker)

# 3. predict
path_raw = f'./Pred_Results_{gate1}/Pred_Results_'
val(x_axis, y_axis, gate2, path_raw, n_worker, device, seq = True, gate_pre = gate1)