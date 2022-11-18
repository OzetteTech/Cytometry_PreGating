# Cytometry_PreGating
With the advancement in cytometric technology, there are a large number of cytometric data analysis tools available, but the problem of biological variance across subjects still have not been addressed in other softwares. This pipeline provides an effective way to address this problem and clean the cytometric data in excluding doublets and debris. The pipeline provides two detecting modes, which is single gate prediction and two gates sequential prediction.

## Usage
1. Put the raw tabular data into the project folder with name 'Raw_Data'
2. If running the single gate prediction pipeline, run the Single_Gate.py with arguments of two measurments and the gate name in the csv file. If running the sequential gate prediction, run the Sequential_Gate.py with arguements of two sets of measurements and gate respectively. 
3. The training process will generate model as well as predictions for validation set. If you need to predict new data, you can formulate the file name of the data into a csv file and replace that in the ./Data_{gate}/Train_Test_Val/Val.csv, then run the Predict_New.py file with the measurement and gate name. 
