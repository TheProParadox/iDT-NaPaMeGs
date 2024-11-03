import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class MyDataset(Dataset):
    def __init__(self, csv_files, root_dir):
        self.dataframes = [pd.read_csv(os.path.join(root_dir, fname)) for fname in csv_files]

    def __len__(self):
        return sum([len(df) for df in self.dataframes])

    def __getitem__(self, idx):
        for df in self.dataframes:
            if idx < len(df):
                row = df.iloc[idx, :]
                input_data = torch.tensor(row[2:8], dtype=torch.float32)
                output_data = torch.tensor(row[8:], dtype=torch.float32)
                return input_data, output_data
            idx -= len(df)

# Model Definition
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = nn.Linear(6, 64)
        self.dense2 = nn.Linear(64, 128)
        self.dense3 = nn.Linear(128, 64)
        self.dense4 = nn.Linear(64, 4)

    def forward(self, x):
        x1 = F.relu(self.dense1(x))
        x2 = F.relu(self.dense2(x1))
        block1 = F.relu(self.dense3(x2))
        x3 = F.relu(x1 + block1)
        output = self.dense4(x3)
        return output

INPUT_DATA_DIR = "/scratch/napamegs/Ag_height_1to10/Ag_h1to10_cleaned/"
files = [f for f in os.listdir(INPUT_DATA_DIR) if os.path.isfile(os.path.join(INPUT_DATA_DIR, f))]

train_files, validation_files = train_test_split(files, train_size=0.7, random_state=42)

train_dataset = MyDataset(train_files, INPUT_DATA_DIR)
validation_dataset = MyDataset(validation_files, INPUT_DATA_DIR)

train_loader = DataLoader(train_dataset, batch_size=10000, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=10000, shuffle=False)

device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")

torch.save(model.state_dict(), 'fw_Ag.pth')



# df1 = pd.read_csv('/content/combined_csv.csv')
# df1 = df1[df1.Ts !=0]
# df1 = df1.drop_duplicates()

# new_df = df1.loc[df1.Ts > 20]
# #plt.stem(x_axis['lambda_val'], x_axis['Ts'])
# new_df

# dataset = new_df.values  #df to np_array
# X = dataset[:,2:8]
# y = dataset[:, 8:]
# x_train, x_test, y_train, y_test = train_test_split(X,
#                                                     y,
#                                                     test_size=0.4,
#                                                     random_state=50)

# x_train = np.asarray(x_train).astype(np.float32)
# y_train = np.asarray(y_train).astype(np.float32)
# x_test = np.asarray(x_test).astype(np.float32)
# y_test = np.asarray(y_test).astype(np.float32)
# y_train

# y_pred = new_model.predict(x_test)

# y_pred,y_test

# mse = tf.keras.losses.MeanSquaredError()

# ts_mse = mse(y_test[:,0], y_pred[:,0]).numpy()
# tp_mse = mse(y_test[:,1], y_pred[:,1]).numpy()
# rs_mse = mse(y_test[:,2], y_pred[:,2]).numpy()
# rp_mse = mse(y_test[:,3], y_pred[:,2]).numpy()
# print((ts_mse+tp_mse+rs_mse+rs_mse+rp_mse)/4)

# ts_mse,tp_mse,rs_mse,rs_mse

# mae = tf.keras.losses.MeanAbsoluteError()

# ts_mae = mae(y_test[:,0], y_pred[:,0]).numpy()
# tp_mae = mae(y_test[:,1], y_pred[:,1]).numpy()
# rs_mae = mae(y_test[:,2], y_pred[:,2]).numpy()
# rp_mae = mae(y_test[:,3], y_pred[:,3]).numpy()
# print((ts_mae+tp_mae+rs_mae+rp_mae)/4)

# ts_mae,tp_mae,rs_mae,rp_mae

# row =150
# param = 0 # 0 --> Ts, 1 --> Tp, 2 --> Rs, 3 --> Rp 
# pred = new_model.predict(x_test[:row,:])
# fig, ax = plt.subplots(figsize =(20, 10))

# plt.scatter(pred[:,param], range(row))
# plt.scatter(y_test[:row,param], range(row))
# plt.legend(['predicted', 'actual'], loc='upper left')
# plt.ylabel('row number')
# plt.xlabel('value')
# plt.grid()
# plt.title('T_s')
# plt.show()
# #x_test[:row,:]





# row = 30
# param = 1 # 0 --> Ts, 1 --> Tp, 2 --> Rs, 3 --> Rp 
# pred = new_model.predict(x_test[:row,:])
# fig, ax = plt.subplots(figsize =(20, 10))

# plt.scatter(pred[:,param], range(row))
# plt.scatter(y_test[:row,param], range(row))
# plt.legend(['predicted', 'actual'], loc='upper left')
# plt.ylabel('row number')
# plt.xlabel('value')
# plt.grid()
# plt.title('T_p')
# plt.show()
# #x_test[:row,:]

# row = 10
# param = 2# 0 --> Ts, 1 --> Tp, 2 --> Rs, 3 --> Rp 
# pred = new_model.predict(x_test[:row,:])
# fig, ax = plt.subplots(figsize =(20, 10))

# plt.scatter(pred[:,param], range(row))
# plt.scatter(y_test[:row,param], range(row))
# plt.legend(['predicted', 'actual'], loc='upper left')
# plt.ylabel('row number')
# plt.xlabel('value')
# plt.grid()
# plt.title('R_s')
# plt.show()
# #x_test[:row,:]

# row = 30
# param = 3 # 0 --> Ts, 1 --> Tp, 2 --> Rs, 3 --> Rp 
# pred = new_model.predict(x_test[:row,:])
# fig, ax = plt.subplots(figsize =(20, 10))

# plt.scatter(pred[:,param], range(row))
# plt.scatter(y_test[:row,param], range(row))
# plt.legend(['predicted', 'actual'], loc='upper left')
# plt.ylabel('row number')
# plt.xlabel('value')
# plt.title('R_p')
# plt.grid()
# plt.show()
# #x_test[:row,:]

# # Sprctrum 
# a  = new_df.iloc[:, 2:8] # Actual_input
# b = new_df.iloc[:, 8:] # Actual_output
# y_axis = a.iloc[:,-1]# wavelength for plottting purpose

# a_pred = new_model.predict(a) # prediction

# len(a_pred)

# fig, ax = plt.subplots(figsize =(25, 7))
# plt.plot(range(21598),a_pred[:21598,0])
# plt.plot(range(21598),b.iloc[:21598,0])
# plt.legend(['predicted', 'actual'], loc='upper left')

# plt.ylabel('Ts')
# plt.xlabel('wavelength')
# plt.title('Ts')
# plt.grid()
# plt.show()

# fig, ax = plt.subplots(figsize =(25, 7))
# plt.plot(range(21598),a_pred[:21598,1])
# plt.plot(range(21598),b.iloc[:21598,1])
# plt.grid()
# plt.legend(['predicted', 'actual'], loc='upper left')
# plt.ylabel('values')
# plt.xlabel('wavelength')
# plt.title('Tp')
# plt.show()

# fig, ax = plt.subplots(figsize =(25, 7))
# plt.plot(range(21598),a_pred[:21598,2])
# plt.plot(range(21598),np.round(b.iloc[:21598,2],2))
# plt.legend(['predicted', 'actual'], loc='upper left')
# plt.ylabel('values')
# plt.xlabel('wavelength')
# plt.title('Rs')

# plt.show()

# fig, ax = plt.subplots(figsize =(25, 7))
# plt.plot(range(21598),a_pred[:21598,3])
# plt.plot(range(21598),b.iloc[:21598,3])
# plt.legend(['predicted', 'actual'], loc='upper left')
# plt.ylabel('Rp')
# plt.xlabel('wavelength')
# plt.title('Rp')

# plt.show()

# data = pd.read_csv('/content/combined_csv.csv')
# data

