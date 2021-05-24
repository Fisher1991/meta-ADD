import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from preprocessing_200 import LoadDriftData

class Net50(nn.Module):
    """
    Image2Vector CNN which takes image of dimension (28x28x3) and return column vector length 64
    """

    def __init__(self):
        super(Net50, self).__init__()
        self.fc1 = nn.Linear(50, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 30)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # x = torch.flatten(x, start_dim=1)
        return x

class Net100(nn.Module):
    """
    Image2Vector CNN which takes image of dimension (28x28x3) and return column vector length 64
    """

    def __init__(self):
        super(Net100, self).__init__()
        self.fc1 = nn.Linear(100, 250)
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, 250)
        self.fc4 = nn.Linear(250, 30)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # x = torch.flatten(x, start_dim=1)
        return x

class Net200(nn.Module):
    """
    Image2Vector CNN which takes image of dimension (28x28x3) and return column vector length 64
    """

    def __init__(self):
        super(Net200, self).__init__()
        self.fc1 = nn.Linear(200, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 300)
        self.fc4 = nn.Linear(300, 30)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # x = torch.flatten(x, start_dim=1)
        return x


def InputEmbeding(input, BASE_PATH, Data_Vector_Length):
    PATH = BASE_PATH+'/model_embeding.pkl'
    if Data_Vector_Length == 50:
        model_embeding = Net50()
    elif Data_Vector_Length == 100:
        model_embeding = Net100()
    else:
        model_embeding = Net200()
    model_embeding.load_state_dict(torch.load(PATH, map_location ='cpu'))
    return model_embeding(input)

def get_Input_Data(BASE_PATH):

    Test_Example_Label = 3.0

    Data_Vector_Length = 50
    DATA_FILE = 'drift-50-25'
    ModelSelect = 'FCN'

    DATA_PATH = BASE_PATH+'/Test_Example_Data.pt'

    # Reading the data
    print('Reading the data')
    all_data_frame = LoadDriftData(Data_Vector_Length, DATA_FILE)
    Drift_data_array = all_data_frame.values
    where_are_nan = np.isnan(Drift_data_array)
    where_are_inf = np.isinf(Drift_data_array)
    Drift_data_array[where_are_nan] = 0.0
    Drift_data_array[where_are_inf] = 0.0
    print(True in np.isnan(Drift_data_array))

    np.random.shuffle(Drift_data_array)
    Drift_data_tensor = torch.tensor(Drift_data_array)

    train_size = int(0.8 * len(Drift_data_tensor))
    test_size = int(len(Drift_data_tensor) - train_size)
    train_drift_data = Drift_data_array[0:train_size, :]
    test_drift_data = Drift_data_array[train_size:int(len(Drift_data_tensor)), :]
    # Drift_train_x, Drift_train_y = torch.tensor(train_drift_data).split(Data_Vector_Length, 1)
    Drift_test_x, Drift_test_y = torch.tensor(test_drift_data).split(Data_Vector_Length, 1)

    datay = Drift_test_y.numpy()
    data = Drift_test_x[(np.nonzero(datay == Test_Example_Label))[0]]

    perm = torch.randperm(data.shape[0])
    idx = perm[:100]
    S_cls = data[idx]
    # return S_cls.float(), Q_cls.float()
    Test_Example_Data = S_cls.float()
    return Test_Example_Data

def Detector(Query_x, BASE_PATH, Data_Vector_Length):
    DATA_PATH = BASE_PATH+'/centroid_matrix.pt'
    centroid_matrix = torch.load(DATA_PATH)

    Query_x = InputEmbeding(Query_x, BASE_PATH, Data_Vector_Length)
    m = Query_x.size(0)
    n = centroid_matrix.size(0)
    centroid_matrix = centroid_matrix.expand(
        m, centroid_matrix.size(0), centroid_matrix.size(1))

    Query_matrix = Query_x.expand(n, Query_x.size(0), Query_x.size(
        1)).transpose(0, 1)  # Expanding Query matrix "n" times
    Qx = torch.cosine_similarity(centroid_matrix.transpose(
        1, 2), Query_matrix.transpose(1, 2),dim=1)
    return Qx

def main(BASE_PATH, Data_Vector_Length):
    Test_Example_Data = get_Input_Data(BASE_PATH)
    Qx = Detector(Test_Example_Data, BASE_PATH, Data_Vector_Length)
    pred = torch.log_softmax(Qx, dim=-1)
    # DataType: float
    Label = (torch.argmax(pred, 1)).float()

    ACC = np.nonzero(Label.data == 3.0).shape[0] / 100

    print(ACC)

if __name__ == "__main__":
    # File address
    DATA_FILE = 'drift-50-25'

    # 50 OR 100 OR 200
    Data_Vector_Length = 50

    BASE_PATH = '/home/tianyliu/Data/ConceptDrift/input/Model/'+DATA_FILE
    main(BASE_PATH, Data_Vector_Length)