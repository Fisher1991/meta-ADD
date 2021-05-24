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


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.n_hidden = 64
        feature_len = 1
        self.encoder = nn.GRU(input_size=feature_len, hidden_size=self.n_hidden, num_layers=1, dropout=0.1)
        self.decoder = nn.GRU(input_size=feature_len, hidden_size=self.n_hidden, num_layers=1, dropout=0.1)
        self.fc = nn.Linear(self.n_hidden, 30)
        # self.gpu = torch.cuda.is_available()

    def forward(self, src_batch: torch.LongTensor):

        # src_batch: [seq_len, batch_size, feature_len] = [200, 20, 1]
        src_batch = src_batch.transpose(0, 1).unsqueeze(dim=2)
        seq_len, batch_size, feature_len = src_batch.shape
        # if self.gpu:
        #     src_batch = src_batch.cuda()
        enc_outputs, hidden_cell = self.encoder(src_batch, torch.zeros(1, batch_size, self.n_hidden))
        decoder_input = torch.zeros(1, batch_size, 1) # initialize the initial state
        # if(self.gpu):
        #     decoder_input = decoder_input.cuda()
        #     hidden_cell = hidden_cell.cuda()
        decoder_output, hidden_cell = self.decoder(decoder_input, hidden_cell)
        # if(self.gpu):
        #     hidden_cell = hidden_cell.cuda()
        outputs = self.fc(hidden_cell).squeeze(0)
        return outputs


def InputEmbeding(input, BASE_PATH, Data_Vector_Length, ModelSelect):
    PATH = BASE_PATH + '/model_embeding.pkl'

    if ModelSelect == 'RNN':
        model_embeding = RNN()
    elif ModelSelect == 'FCN':
        if Data_Vector_Length == 50:
            model_embeding = Net50()
        elif Data_Vector_Length == 100:
            model_embeding = Net100()
        else:
            model_embeding = Net200()

    model_embeding.load_state_dict(torch.load(PATH))
    return model_embeding(input)

def main(DATA_FILE, BASE_PATH, Data_Vector_Length, ModelSelect):

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
    train_drift_data = Drift_data_array[0:train_size, :]
    Drift_train_x, Drift_train_y = torch.tensor(train_drift_data).split(Data_Vector_Length, 1)
    Qy = np.unique(Drift_train_y)

    Test_Example_Label = 1.0
    Test_Example_Data = random_sample_cls(Drift_train_x, Drift_train_y, 1, Test_Example_Label)

    DATA_PATH = BASE_PATH + '/Test_Example_Data.pt'
    torch.save(Test_Example_Data, DATA_PATH)

    Qx = centroid(Drift_train_x, Drift_train_y, 20, 4, np.unique(Drift_train_y), BASE_PATH, Data_Vector_Length, ModelSelect)
    pred = torch.log_softmax(Qx, dim=-1)
    Label = float((torch.argmax(pred, 1)[0]))
    print(Label)
    return Label

def centroid(datax, datay, Ns, Nc, total_classes, BASE_PATH, Data_Vector_Length, ModelSelect):

    k = total_classes.shape[0]
    K = np.random.choice(total_classes, Nc, replace=False)
    Query_x = torch.Tensor()
    Query_y = []
    Query_y_count = []
    centroid_per_class = {}
    class_label = {}
    for cls in total_classes:
        S_cls = random_sample_cls(datax, datay, Ns, cls)
        centroid_per_class[cls] = torch.sum(InputEmbeding(S_cls.float(), BASE_PATH, Data_Vector_Length, ModelSelect), 0).unsqueeze(1).transpose(0, 1) / Ns
    centroid_matrix = torch.Tensor()
    for label in total_classes:
        centroid_matrix = torch.cat(
            (centroid_matrix, centroid_per_class[label]))

    DATA_PATH = BASE_PATH + '/centroid_matrix.pt'
    torch.save(centroid_matrix, DATA_PATH)

    DATA_PATH = BASE_PATH + '/Test_Example_Data.pt'
    Test_Example_Data = torch.load(DATA_PATH)

    Query_x = InputEmbeding(Test_Example_Data, BASE_PATH, Data_Vector_Length, ModelSelect)
    m = Query_x.size(0)
    n = centroid_matrix.size(0)
    centroid_matrix = centroid_matrix.expand(
        m, centroid_matrix.size(0), centroid_matrix.size(1))

    Query_matrix = Query_x.expand(n, Query_x.size(0), Query_x.size(
        1)).transpose(0, 1)  # Expanding Query matrix "n" times
    Qx = torch.cosine_similarity(centroid_matrix.transpose(
        1, 2), Query_matrix.transpose(1, 2),dim=1)
    return Qx


def random_sample_cls(datax, datay, Ns, cls):
    """
    Randomly samples Ns examples as support set and Nq as Query set
    """
    datay = datay.numpy()
    data = datax[(np.nonzero(datay == cls))[0]]

    perm = torch.randperm(data.shape[0])
    idx = perm[:Ns]
    S_cls = data[idx]
    # S_cls = S_cls.cuda()
    return S_cls.float()



if __name__ == "__main__":
    # File address
    DATA_FILE = 'drift-50-1'
    Data_Vector_Length = 50
    ModelSelect = 'RNN' # 'RNN', 'FCN', 'Seq2Seq'

    BASE_PATH = '/home/tianyliu/Data/ConceptDrift/input/Model/'+ ModelSelect+'/'+DATA_FILE

    main(DATA_FILE, BASE_PATH, Data_Vector_Length, ModelSelect)