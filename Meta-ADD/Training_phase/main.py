# Check out the code in work at https://www.kaggle.com/hsankesara/prototypical-net/
# Check out the blog at <COMING SOON>

import torch
import numpy as np  # linear algebra
from ProtoNetDrift import PrototypicalNet, train_step, test_step, load_weights
import torch.optim as optim
from preprocessing_200 import LoadDriftData

def main():

    Data_Vector_Length = 200
    DATA_FILE = 'drift-200-25'
    ModelSelect = 'RNN' # 'RNN', 'FCN', 'Seq2Seq'

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
    Drift_train_x, Drift_train_y = torch.tensor(train_drift_data).split(Data_Vector_Length, 1)
    Drift_test_x, Drift_test_y = torch.tensor(test_drift_data).split(Data_Vector_Length, 1)

    # Drift_data, Drift_label = Drift_dat_tensor.split(49, 1)

    print('Checking if GPU is available')
    use_gpu = torch.cuda.is_available()
    # use_gpu = False

    # Converting input to pytorch Tensor
    # trainx = torch.from_numpy(trainx).float()
    # testx = torch.from_numpy(testx).float()
    if use_gpu:
        Drift_train_x = Drift_train_x.cuda()
        Drift_test_x = Drift_test_x.cuda()
    # Priniting the data
    print(Drift_train_x.size(), Drift_test_x.size())
    # Set training iterations and display period
    num_episode = 40000
    frame_size = 100
    # trainx = trainx.permute(0, 3, 1, 2)
    # testx = testx.permute(0, 3, 1, 2)

    # Initializing prototypical net
    print('Initializing prototypical net')

    protonet = PrototypicalNet(use_gpu=use_gpu, Data_Vector_Length =Data_Vector_Length, ModelSelect=ModelSelect)
    # optimizer = optim.SGD(protonet.parameters(), lr=0.03, momentum=0.99)
    optimizer = optim.Adam(protonet.parameters(), lr=0.035)



    # Training loop
    frame_loss = 0
    frame_acc = 0
    for i in range(num_episode):
        # print(i)
        loss, acc, model_embeding = train_step(protonet, Drift_train_x, Drift_train_y, 5, 4, 5, optimizer)
        # print(acc)
        frame_loss += loss.data
        frame_acc += acc.data
        if((i+1) % frame_size == 0):
            print("Frame Number:", ((i+1) // frame_size), 'Frame Loss: ', frame_loss.data.cpu().numpy().tolist() /
                  frame_size, 'Frame Accuracy:', (frame_acc.data.cpu().numpy().tolist() * 100) / frame_size)
            frame_loss = 0
            frame_acc = 0
    PATH = '/home/tianyliu/Data/ConceptDrift/input/Model/'+ModelSelect+'/'+DATA_FILE+'/model_embeding.pkl'
    torch.save(model_embeding.state_dict(), PATH)

    # Test loop
    num_test_episode = 5000
    avg_loss = 0
    avg_acc = 0
    for _ in range(num_test_episode):
        loss, acc = test_step(protonet, Drift_test_x, Drift_test_y, 5, 4, 15)
        avg_loss += loss.data
        avg_acc += acc.data
        # print(acc)
    print('Avg Loss: ', avg_loss.data.cpu().numpy().tolist() / num_test_episode,
          'Avg Accuracy:', (avg_acc.data.cpu().numpy().tolist() * 100) / num_test_episode)

    # Using Pretrained Model
    # protonet = load_weights('./protonet.pt', protonet, use_gpu)


if __name__ == "__main__":
    main()
