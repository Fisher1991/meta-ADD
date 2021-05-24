import numpy as np
import torch

from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector

import matplotlib.pyplot as plt
from torch import optim

from DriftDetector_CNN import Detector
#from DriftDetector_RNN import Detector


class HANG(BaseDriftDetector):

    def __init__(self, the_k=25, compare_window_len=1275):
        super().__init__()
        self.the_k = the_k
        self.correct_sample = 0
        self.feature = []
        self.accuracyList = []
        self.compare_window_len = compare_window_len
        self.sample_num = 0
        self.reset()

    def reset(self):
        """ reset

        Resets the change detector parameters.

        """
        super().reset()
        self.the_k = 25
        self.correct_sample = 0
        self.feature = []
        self.accuracyList = []
        self.compare_window_len = 1275
        self.sample_num = 0

    def add_element(self, prediction):

        if self.in_concept_change:
            self.reset()

        self.in_concept_change = False

        self.sample_num += 1

        if prediction == 0.0:
            self.correct_sample += 1

        if self.sample_num % self.compare_window_len != 0:
            if self.sample_num % self.the_k == 0:
                self.accuracyList.append(self.correct_sample / self.sample_num)

        else:
            self.accuracyList.append(self.correct_sample / self.sample_num)
            for i in range(len(self.accuracyList)-1):
                j = i+1
                if self.accuracyList[i] == 0.0:
                    self.feature.append(0.0)
                else:
                    self.feature.append((self.accuracyList[j] - self.accuracyList[i])/self.accuracyList[i])
            # print("self.feature: ", self.feature)
            # prediction
            BASE_PATH = 'input/model/FCN/drift-50-1/'
            Data_Vector_Length = self.compare_window_len/self.the_k - 1

            Qx, model_embeding = Detector(torch.tensor([self.feature]), BASE_PATH, Data_Vector_Length)

            #Qx = Detector(torch.tensor([self.feature]), BASE_PATH, Data_Vector_Length, 'RNN')

            # probability = torch.softmax(Qx, dim=-1)
            # print("softmax: ", probability)
            # plt.plot(self.accuracyList)
            # plt.show()
            pred = torch.log_softmax(Qx, dim=-1)
            #print("pred: ", pred)
            # entropy = torch.sum(probability*pred)
            result = float((torch.argmax(pred, 1)[0]))
            #print("entropy: ", entropy)
			#### active learning
            # if entropy<-1.22:
            #     # plt.plot(self.accuracyList)
            #     # plt.show()
            #     print("probability: ", probability)
            #     result_human = torch.tensor([3]).long()
            #
            #     model = Embeding(BASE_PATH, Data_Vector_Length)
            #
            #     optimizer = optim.Adam(model.parameters(), lr=0.001)
            #     for i in range(100):
            #         optimizer.zero_grad()
            #         Query_x = model(torch.tensor([self.feature]))
            #         #print("Query_x: ", Query_x)
            #         DATA_PATH = BASE_PATH + '/centroid_matrix.pt'
            #         centroid_matrix = torch.load(DATA_PATH)
            #         m = Query_x.size(0)
            #         n = centroid_matrix.size(0)
            #         centroid_matrix = centroid_matrix.expand(
            #             m, centroid_matrix.size(0), centroid_matrix.size(1))
            #         # centroid_matrix = centroid_matrix.expand(
            #         #     m, centroid_matrix.size(0), centroid_matrix.size(1))
            #         Query_matrix = Query_x.expand(n, Query_x.size(0), Query_x.size(
            #             1)).transpose(0, 1)  # Expanding Query matrix "n" times
            #         Qx = torch.cosine_similarity(centroid_matrix.transpose(
            #             1, 2), Query_matrix.transpose(1, 2), dim=1)
            #         if i==99:
            #             print("after-trained: ", torch.softmax(Qx, dim=-1))
            #         pred = torch.log_softmax(Qx, dim=-1)
            #         loss = torch.nn.NLLLoss()(pred, result_human)
            #         loss.backward()
            #         # optimizer parameters
            #         optimizer.step()
            #     PATH = 'input/model/drift-50-1/model_embeding.pkl'
            #     torch.save(model.state_dict(), PATH)
            if result == 0 or result == 1 or result == 2:
                self.in_concept_change = True
            self.accuracyList = []
            self.feature = []