Requirements:

cycler==0.10.0
joblib==1.0.0
kiwisolver==1.3.1
matplotlib==3.3.3
numpy==1.19.4
liac-arff==2.5.0
pandas==1.1.5
Pillow==8.0.1
pyparsing==2.4.7
python-dateutil==2.8.1
pytz==2020.5
scikit-learn==0.24.0
scipy==1.5.4
six==1.15.0
sklearn==0.0
threadpoolctl==2.1.0
torch==1.3.1
torchvision==0.4.2
tqdm==4.54.1




Training phase:
datagerneration.py: generate training data (input\data)
preprocessing_200.py: preprocessing training data to extract meta-features
ProtoNetDrift.py: implementation of prototypical neural network
DriftCenterGenerator.py: generate the prototype for different concept drift classes
ClassAcc.py: verify the classification accuracy of different concept drift classes
DriftDetector.py: verify the effectiveness of the learned pretrain model



Detecting phase:

Datasets: save all experimental data. The experimental data are all .arff files and can be downloaded from https://moa.cms.waikato.ac.nz/datasets/
data_handler.py: preprocess all experimental data
external_ddm\hang.py: implemente the details of detecting drifts by our method Meta-ADD. In addition, this .py file will call the DriftDetector_CNN.py and DriftDetector_RNN.py.
input\model: save meta-detector
landmarkwindow.py: the main program
stream_learning_lib.py: simulate the prequential prediction errors of a GaussianNB model.