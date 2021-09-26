from model import lstm_activity
import torch.nn.functional as F
# import pytorch_forecasting
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch
import os
import matplotlib.pyplot as plt
from visualize import *
from pyquaternion import Quaternion
import numpy as np
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_id = 110

x_dim = 6
h_dim = 96
n_layers = 2
output_dim = 6
epoch_num = 200
learning_rate = 0.005
batch_size = 400
len_sequence = 50

seed = 400
clip = 10
print_every = 10
save_every = 10

torch.manual_seed(seed)

model = lstm_activity(x_dim, h_dim, batch_size, n_layers, output_dim)

model_path = './saves/training_23/lstm_state_dict_' + str(model_id) + '.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

ate = np.zeros(188)

if torch.cuda.is_available():
    print("Using GPU")
    model.cuda()

for i in range(188):

    data_x = np.loadtxt('./eval_data/eval_' + str(i) + '_x.csv', delimiter=",")
    data_y = np.loadtxt('./eval_data/eval_' + str(i) + '_y.csv', delimiter=",")
    data_gt = np.loadtxt('./eval_data/eval_' + str(i) + '_gt.csv', delimiter=",")

    len_draw = data_x.shape[0]
    data_x = data_x.reshape(-1, len_sequence, 6)

    data_x_cuda = Variable(torch.from_numpy(data_x).type(torch.FloatTensor).transpose(0, 1)).cuda()
    model.zero_grad()
    model.hidden = model.init_hidden_pred(len_draw)

    pred_y = np.zeros((len_draw, 7))
    pred_y[:, 1:7] = model(data_x_cuda).cpu().data.numpy()
    pred_y[:, 0] = np.sqrt(1 - np.square(pred_y[:, 1]) -
                           np.square(pred_y[:, 2]) -
                           np.square(pred_y[:, 3]))

    traj_y = np.zeros((len_draw + 1, 3))
    traj_gt = np.zeros((len_draw + 1, 3))
    traj_q = []
    traj_q.append(Quaternion(data_gt[0][0:4]))
    for j in range(len_draw):
        traj_q.append(traj_q[j] * Quaternion(pred_y[j][0:4]))
        traj_y[j + 1] = traj_y[j] + traj_q[j].conjugate.rotate(pred_y[j][4:7])
        traj_gt[j + 1] = data_gt[j + 1][4:7] - data_gt[0][4:7]

    traj_diff = traj_y - traj_gt
    traj_norm = np.linalg.norm(traj_diff, ord=2, axis=1, keepdims=False)
    ate[i] = traj_norm.mean()

    # figname = 'eval_results/' + str(model_id) + '/paper/traj_' + str(i) + '.jpg'
    figname = 'eval_results/paper/traj_' + str(i) + '.pdf'
    draw_trajectory_paper(data_y,
                          pred_y,
                          traj_gt,
                          traj_y,
                          traj_norm,
                          figname)

print("ATE list", ate)
draw_ate_hist(ate, 'ate_hist_' + str(model_id) + '.pdf')
print("ATE mean", ate.mean())
print("ATE max", ate.max())
print("ATE std", ate.std())
