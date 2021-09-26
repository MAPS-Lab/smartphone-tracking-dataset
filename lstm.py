from model import lstm_activity
import torch.nn.functional as F
# import pytorch_forecasting
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch
from data_prepare_handheld import *
import os
import matplotlib.pyplot as plt
from visualize import *
from pyquaternion import Quaternion
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# loss_function = pytorch_forecasting.metrics.SMAPE()


def MAPELoss(output, target):
    return torch.mean(torch.abs((target - output) / target))


def train(epoch):

    train_loss = 0.0
    train_n = 0
    for batch_idx, (data_x, data_y) in enumerate(train_loader):

        if len(data_x) != batch_size:
            continue

        if torch.cuda.is_available():
            data = Variable(data_x.transpose(0, 1)).cuda()
            label = Variable(data_y).cuda()
        else:
            data = Variable(data_x.transpose(0, 1))
            label = Variable(data_y)

        # data = (data - data.min().data[0]) / (data.max().data[0] - data.min().data[0])

        model.zero_grad()
        model.hidden = model.init_hidden()

        result = model(data)
        # loss = loss_function(result, label)
        # loss = torch.add(F.mse_loss(result[:, :3], label[:, :3]), F.mse_loss(result[:, :3], label[:, :3]))
        loss = 100000 * \
            F.mse_loss(result[:, :3], label[:, :3]) + \
            F.mse_loss(result[:, 3:], label[:, 3:])
        # F.mse_loss(result[:, 3], label[:, 3]) + \
        # F.mse_loss(result[:, 4], label[:, 4]) * 200000 + \
        # F.mse_loss(result[:, 5], label[:, 5])
        # loss = F.mse_loss(result[:, :3], label[:, :3]) + \
        #     100000 * F.mse_loss(result[:, 3:], label[:, 3:])
        # loss = MAPELoss(result, label)

        # train_loss += loss.data[0]
        train_loss += loss.data
        train_n += batch_size
        if batch_idx % print_every == 0:
            # print('Train Epoch: {}, Data: {}, Loss: {:.4f}'.format(
            #     epoch, (batch_idx + 1) * len(data_x), loss.data[0] / batch_size * 1000))
            print('Train Epoch: {}, Data: {}, Loss: {:.4f}'.format(
                epoch, (batch_idx + 1) * len(data_x), loss.data / batch_size * 1000))

        loss.backward()
        optimizer.step()

        # nn.utils.clip_grad_norm(model.parameters(), clip)

    average_loss = train_loss / train_n * 1000
    print('====> Train Epoch: {} Average loss: {:.4f}'.format(epoch, average_loss))
    return average_loss


def test(epoch):

    test_loss = 0.0
    test_n = 0
    for batch_idx, (data_x, data_y) in enumerate(test_loader):

        if len(data_x) != batch_size:
            continue

        model.zero_grad()
        model.hidden = model.init_hidden()

        # data_x shape [400, 200, 6]
        if torch.cuda.is_available():
            data = Variable(data_x.transpose(0, 1)).cuda()
            label = Variable(data_y).cuda()
        else:
            data = Variable(data_x.transpose(0, 1))
            label = Variable(data_y)
        # data shape [200, 400, 6]

        # data = (data - data.min().data[0]) / (data.max().data[0] - data.min().data[0])

        result = model(data)
        # result shape [400, 3]

        # loss = F.mse_loss(result, label)
        loss = 100000 * \
            F.mse_loss(result[:, :3], label[:, :3]) + \
            F.mse_loss(result[:, 3:], label[:, 3:])
        # loss = MAPELoss(result, label)

        # test_loss += loss.data[0]
        test_loss += loss.data
        test_n += batch_size

    average_loss = test_loss / test_n * 1000
    print('====> Test Epoch: {} Average loss: {:.4f}'.format(epoch, average_loss))
    return average_loss


x_dim = 6
h_dim = 96
n_layers = 2
output_dim = data_testy.shape[1]
epoch_num = 200
learning_rate = 0.005

seed = 400
clip = 10
print_every = 10
save_every = 10

torch.manual_seed(seed)

train_data = TensorDataset(torch.from_numpy(data_trainx).type(torch.FloatTensor),
                           torch.from_numpy(data_trainy).type(torch.FloatTensor))
test_data = TensorDataset(torch.from_numpy(data_testx).type(torch.FloatTensor),
                          torch.from_numpy(data_testy).type(torch.FloatTensor))

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

model = lstm_activity(x_dim, h_dim, batch_size, n_layers, output_dim)

if torch.cuda.is_available():
    print("Using GPU")
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss_list = []
test_loss_list = []

best_test_loss = 1000000.0

for epoch in range(1, epoch_num + 1):

    train_loss = train(epoch)
    test_loss = test(epoch)

    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), 'saves/best.pth')

        # data_drawx_cuda = Variable(torch.from_numpy(
        #     data_drawx).type(torch.FloatTensor).transpose(0, 1)).cuda()
        # model.zero_grad()
        # model.hidden = model.init_hidden_pred(len_draw)
        # delta_trajectory = model(data_drawx_cuda).cpu().data.numpy()
        # accu_trajectory = np.zeros((len_draw + 1, 3))
        # for i in range(len_draw):
        #     accu_trajectory[i + 1] = accu_trajectory[i] + delta_trajectory[i]
        # figname = 'saves/best.jpg'
        # draw_trajectory(data_drawdy,
        #                 data_drawy,
        #                 delta_trajectory,
        #                 accu_trajectory,
        #                 figname)

        # data_drawx_cuda = Variable(torch.from_numpy(
        #     draw_x).type(torch.FloatTensor).transpose(0, 1)).cuda()
        # model.zero_grad()
        # model.hidden = model.init_hidden_pred(len_draw)
        # pred_y = model(data_drawx_cuda).cpu().data.numpy()
        # # y_revert(pred_y)
        # draw_y_traj = np.zeros((len_draw + 1, 3))
        # draw_q = []
        # draw_q.append(Quaternion(data_gt[start_seed][0:4]))
        # for i in range(len_draw):
        #     draw_q.append(draw_q[i] * Quaternion(pred_y[i][0:4]))
        #     draw_y_traj[i + 1] = draw_y_traj[i] + \
        #         draw_q[i].inverse.rotate(pred_y[i][4:7])
        # figname = 'saves/best.jpg'
        # draw_trajectory_6dof(draw_gt_qxyz,
        #                      pred_y,
        #                      draw_gt_traj,
        #                      draw_y_traj,
        #                      figname)

    # saving model
    if epoch % save_every == 0:
        fn = 'saves/lstm_state_dict_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), fn)
        print('Saved model to ' + fn)

        # # save graph for Global XYZ prediction
        # data_drawx_cuda = Variable(torch.from_numpy(
        #     data_drawx).type(torch.FloatTensor).transpose(0, 1)).cuda()
        # model.zero_grad()
        # model.hidden = model.init_hidden_pred(len_draw)
        # delta_trajectory = model(data_drawx_cuda).cpu().data.numpy()
        # accu_trajectory = np.zeros((len_draw + 1, 3))
        # for i in range(len_draw):
        #     accu_trajectory[i + 1] = accu_trajectory[i] + delta_trajectory[i]
        # figname = 'saves/traj_' + str(epoch) + '.jpg'
        # draw_trajectory(data_drawdy,
        #                 data_drawy,
        #                 delta_trajectory,
        #                 accu_trajectory,
        #                 figname)

        # # save graph for Global Q prediction
        # data_drawx_cuda = Variable(torch.from_numpy(
        #     data_drawx).type(torch.FloatTensor).transpose(0, 1)).cuda()
        # model.zero_grad()
        # model.hidden = model.init_hidden_pred(len_draw)
        # pred_y = model(data_drawx_cuda).cpu().data.numpy()
        # figname = 'saves/q_' + str(epoch) + '.jpg'
        # draw_trajectory_q(data_drawy,
        #                   pred_y,
        #                   figname)

        # save ==Multiple== graph for Local ==RXYZ== prediction
        for j, start_seed in enumerate(start_seeds):
            # ----Prepare X----
            data_drawx_cuda = Variable(torch.from_numpy(
                draw_xs[j]).type(torch.FloatTensor).transpose(0, 1)).cuda()
            # ----Prepare Model----
            model.zero_grad()
            model.hidden = model.init_hidden_pred(len_draw)
            # ----Prediction----
            pred_y = np.zeros((len_draw, 7))
            pred_y[:, 1:7] = model(data_drawx_cuda).cpu().data.numpy()
            pred_y[:, 0] = np.sqrt(1 - np.square(pred_y[:, 1]) -
                                   np.square(pred_y[:, 2]) -
                                   np.square(pred_y[:, 3]))
            # ----GT Replacement----
            # pred_y = draw_gt_qxyz.copy()
            # pred_y[:, :4] = draw_gt_qxyzs[j][:, :4].copy()
            # ----Build Trajectory----
            draw_y_traj = np.zeros((len_draw + 1, 3))
            draw_q = []
            draw_q.append(Quaternion(data_gt[start_seed][0:4]))
            for i in range(len_draw):
                draw_q.append(draw_q[i] * Quaternion(pred_y[i][0:4]))
                draw_y_traj[i + 1] = draw_y_traj[i] + \
                    draw_q[i].inverse.rotate(pred_y[i][4:7])
            # ----Draw Figures----
            figname = 'saves/traj_' + str(epoch) + str(start_seed) + '.jpg'
            draw_trajectory_6dof(draw_gt_qxyzs[j],
                                 pred_y,
                                 draw_gt_trajs[j],
                                 draw_y_traj,
                                 figname)

        # # save ==Multiple== graph for ==QX== prediction and ==YZ== Ground Truth
        # for j, start_seed in enumerate(start_seeds):
        #     data_drawx_cuda = Variable(torch.from_numpy(
        #         draw_xs[j]).type(torch.FloatTensor).transpose(0, 1)).cuda()
        #     model.zero_grad()
        #     model.hidden = model.init_hidden_pred(len_draw)
        #     pred_y = np.zeros((len_draw, 7))
        #     pred_y[:, 1:5] = model(data_drawx_cuda).cpu().data.numpy()
        #     # y_revert(pred_y)
        #     # pred_y = draw_gt_qxyz.copy()
        #     # pred_y[:, :4] = draw_gt_qxyzs[j][:, :4].copy()
        #     pred_y[:, 5:] = draw_gt_qxyzs[j][:, 5:].copy()
        #     pred_y[:, 0] = np.sqrt(
        #         1 - np.square(pred_y[:, 1]) - np.square(pred_y[:, 2]) - np.square(pred_y[:, 3]))
        #     draw_y_traj = np.zeros((len_draw + 1, 3))
        #     draw_q = []
        #     draw_q.append(Quaternion(data_gt[start_seed][0:4]))
        #     for i in range(len_draw):
        #         draw_q.append(draw_q[i] * Quaternion(pred_y[i][0:4]))
        #         draw_y_traj[i + 1] = draw_y_traj[i] + \
        #             draw_q[i].inverse.rotate(pred_y[i][4:7])
        #     figname = 'saves/traj_' + str(epoch) + str(start_seed) + '.jpg'
        #     draw_trajectory_6dof(draw_gt_qxyzs[j],
        #                          pred_y,
        #                          draw_gt_trajs[j],
        #                          draw_y_traj,
        #                          figname)

draw_loss(train_loss_list, test_loss_list)
np.savetxt('saves/train_loss.csv', train_loss_list, delimiter=',')
np.savetxt('saves/test_loss.csv', test_loss_list, delimiter=',')
