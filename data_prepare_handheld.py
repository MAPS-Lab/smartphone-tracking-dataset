import numpy as np
# handheld_regular
# print("input traning data")
# #data1
# data_x=np.loadtxt("../../handheld/data1/stride_x1.csv",delimiter=",")
# data_y=np.loadtxt("../../handheld/data1/stride_y1.csv",delimiter=",")
# len_x=len(data_x)
# data_x=data_x.reshape(len_x, 6, 200)
# data_x=np.transpose(data_x,(0,2,1))
# #len_x=int(len_x/batch_size)*batch_size
# data_trainx=np.copy(data_x[:len_x,:,:])
# data_trainy=np.copy(data_y[:len_x])
#
# file_address='../../handheld/data1/'
# for index in range(2,8):
# 	datax_name=file_address+'stride_x'+str(index)+'.csv'
# 	datay_name=file_address+'stride_y'+str(index)+'.csv'
# 	data_x=np.loadtxt(datax_name, delimiter=",")
# 	data_y=np.loadtxt(datay_name, delimiter=",")
# 	len_x=len(data_x)
# 	data_x=data_x.reshape(len_x, 6, 200)
# 	data_x=np.transpose(data_x,(0,2,1))
# 	#len_x=int(len_x/batch_size)*batch_size
# 	data_trainx=np.concatenate((data_trainx, data_x[:len_x,:,:]), axis=0)
# 	data_trainy=np.concatenate((data_trainy, data_y[:len_x]), axis=0)
#
# file_address='../../handheld/data2/'
# for index in range(2,5):
# 	datax_name=file_address+'stride_x'+str(index)+'.csv'
# 	datay_name=file_address+'stride_y'+str(index)+'.csv'
# 	data_x=np.loadtxt(datax_name, delimiter=",")
# 	data_y=np.loadtxt(datay_name, delimiter=",")
# 	len_x=len(data_x)
# 	data_x=data_x.reshape(len_x, 6, 200)
# 	data_x=np.transpose(data_x,(0,2,1))
# 	#len_x=int(len_x/batch_size)*batch_size
# 	data_trainx=np.concatenate((data_trainx, data_x[:len_x,:,:]), axis=0)
# 	data_trainy=np.concatenate((data_trainy, data_y[:len_x]), axis=0)
#
# file_address='../../handheld/data3/'
# for index in range(1,5):
# 	datax_name=file_address+'stride_x'+str(index)+'.csv'
# 	datay_name=file_address+'stride_y'+str(index)+'.csv'
# 	data_x=np.loadtxt(datax_name, delimiter=",")
# 	data_y=np.loadtxt(datay_name, delimiter=",")
# 	len_x=len(data_x)
# 	data_x=data_x.reshape(len_x, 6, 200)
# 	data_x=np.transpose(data_x,(0,2,1))
# 	#len_x=int(len_x/batch_size)*batch_size
# 	data_trainx=np.concatenate((data_trainx, data_x[:len_x,:,:]), axis=0)
# 	data_trainy=np.concatenate((data_trainy, data_y[:len_x]), axis=0)
#
#
# #test data
# print("input test data")
# data_x=np.loadtxt("../../handheld/data3/stride_x5.csv",delimiter=",")
# data_y=np.loadtxt("../../handheld/data3/stride_y5.csv",delimiter=",")
# len_x=len(data_x)
# data_x=data_x.reshape(len_x, 6, 200)
# data_x=np.transpose(data_x,(0,2,1))
#
# data_testx=np.copy(data_x)
# data_testy=np.copy(data_y)


def y_adjust(y):
    # RW, RX, RY, RZ, X, Y, Z
    y[0] = (1 - y[0]) * 1000
    y[1] = y[1] * 1000
    y[2] = y[2] * 1000
    y[3] = y[3] * 1000
    y[5] = y[5] * 10


def y_revert(y):
    # RW, RX, RY, RZ, X, Y, Z
    y[0] = 1 - y[0] / 1000
    y[1] = y[1] / 1000
    y[2] = y[2] / 1000
    y[3] = y[3] / 1000
    y[5] = y[5] / 10


def x_adjust(x):
    x[:, :, :3] = x[:, :, :3] * 100
    x[:, :, 3:5] = x[:, :, 3:5] * 10
    x[:, :, 5] = (10 - x[:, :, 5]) * 100


len_sequence = 50
sample_step = 10
batch_size = 400

print("input samsung tracking data")
data_x = np.loadtxt(
    "../../handheld/samsung/x_50_10_3.csv", delimiter=",")
data_y = np.loadtxt(
    "../../handheld/samsung/y_50_10_3.csv", delimiter=",")
data_gt_all = np.loadtxt(
    "../../handheld/samsung/gt_50_10_3.csv", delimiter=",")
# data_y = data_y[:, 4:7]  # Training for Translation
# data_y = data_y[:, :4]  # Training for Rotation
# len_x = int(len(data_x) / batch_size) * batch_size
# split = int(len(data_x) / batch_size * 0.8) * batch_size
# data_x = data_x.reshape(len_x, 6, 200)
# data_x = np.transpose(data_x, (0, 2, 1))
data_x = data_x.reshape(-1, len_sequence, 6)
spliter = 65000
data_trainx = np.copy(data_x[:spliter])
data_trainy = np.copy(data_y[:spliter])
data_testx = np.copy(data_x[spliter:])
data_testy = np.copy(data_y[spliter:])
# data_gt = np.copy(data_gt_all[20000:])
data_gt = np.copy(data_gt_all[spliter:])

# data_x = np.loadtxt(
#     "../../handheld/samsung/x_50_10_2_int.csv", delimiter=",")
# data_y = np.loadtxt(
#     "../../handheld/samsung/len50_stride10_batch2_y.csv", delimiter=",")
# data_gt_all = np.loadtxt(
#     "../../handheld/samsung/len50_stride10_batch2_gt.csv", delimiter=",")
# # len_x = int(len(data_x) / batch_size) * batch_size
# # split = int(len(data_x) / batch_size * 0.8) * batch_size
# data_x = data_x.reshape(-1, len_sequence, 12)
# data_trainx = np.concatenate((data_trainx, data_x[:20000, :, :]), axis=0)
# data_trainy = np.concatenate((data_trainy, data_y[:20000]), axis=0)
# data_testx = np.concatenate((data_testx, data_x[20000:, :, :]), axis=0)
# data_testy = np.concatenate((data_testy, data_y[20000:]), axis=0)
# # data_gt = np.concatenate((data_gt, data_gt[20000:]), axis=0)
# data_gt = np.concatenate((data_gt, data_gt[20000:]), axis=0)

# data_trainy = data_trainy[:, 0:4]
# data_testy = data_testy[:, 0:4]

# x_adjust(data_trainx)
# x_adjust(data_testx)

# # Trajectory Draw for Global XYZ prediction
# len_draw = 20
# start_seed = 0
# traj_interval = 20
# data_drawx = np.zeros((len_draw, 200, 6))
# data_drawdy = np.zeros((len_draw, 3))
# data_drawy = np.zeros((len_draw + 1, 3))
# for i_draw, i_data in enumerate(range(start_seed, start_seed + len_draw * traj_interval, traj_interval)):
#     data_drawx[i_draw] = data_testx[i_data]
#     data_drawdy[i_draw] = data_testy[i_data]
#     data_drawy[i_draw + 1] = data_drawy[i_draw] + data_testy[i_data]

# # Trajectory Draw for Global Q prediction
# len_draw = 30
# start_seed = 3000
# traj_interval = 20
# data_drawx = np.zeros((len_draw, len_sequence, 6))
# data_drawy = np.zeros((len_draw, 4))
# for i_draw, i_data in enumerate(range(start_seed, start_seed + len_draw * traj_interval, traj_interval)):
#     data_drawx[i_draw] = data_testx[i_data]
#     data_drawy[i_draw] = data_testy[i_data]

# # Trajectory Draw for Local QXYZ prediction
# len_draw = 30
# # start_seed = 5000
# start_seeds = [1000, 2000, 3000, 4000, 5000, 6000]
# traj_interval = len_sequence // sample_step
# draw_gt_trajs = []
# draw_gt_qxyzs = []
# draw_xs = []
# for start_seed in start_seeds:
#     draw_gt_traj = np.zeros((len_draw + 1, 3))  # GT from gt.csv file
#     draw_gt_qxyz = np.zeros((len_draw, 7))  # GT from y.csv file
#     draw_x = np.zeros((len_draw, len_sequence, 6))
#     # draw trajectory from gt
#     for i in range(len_draw):
#         i_data = start_seed + i * traj_interval
#         draw_gt_qxyz[i] = data_trainy[i_data].copy()
#         # draw_gt_qxyz[i] = data_testy[i_data].copy()
#         draw_gt_traj[i + 1] = data_gt[i_data + traj_interval][4:7] - \
#             data_gt[start_seed][4:7]
#         draw_x[i] = data_trainx[i_data]
#         # draw_x[i] = data_testx[i_data]
#     draw_gt_trajs.append(draw_gt_traj)
#     draw_gt_qxyzs.append(draw_gt_qxyz)
#     draw_xs.append(draw_x)

# Trajectory Draw for Local QXYZ prediction
len_draw = 30
# start_seed = 5000
start_seeds = [1000, 2000, 3000, 4000, 5000]
traj_interval = len_sequence // sample_step
draw_gt_trajs = []
draw_gt_qxyzs = []
draw_xs = []
for start_seed in start_seeds:
    draw_gt_traj = np.zeros((len_draw + 1, 3))  # GT from gt.csv file
    draw_gt_qxyz = np.zeros((len_draw, 7))  # GT from y.csv file
    draw_x = np.zeros((len_draw, len_sequence, 6))
    # draw trajectory from gt
    for i in range(len_draw):
        i_data = start_seed + i * traj_interval
        draw_gt_qxyz[i] = data_testy[i_data].copy()
        # draw_gt_qxyz[i] = data_testy[i_data].copy()
        draw_gt_traj[i + 1] = data_gt[i_data + traj_interval][4:7] - \
            data_gt[start_seed][4:7]
        draw_x[i] = data_testx[i_data]
        # draw_x[i] = data_testx[i_data]
    draw_gt_trajs.append(draw_gt_traj)
    draw_gt_qxyzs.append(draw_gt_qxyz)
    draw_xs.append(draw_x)

data_trainy = data_trainy[:, 1:7]
data_testy = data_testy[:, 1:7]

# y_adjust(data_trainy)
# y_adjust(data_testy)
