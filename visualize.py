import matplotlib.pyplot as plt
import numpy as np


def draw_trajectory(gt_delta, gt_trajectory, pred_delta, pred_trajectory, figname):
    plt.rcParams['figure.figsize'] = [15, 15]
    fig, axs = plt.subplots(3, 3)
    axs[0, 0].plot(gt_delta[:, 0])
    axs[0, 0].plot(pred_delta[:, 0])
    axs[0, 0].set_title('dX-t')
    axs[0, 1].plot(gt_delta[:, 1])
    axs[0, 1].plot(pred_delta[:, 1])
    axs[0, 1].set_title('dY-t')
    axs[0, 2].plot(gt_delta[:, 2])
    axs[0, 2].plot(pred_delta[:, 2])
    axs[0, 2].set_title('dZ-t')
    axs[1, 0].plot(gt_trajectory[:, 0])
    axs[1, 0].plot(pred_trajectory[:, 0])
    axs[1, 0].set_title('X-t')
    axs[1, 1].plot(gt_trajectory[:, 1])
    axs[1, 1].plot(pred_trajectory[:, 1])
    axs[1, 1].set_title('Y-t')
    axs[1, 2].plot(gt_trajectory[:, 2])
    axs[1, 2].plot(pred_trajectory[:, 2])
    axs[1, 2].set_title('Z-t')
    axs[2, 0].plot(gt_trajectory[:, 0], gt_trajectory[:, 1])
    axs[2, 0].plot(pred_trajectory[:, 0], pred_trajectory[:, 1])
    axs[2, 0].set_title('X-Y')
    axs[2, 1].plot(gt_trajectory[:, 1], gt_trajectory[:, 2])
    axs[2, 1].plot(pred_trajectory[:, 1], pred_trajectory[:, 2])
    axs[2, 1].set_title('Y-Z')
    axs[2, 2].plot(gt_trajectory[:, 0], gt_trajectory[:, 2])
    axs[2, 2].plot(pred_trajectory[:, 0], pred_trajectory[:, 2])
    axs[2, 2].set_title('X-Z')
    fig.savefig(figname)


def draw_trajectory_6dof(y_true, y_pred, traj_true, traj_pred, traj_norm, figname):
    plt.rcParams['figure.figsize'] = [20, 15]
    fig, axs = plt.subplots(3, 4)
    axs[0, 0].plot(y_true[:, 0], label='GT')
    axs[0, 0].plot(y_pred[:, 0], label='Pred')
    axs[0, 0].set_title('Rotation(Quaternion): RW-t')
    axs[0, 1].plot(y_true[:, 1], label='GT')
    axs[0, 1].plot(y_pred[:, 1], label='Pred')
    axs[0, 1].set_title('Rotation(Quaternion): RX-t')
    axs[0, 2].plot(y_true[:, 2], label='GT')
    axs[0, 2].plot(y_pred[:, 2], label='Pred')
    axs[0, 2].set_title('Rotation(Quaternion): RY-t')
    axs[0, 3].plot(y_true[:, 3], label='GT')
    axs[0, 3].plot(y_pred[:, 3], label='Pred')
    axs[0, 3].set_title('Rotation(Quaternion): RZ-t')
    axs[1, 0].plot(y_true[:, 4], label='GT')
    axs[1, 0].plot(y_pred[:, 4], label='Pred')
    axs[1, 0].set_title('Translation(Cartesian): X-t')
    axs[1, 1].plot(y_true[:, 5], label='GT')
    axs[1, 1].plot(y_pred[:, 5], label='Pred')
    axs[1, 1].set_title('Translation(Cartesian): Y-t')
    axs[1, 2].plot(y_true[:, 6], label='GT')
    axs[1, 2].plot(y_pred[:, 6], label='Pred')
    axs[1, 2].set_title('Translation(Cartesian): Z-t')
    x_tmp = (traj_true[:, 0].min() + traj_true[:, 0].max()) / 2
    y_tmp = (traj_true[:, 1].min() + traj_true[:, 1].max()) / 2
    z_tmp = (traj_true[:, 2].min() + traj_true[:, 2].max()) / 2
    axs[2, 0].plot(traj_true[:, 0], traj_true[:, 1], label='GT')
    axs[2, 0].plot(traj_pred[:, 0], traj_pred[:, 1], label='Pred')
    axs[2, 0].set_xlim([x_tmp - 250, x_tmp + 250])
    axs[2, 0].set_ylim([y_tmp - 250, y_tmp + 250])
    axs[2, 0].set_title('Trajectory: X-Y, Top view')
    axs[2, 1].plot(traj_true[:, 1], traj_true[:, 2], label='GT')
    axs[2, 1].plot(traj_pred[:, 1], traj_pred[:, 2], label='Pred')
    axs[2, 1].set_xlim([y_tmp - 250, y_tmp + 250])
    axs[2, 1].set_ylim([z_tmp - 250, z_tmp + 250])
    axs[2, 1].set_title('Trajectory: Y-Z, Right view')
    axs[2, 2].plot(traj_true[:, 0], traj_true[:, 2], label='GT')
    axs[2, 2].plot(traj_pred[:, 0], traj_pred[:, 2], label='Pred')
    axs[2, 2].set_xlim([x_tmp - 250, x_tmp + 250])
    axs[2, 2].set_ylim([z_tmp - 250, z_tmp + 250])
    axs[2, 2].set_title('Trajectory: X-Z, Front view')
    axs[2, 3].plot(traj_norm, label='Error')
    axs[2, 3].set_title('Trajectory Error: Euclidean Distance (mm)')
    for i in axs.flat:
        i.legend(loc=0)
    fig.savefig(figname)
    plt.close('all')


def draw_trajectory_paper(y_true, y_pred, traj_true, traj_pred, traj_norm, figname):
    plt.rcParams['figure.figsize'] = [20, 5]
    fig, axs = plt.subplots(1, 4)
    lw = 2
    plt.rcParams['axes.facecolor'] = 'w'
    plt.rcParams['savefig.facecolor'] = 'w'
    plt.rcParams['grid.color'] = '#C0C0C0'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.size'] = 12
    x_tmp = (traj_true[:, 0].min() + traj_true[:, 0].max()) / 2
    y_tmp = (traj_true[:, 1].min() + traj_true[:, 1].max()) / 2
    z_tmp = (traj_true[:, 2].min() + traj_true[:, 2].max()) / 2
    axs[0].plot(traj_true[:, 0], traj_true[:, 1], label='GT', linewidth=lw)
    axs[0].plot(traj_pred[:, 0], traj_pred[:, 1], label='Pred', linewidth=lw)
    axs[0].set_xlim([x_tmp - 250, x_tmp + 250])
    axs[0].set_ylim([y_tmp - 250, y_tmp + 250])
    axs[0].set_xlabel('x(mm)')
    axs[0].set_ylabel('y(mm)')
    axs[0].set_title('Trajectory: X-Y, Top view')
    axs[1].plot(traj_true[:, 1], traj_true[:, 2], label='GT', linewidth=lw)
    axs[1].plot(traj_pred[:, 1], traj_pred[:, 2], label='Pred', linewidth=lw)
    axs[1].set_xlim([y_tmp - 250, y_tmp + 250])
    axs[1].set_ylim([z_tmp - 250, z_tmp + 250])
    axs[1].set_xlabel('y(mm)')
    axs[1].set_ylabel('z(mm)')
    axs[1].set_title('Trajectory: Y-Z, Right view')
    axs[2].plot(traj_true[:, 0], traj_true[:, 2], label='GT', linewidth=lw)
    axs[2].plot(traj_pred[:, 0], traj_pred[:, 2], label='Pred', linewidth=lw)
    axs[2].set_xlim([x_tmp - 250, x_tmp + 250])
    axs[2].set_ylim([z_tmp - 250, z_tmp + 250])
    axs[2].set_xlabel('x(mm)')
    axs[2].set_ylabel('z(mm)')
    axs[2].set_title('Trajectory: X-Z, Front view')
    axs[3].plot(traj_norm, label='Error', linewidth=lw)
    axs[3].set_xlabel('Trajectory Length(mm)')
    axs[3].set_ylabel('Reconstruction Error(mm)')
    axs[3].set_title('Trajectory Error: Euclidean Distance (mm)')
    for i in axs.flat:
        i.legend(loc=0)
    fig.savefig(figname, bbox_inches='tight')
    plt.close('all')


def draw_trajectory_1(y_true, y_pred, traj_true, traj_pred, traj_norm, figname):
    plt.rcParams['figure.figsize'] = [5, 5]
    lw = 3
    plt.rcParams['axes.facecolor'] = 'w'
    plt.rcParams['savefig.facecolor'] = 'w'
    plt.rcParams['grid.color'] = '#C0C0C0'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.size'] = 18
    x_tmp = (traj_true[:, 0].min() + traj_true[:, 0].max()) / 2
    y_tmp = (traj_true[:, 1].min() + traj_true[:, 1].max()) / 2
    z_tmp = (traj_true[:, 2].min() + traj_true[:, 2].max()) / 2
    plt.plot(traj_true[:, 0], traj_true[:, 1], label='GT', linewidth=lw)
    plt.plot(traj_pred[:, 0], traj_pred[:, 1], label='Pred', linewidth=lw)
    plt.xlim([x_tmp - 250, x_tmp + 300])
    plt.ylim([y_tmp - 275, y_tmp + 275])
    plt.xlabel('x(mm)')
    plt.ylabel('y(mm)')
    plt.legend(loc='best', ncol=1)
    plt.savefig(figname, bbox_inches='tight')
    plt.close('all')


def draw_trajectory_2(y_true, y_pred, traj_true, traj_pred, traj_norm, figname):
    plt.rcParams['figure.figsize'] = [5, 5]
    lw = 3
    plt.rcParams['axes.facecolor'] = 'w'
    plt.rcParams['savefig.facecolor'] = 'w'
    plt.rcParams['grid.color'] = '#C0C0C0'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.size'] = 18
    x_tmp = (traj_true[:, 0].min() + traj_true[:, 0].max()) / 2
    y_tmp = (traj_true[:, 1].min() + traj_true[:, 1].max()) / 2
    z_tmp = (traj_true[:, 2].min() + traj_true[:, 2].max()) / 2
    plt.plot(traj_true[:, 1], traj_true[:, 2], label='GT', linewidth=lw)
    plt.plot(traj_pred[:, 1], traj_pred[:, 2], label='Pred', linewidth=lw)
    plt.xlim([y_tmp - 275, y_tmp + 275])
    plt.ylim([z_tmp - 275, z_tmp + 275])
    plt.xlabel('y(mm)')
    plt.ylabel('z(mm)')
    plt.legend(loc='best', ncol=1)
    plt.savefig(figname, bbox_inches='tight')
    plt.close('all')


def draw_trajectory_3(y_true, y_pred, traj_true, traj_pred, traj_norm, figname):
    plt.rcParams['figure.figsize'] = [5, 5]
    lw = 3
    plt.rcParams['axes.facecolor'] = 'w'
    plt.rcParams['savefig.facecolor'] = 'w'
    plt.rcParams['grid.color'] = '#C0C0C0'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.size'] = 18
    x_tmp = (traj_true[:, 0].min() + traj_true[:, 0].max()) / 2
    y_tmp = (traj_true[:, 1].min() + traj_true[:, 1].max()) / 2
    z_tmp = (traj_true[:, 2].min() + traj_true[:, 2].max()) / 2
    plt.plot(traj_true[:, 0], traj_true[:, 2], label='GT', linewidth=lw)
    plt.plot(traj_pred[:, 0], traj_pred[:, 2], label='Pred', linewidth=lw)
    plt.xlim([x_tmp - 250, x_tmp + 300])
    plt.ylim([z_tmp - 275, z_tmp + 275])
    plt.xlabel('x(mm)')
    plt.ylabel('z(mm)')
    plt.legend(loc='best', ncol=1)
    plt.savefig(figname, bbox_inches='tight')
    plt.close('all')


def draw_trajectory_4(y_true, y_pred, traj_true, traj_pred, traj_norm, figname):
    plt.rcParams['figure.figsize'] = [5, 5]
    lw = 3
    plt.rcParams['axes.facecolor'] = 'w'
    plt.rcParams['savefig.facecolor'] = 'w'
    plt.rcParams['grid.color'] = '#C0C0C0'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.size'] = 18
    x_tmp = (traj_true[:, 0].min() + traj_true[:, 0].max()) / 2
    y_tmp = (traj_true[:, 1].min() + traj_true[:, 1].max()) / 2
    z_tmp = (traj_true[:, 2].min() + traj_true[:, 2].max()) / 2
    time = np.array(list(range(traj_norm.shape[0]))) * 0.25
    plt.plot(time, traj_norm, label='Error', linewidth=lw)
    plt.xlabel('Time(s)')
    plt.ylabel('Trajectory Error(mm)')
    plt.legend(loc='best', ncol=1)
    plt.savefig(figname, bbox_inches='tight')
    plt.close('all')


def draw_trajectory_5(y_true, y_pred, traj_true, traj_pred, traj_norm, figname):
    plt.rcParams['figure.figsize'] = [5, 5]
    lw = 3
    plt.rcParams['axes.facecolor'] = 'w'
    plt.rcParams['savefig.facecolor'] = 'w'
    plt.rcParams['grid.color'] = '#C0C0C0'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.size'] = 16
    x_tmp = (traj_true[:, 0].min() + traj_true[:, 0].max()) / 2
    y_tmp = (traj_true[:, 1].min() + traj_true[:, 1].max()) / 2
    z_tmp = (traj_true[:, 2].min() + traj_true[:, 2].max()) / 2
    ax = plt.axes(projection='3d')
    # Data for a three-dimensional line
    ax.plot3D(traj_true[:, 0], traj_true[:, 1], traj_true[:, 2], label='GT', linewidth=lw)
    ax.plot3D(traj_pred[:, 0], traj_pred[:, 1], traj_pred[:, 2], label='Pred', linewidth=lw)
    time = np.array(list(range(21))) * 0.25
    ax.set_xlim([x_tmp - 250, x_tmp + 300])
    ax.set_ylim([y_tmp - 275, y_tmp + 275])
    ax.set_zlim([z_tmp - 275, z_tmp + 275])
    ax.set_xlabel('x(mm)')
    ax.set_ylabel('y(mm)')
    ax.set_zlabel('z(mm)')
    ax.legend(loc='best', ncol=1)
    # plt.plot(time, traj_norm, label='Error', linewidth=lw)
    # plt.xlabel('Time(s)')
    # plt.ylabel('Trajectory Error(mm)')
    # plt.legend(loc='best', ncol=1)
    plt.savefig(figname, bbox_inches='tight')
    plt.close('all')


def draw_trajectory_6(y_true, y_pred, traj_true, traj_pred, errors, figname):
    plt.rcParams['figure.figsize'] = [5, 5]
    lw = 3
    plt.rcParams['axes.facecolor'] = 'w'
    plt.rcParams['savefig.facecolor'] = 'w'
    plt.rcParams['grid.color'] = '#C0C0C0'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.size'] = 18
    x_tmp = (traj_true[:, 0].min() + traj_true[:, 0].max()) / 2
    y_tmp = (traj_true[:, 1].min() + traj_true[:, 1].max()) / 2
    z_tmp = (traj_true[:, 2].min() + traj_true[:, 2].max()) / 2
    names = ['A', 'B', 'C']
    for i in range(len(errors)):
        traj_norm = errors[i]
        time = np.array(list(range(traj_norm.shape[0]))) * 0.25
        plt.plot(time, traj_norm, label=names[i], linewidth=lw)
    plt.xlabel('Time(s)')
    plt.ylabel('Trajectory Error(mm)')
    plt.legend(loc='best', ncol=1)
    plt.savefig(figname, bbox_inches='tight')
    plt.close('all')


def draw_ate_hist(ate, figname):
    plt.rcParams['figure.figsize'] = [5, 5]
    lw = 3
    plt.rcParams['axes.facecolor'] = 'w'
    plt.rcParams['savefig.facecolor'] = 'w'
    plt.rcParams['grid.color'] = '#C0C0C0'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.size'] = 16
    fig, axs = plt.subplots()
    axs.hist(ate)
    axs.set_xlabel('ATE (mm)')
    axs.set_ylabel('Trajectory Count')
    fig.savefig(figname)
    plt.close('all')


def draw_trajectory_q(y_true, y_pred, figname):
    plt.rcParams['figure.figsize'] = [20, 5]
    fig, axs = plt.subplots(1, 4)
    axs[0].plot(y_true[:, 0], label='GT')
    axs[0].plot(y_pred[:, 0], label='Pred')
    axs[0].set_title('Rotation(Quaternion): RW-t')
    axs[1].plot(y_true[:, 1], label='GT')
    axs[1].plot(y_pred[:, 1], label='Pred')
    axs[1].set_title('Rotation(Quaternion): RX-t')
    axs[2].plot(y_true[:, 2], label='GT')
    axs[2].plot(y_pred[:, 2], label='Pred')
    axs[2].set_title('Rotation(Quaternion): RY-t')
    axs[3].plot(y_true[:, 3], label='GT')
    axs[3].plot(y_pred[:, 3], label='Pred')
    axs[3].set_title('Rotation(Quaternion): RZ-t')
    for i in axs:
        i.legend(loc=0)
    fig.savefig(figname)
    plt.close('all')


def draw_loss(train_loss, test_loss):
    plt.rcParams['figure.figsize'] = [8, 6]
    fig, axs = plt.subplots()
    axs.plot(train_loss)
    axs.plot(test_loss)
    fig.savefig("./saves/loss.jpg")
