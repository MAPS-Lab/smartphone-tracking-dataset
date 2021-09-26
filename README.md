# Dataset: Motion Tracklet Oriented 6-DoF Inertial Tracking Using Commodity Smartphones

## Dataset

Dataset can be downloaded from [Google Cloud](https://drive.google.com/drive/folders/1EeHUPEwtCZmkarHPu0PwDPU_cKFJBcLB?usp=sharing).

Raw data has 3 folders contain files of 3 data sessions. Each session consists of:

1. gyro_accel.csv, inertial data from smartphone;
2. vicon_capture_quaternion.csv, ground truth position from Vicon tracker;
3. movie.mp4, smartphone camera video stream from smartphone;
4. frame_timestamps.txt and edge_epochs.txt, smartphone camera video stream timestamp information;
5. movie_metadata.csv, smartphone camera video stream metadata;
6. Exclusively for data session 3, it also includes 2021-06-16-22-54-40.bag, the rosbag file of Xsens IMU data.

Other than raw data, we also provide pre-processed data of session 3 that we used to train our model. This consists of:

1. training_data/len50_stride10_session3_x.csv, model input variable;
2. training_data/len50_stride10_session3_y.csv, model output target;
3. training_data/len50_stride10_session3_gt.csv, ground truth trajectory;
4. eval_data folder, contains 188 testing tracklet data for model evaluation.

## Environment

The code is tested in python environment with following package versions:

```
python 3.8.8
pandas 1.2.4
numpy 1.21.2
matplotlib 3.3.4
torch 1.8.1
pyquaternion 0.9.9
scipy 1.6.2
scikit-learn 0.24.1
```

## Preprocessing

preprocessing folder contains jupyter notebooks for 2 functions:

1. Time Synchronization
2. Generating Training data

We provide 1 notebook for each data session.

## Training

Training is done with lstm.py. It can reproduce the model from our paper.

Put training_data folder from Google Cloud in the same path as lstm.py, then run:

```
python lstm.py
```

## Testing

Testing is done with evaluation.py. It generates ATE information on 188 testing tracklets and draws the ATE histogram.

Saves folder contains our best model, epoch 110.

Put eval_data folder from Google Cloud in the same path as evaluation.py, then run:

```
python evaluation.py
```

## Plotting Example Trajectories

The 3D plots of Example Tracklets are drawn by bench.py. The example tracklets [A, B, C] we show in our paper correspond to tracklet index [7, 5, 22] repectively.
