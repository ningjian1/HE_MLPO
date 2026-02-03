"""
DTU Dataset Loader (Cross-Trial Setting)
---------------------------------------
功能：
1) 读取 DTU 的 *_data_preproc.mat
2) 提取 EEG 前 64 通道
3) trial 内按时间切分 train/test（默认 0.9 / 0.1）
4) 归一化（每个 trial max-abs）
5) CSP 提取特征
6) 拼接 train+test 特征 -> 返回 (eeg_features, labels)

输出：
- eeg_features: [n_trials, n_features, n_components] (与原代码一致)
- labels:       [n_trials]
"""

import numpy as np
import torch
import mne

from dotmap import DotMap
from scipy.io import loadmat

# =========================
# Utils
# =========================
def _bandpass_filter_mne(raw_data: np.ndarray, ch_names, sfreq: int, l_freq=12, h_freq=30):
    """
    使用 MNE 对 EEG 做带通滤波
    raw_data: [n_channels, n_times]
    return:   [n_channels, n_times]
    """
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    eeg = mne.io.RawArray(raw_data, info, verbose=False)
    eeg_f = eeg.copy().filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)
    return eeg_f.get_data()


def _normalize_by_trial_max(eeg_trials: np.ndarray):
    """
    每个 trial 按 max(abs(trial)) 归一化
    eeg_trials: [n_trials, n_channels, n_times]
    """
    data_copy = np.copy(eeg_trials)
    for i in range(len(data_copy)):
        denom = np.max(np.abs(data_copy[i])) + 1e-12
        data_copy[i] = data_copy[i] / denom
    return data_copy


def _split_train_test_by_time(eeg_trials: np.ndarray, train_ratio=0.9):
    """
    trial 内沿时间维切分 train/test
    eeg_trials: [n_trials, n_channels, n_times]
    return:
        train: [n_trials, n_channels, n_train_times]
        test:  [n_trials, n_channels, n_test_times]
    """
    n_trials, n_channels, n_times = eeg_trials.shape
    split_idx = int(n_times * train_ratio)

    train = eeg_trials[:, :, :split_idx]
    test = eeg_trials[:, :, split_idx:]
    return train, test


def _load_dtu_mat(mat_path: str):
    """
    从 DTU mat 文件中读取 EEG 和标签
    return:
        eeg_trials: [n_trials, n_times, n_channels_total]
        labels: [n_trials]
    """
    mat = loadmat(mat_path)
    data_struct = mat["data"]

    # labels
    mat_event = data_struct[0, 0]["event"]["eeg"].item()
    mat_event_value = mat_event[0]["value"]  # 1*60, 1=male, 2=female

    # eeg: 60 trials, each trial: [3200, 66]
    mat_eeg = data_struct[0, 0]["eeg"]

    eeg_list = []
    label_list = []
    for i in range(mat_eeg.shape[1]):
        eeg_list.append(mat_eeg[0, i])  # [time, channels]
        label_list.append(mat_event_value[i][0][0])  # 1 or 2

    eeg_trials = np.array(eeg_list)  # [60, 3200, 66]
    labels = np.array(label_list)    # [60]
    return eeg_trials, labels


# =========================
# Main API
# =========================
def get_DTU_trail_data(name="S1", timelen=1, data_document_path="D:\\数据集\\DTU\\DATA_preproc"):

    def get_data_from_mat(mat_path):
        '''
        discription:load data from mat path and reshape
        param{type}:mat_path: Str
        return{type}: onesub_data
        '''
        mat_eeg_data = []
        mat_wavA_data = []
        mat_wavB_data = []
        mat_event_data = []
        matstruct_contents = loadmat(mat_path)
        matstruct_contents = matstruct_contents['data']
        mat_event = matstruct_contents[0, 0]['event']['eeg'].item()
        mat_event_value = mat_event[0]['value']  # 1*60 1=male, 2=female
        mat_eeg = matstruct_contents[0, 0]['eeg']  # 60 trials 3200*66
        mat_wavA = matstruct_contents[0, 0]['wavA']
        mat_wavB = matstruct_contents[0, 0]['wavB']
        for i in range(mat_eeg.shape[1]):
            mat_eeg_data.append(mat_eeg[0, i])
            mat_wavA_data.append(mat_wavA[0, i])
            mat_wavB_data.append(mat_wavB[0, i])
            mat_event_data.append(mat_event_value[i][0][0])

        return mat_eeg_data, mat_event_data


    print("Num GPUs Available: ", torch.cuda.is_available())
    print(name)
    time_len = timelen
    args = DotMap()
    args.name = name
    args.subject_number = int(args.name[1:])
    args.data_document_path = data_document_path
    args.fs = 128
    args.people_number = 18
    args.eeg_channel = 64
    args.audio_channel = 1
    args.channel_number = args.eeg_channel + args.audio_channel * 2
    args.trail_number = 60
    args.cell_number = 3200
    args.test_percent = 0.1
    args.vali_percent = 0.1
    args.log_interval = 20
    args.csp_comp = 64
    args.label_col = 0

    args.set_number = 10
    subpath = args.data_document_path + '/' + str(args.name) + '_data_preproc.mat'
    eeg_data, event_data = get_data_from_mat(subpath)
    eeg_data = np.array(eeg_data)
    eeg_data = eeg_data[:, :, 0:64]

    event_data = np.array(event_data)
    eeg_data = np.vstack(eeg_data)
    eeg_data = eeg_data.reshape([args.trail_number, -1, args.eeg_channel])
    event_data = np.vstack(event_data)
    eeg_data = np.array(eeg_data)

    eeg_data = eeg_data.transpose(0, 2, 1)
    event_data = np.squeeze(event_data - 1)

    # 计算每个通道的均值和标准差（跨试验和时间点）
    means = np.mean(eeg_data, axis=(0, 2))  # 形状(64,)
    stds = np.std(eeg_data, axis=(0, 2))  # 形状(64,)

    # 调整维度以便广播计算
    means = means[np.newaxis, :, np.newaxis]  # 形状(1, 64, 1)
    stds = stds[np.newaxis, :, np.newaxis]  # 形状(1, 64, 1)

    # 执行Z分数归一化
    eeg_data = (eeg_data - means) / stds
    eeg_data = eeg_data.transpose(0, 2, 1)
    print(eeg_data.shape)
    print(event_data.shape)
    return eeg_data, event_data


