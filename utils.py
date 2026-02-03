import math
import random
import numpy as np
import torch
from sklearn.metrics import precision_score, f1_score, recall_score
from torch.utils.data import DataLoader,Dataset


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def compute_direct(label_list, predict_list):
    precision = precision_score(label_list,predict_list)
    f1_direct = f1_score(label_list, predict_list)

    recall_acc = recall_score(label_list, predict_list, average='binary')
    return 100*recall_acc, 100*f1_direct, 100*precision

class CustomDatasets(Dataset):
    # initialization: data and label
    def __init__(self, data, label):

        self.data = data
        self.label = label

    # get the size of data
    def __len__(self):
        return len(self.label)

    # get the data and label
    def __getitem__(self, index):
        sample = self.data[index]
        label = self.label[index]
        return sample, label  # 返回一个包含样本和标签的元组
def build_loader(eeg_datas, labels, time_len, eeg_channel, batch_size):
    overlap = 0.0
    fs = 128 * time_len
    window_length = math.ceil(fs)
    window_size = window_length
    stride = int(window_size * (1 - overlap))
    eeg_datas = eeg_datas.reshape([-1, 6400, eeg_channel])
    labels = labels.reshape([-1, ])
    train_eeg = []
    train_label = []
    for m in range(len(labels)):
        eeg = eeg_datas[m]
        label = labels[m]
        windows = []
        new_label = []
        for i in range(0, eeg.shape[0] - window_size + 1, stride):
            window = eeg[i:i + window_size, :]
            windows.append(window)
            new_label.append(label)
        train_eeg.append(np.array(windows))
        train_label.append(np.array(new_label))
    train_eeg = np.stack(train_eeg, axis=0).reshape(-1, window_size, eeg_channel).transpose(0, 2, 1)
    train_label = np.stack(train_label, axis=0).reshape(-1, 1)
    print(train_eeg.shape)
    train_loader = DataLoader(dataset=CustomDatasets(train_eeg, train_label),
                              batch_size=batch_size, drop_last=False, pin_memory=True, shuffle=True)

    return train_loader



def save_model(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def load_model(path, model, optimizer=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer