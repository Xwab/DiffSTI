import imp
import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import random


class HangZhou_Dataset(Dataset):    #默认缺失率为百分之25，为随机缺失
    def __init__(self, eval_length=16, target_dim=81, mode="train", missing_pattern = "random", missing_ratio = 0.1):

        self.eval_length = eval_length
        self.target_dim = target_dim

        train_data = np.load('./data/hangzhou/16/hangzhou_train_16.npy')
        valid_data = np.load('./data/hangzhou/16/hangzhou_valid_16.npy')
        test_data = np.load('./data/hangzhou/16/hangzhou_test_16.npy')

        if missing_pattern == "random":
            train_mask = np.load('./data/hangzhou/16/hangzhou_train_randommask25_16.npy')
            valid_mask = np.load('./data/hangzhou/16/hangzhou_valid_randommask25_16.npy')
            test_mask = np.load('./data/hangzhou/16/hangzhou_test_randommask25_16.npy')

        else:
            train_mask = np.load('./data/hangzhou/16/hangzhou_train_blockmask25_16.npy')
            valid_mask = np.load('./data/hangzhou/16/hangzhou_valid_blockmask25_16.npy')
            test_mask = np.load('./data/hangzhou/16/hangzhou_test_blockmask25_16.npy')

        if mode == "train":
            full_data = train_data
            observed_values = train_data * train_mask
            origin_mask = train_mask
        elif mode == "test":
            full_data = test_data
            observed_values = test_data * test_mask
            origin_mask = test_mask
        else:
            full_data = valid_data
            observed_values = valid_data * valid_mask
            origin_mask = valid_mask

        data_shape = observed_values.shape
        #scaler = StandardScaler().fit(observed_values.reshape(-1, observed_values.shape[-1]))
        scaler = StandardScaler().fit(full_data.reshape(-1, full_data.shape[-1]))
        full_values = scaler.transform(full_data.reshape(-1, full_data.shape[-1])).reshape(data_shape)
        observed_values = scaler.transform(observed_values.reshape(-1, observed_values.shape[-1])).reshape(data_shape)
        observed_values = observed_values * origin_mask

        observed_masks = []
        gt_masks = []

        if mode == "train" or mode == "valid":
            for i in range(len(observed_values)):
                #ratio_list = [0.1, 0.15]
                observed_mask = origin_mask[i]
                #observed_mask = get_mask_rm(observed_values[i], int(eval_length * 0.25))
                observed_masks.append(observed_mask)
                masks = observed_mask.reshape(-1).copy()
                obs_indices = np.where(masks)[0].tolist()
                miss_indices = np.random.choice(
                obs_indices, 2, replace=False)
                masks[miss_indices] = 0
                gt_mask = masks.reshape(observed_mask.shape)
                gt_masks.append(gt_mask)

        else:
            for i in range(len(observed_values)):
                #ratio_list = [0.1, 0.15]
                observed_mask = origin_mask[i]
                observed_masks.append(observed_mask)
                masks = observed_mask.reshape(-1).copy()
                obs_indices = np.where(masks)[0].tolist()
                miss_indices = np.random.choice(
                obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False)
                masks[miss_indices] = 0
                gt_mask = masks.reshape(observed_mask.shape)
                gt_masks.append(gt_mask)


        observed_masks = np.array(observed_masks)
        gt_masks = np.array(gt_masks)

        self.observed_values = observed_values
        self.full_values = full_values
        self.observed_masks = observed_masks.astype("float32")
        self.gt_masks = gt_masks.astype("float32")
        self.scaler = scaler

        self.use_index_list = np.arange(len(self.observed_values))

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "full_data":self.full_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)

def get_dataloader(seed=1, batch_size=16, missing_pattern = "random", device="cuda:0", missing_ratio=0.1):

    # only to obtain total length of dataset
    dataset = HangZhou_Dataset()
    indlist = np.arange(len(dataset))

    np.random.seed(seed)
    np.random.shuffle(indlist)

    dataset = HangZhou_Dataset(
        mode = "train", missing_pattern = "random"
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = HangZhou_Dataset(
        mode = "valid", missing_pattern = "random"
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = HangZhou_Dataset(
        mode = "test", missing_pattern = "random"
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)

    train_scaler = dataset.scaler
    valid_scaler = valid_dataset.scaler
    test_scaler = test_dataset.scaler

    return train_loader, valid_loader, test_loader, train_scaler, valid_scaler, test_scaler




        







        

        