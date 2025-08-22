import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single
# import torchcde
import torch.nn as nn
from datetime import datetime, timedelta
import random 
warnings.filterwarnings('ignore')

class MLP(nn.Module):
    def __init__(self, layer_nums, in_dim, hid_dim=None, out_dim=None, activation="gelu", layer_norm=True):
        super().__init__()
        if activation == "gelu":
            a_f = nn.GELU()
        elif activation == "relu":
            a_f = nn.ReLU()
        elif activation == "tanh":
            a_f = nn.Tanh()
        else:
            a_f = nn.Identity()
        if out_dim is None:
            out_dim = in_dim
        if layer_nums == 1:
            net = [nn.Linear(in_dim, out_dim)]
        else:
            net = [nn.Linear(in_dim, hid_dim), a_f, nn.LayerNorm(hid_dim)] if layer_norm else [
                nn.Linear(in_dim, hid_dim), a_f]
            for i in range(layer_nums - 2):
                net.append(nn.Linear(hid_dim, hid_dim))
                net.append(a_f)
            net.append(nn.Linear(hid_dim, out_dim))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class Model(nn.Module):
    def __init__(self, input_len, hid_len, out_len, input_dim, hid_dim, out_dim, activation="gelu",
                 layer_norm=True, c_type="None", drop_out=0, layer_nums=3) -> None:
        super().__init__()

        self.c_type = c_type
        if self.c_type == "type1":
            self.net = MLP(layer_nums, in_dim=input_len, hid_dim=hid_len, out_dim=out_len,
                           activation=activation, layer_norm=layer_norm)
        elif self.c_type == "type2":
            self.net = MLP(layer_nums, in_dim=input_dim, hid_dim=hid_dim, out_dim=out_dim,
                           layer_norm=layer_norm, activation=activation)
        self.dropout_net = nn.Dropout(drop_out)
        self.sigmoid = nn.Sigmoid()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.c_type == "type1":
            x = self.net(x.permute(1, 0)).permute(1, 0)
        elif self.c_type == "type2":
            x = self.net(x)
        x = self.dropout_net(x)
        x = self.sigmoid(x)
        return x

class Dataset_MIMIC(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path=None,
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        self.args = args
        self.mask_type = args.mask_type
        self.device = self._acquire_device()
        self.feature_dim = args.enc_in
        self.flag = flag

        self.artificial_missing_rate = args.artificial_missing_rate
        self.c_gt_mask = None
        self.indicating_mask = None
        self.mode = flag

        self.__read_data__()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
        return device

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_list = []
        for i in range(1):
            if self.flag in {'train', 'val'}:
                flag = f'{i+1}_train.npy'
            else:
                flag = f'{i+1}_test.npy'
            df_raw = np.load(os.path.join(self.root_path, flag), allow_pickle=True)
            df_raw = df_raw.item()['data']
            df_raw = torch.tensor(df_raw).to(self.device).reshape(-1, self.args.enc_in)
            data_list.append(df_raw)
        df_raw = torch.cat(data_list, dim=0)
        # print(df_raw.shape)
        num_train = int(df_raw.shape[0] * 0.7)
        if self.flag == 'train':
            df_raw = df_raw[:num_train]
        elif self.flag == 'val':
            df_raw = df_raw[num_train:]
        self.data_x = df_raw

        start_time = datetime(2024, 1, 1, 0, 0, 0)
        length = self.data_x.shape[0]
        time_list = []
        for _ in range(length):  
            new_time = start_time + timedelta(minutes=random.randint(1, 60), seconds=random.randint(1, 60))  
            time_list.append(new_time.strftime("%Y-%m-%d %H:%M:%S"))  
            start_time = new_time 

        df_stamp = pd.DataFrame({'date': time_list})
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_stamp = torch.tensor(data_stamp)

        T, N = self.data_x.shape
        input_len = T
        
        if self.mask_type == 'MAR':
            xo_indices = torch.tensor([i for i in range(T*N)]).to(self.device)
            xo_num_indices = round(len(xo_indices) * 0.8)
            xo_indices = torch.tensor(xo_indices[torch.randperm(len(xo_indices))[:xo_num_indices]])
            xo = self.data_x.clone().reshape(-1)
            Ro = torch.ones_like(xo).to(self.device)
            xo[xo_indices] = 0
            Ro[xo_indices] = 0
            xo = xo.reshape(T, N)
            Ro = Ro.reshape(T, N)
            model = Model(input_len, input_len * 2, input_len, self.feature_dim, self.feature_dim * 2, self.feature_dim,
                        c_type='type1', layer_norm=True, activation='gelu', drop_out=0.01, layer_nums=3).to(self.device)

            model.eval()
            with torch.no_grad():
                Rm = model(torch.tensor(xo, requires_grad=False).to(self.device).float())   
            
            Rm = Rm.reshape(-1)
            sorted_mask, indices = torch.sort(Rm)
            pos = int(Rm.shape[0] * self.args.mask_rate/0.8)
            mask_data = sorted_mask[pos]
            Rm[Rm <= mask_data] = 0  # masked
            Rm[Rm > mask_data] = 1  # remained
            Rm = Rm.reshape(T, N)
            R = torch.mul((1-Ro), (1-Rm))
            R = 1 - R
            mask = R.reshape(-1)
        elif self.mask_type == 'MNAR':
            mask = torch.randn((T, N), requires_grad=False).to(self.device)
            for i in range(T):
                MLP = Model(input_len, input_len * 2, input_len, self.feature_dim, self.feature_dim * 2, self.feature_dim,
                            c_type='type2', layer_norm=True, activation='gelu', drop_out=0.01, layer_nums=3).to(self.device)
                if i == 0:
                    data_x = mask[0]
                else:
                    data_x = torch.tensor(self.data_x[i-1], requires_grad=False).to(self.device)
                MLP.eval()
                with torch.no_grad():
                    mask[i] = MLP(data_x.float())
            mask = mask.reshape(-1)
            sorted_mask, indices = torch.sort(mask)
            pos = int(mask.shape[0] * self.args.mask_rate)
            mask_data = sorted_mask[pos]
            mask[mask <= mask_data] = 0  
            mask[mask > mask_data] = 1  
        print(1-mask.sum()/T/N)
        indices = torch.where(mask == 1)[0]
        num_indices = round(len(indices) * self.artificial_missing_rate)
        indices = torch.tensor(indices[torch.randperm(len(indices))[:num_indices]])
        indicating_mask = mask.clone()
        indicating_mask[indices] = 0
        self.mask = mask.reshape(T, N)
        self.indicating_mask = self.mask - indicating_mask.reshape(T, N)

        self.data_stamp = torch.tensor(self.data_stamp).to(self.device)
        self.mask = torch.tensor(self.mask).to(self.device)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        mask = self.mask[s_begin:s_end]
        indicating_mask = self.indicating_mask[s_begin:s_end]

        return seq_x, mask, seq_x_mark, indicating_mask

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        self.args = args
        self.mask_type = args.mask_type
        self.device = self._acquire_device()
        self.feature_dim = args.enc_in

        self.artificial_missing_rate = args.artificial_missing_rate
        self.c_gt_mask = None
        self.indicating_mask = None
        self.mode = flag

        self.__read_data__()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
        return device

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        self.data_time = df_stamp['date'].values
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp
        observed_mask = torch.tensor(np.isnan(self.data_x).astype(int))
        self.observed_mask = 1 - observed_mask
        T, N = self.observed_mask.shape

        input_len = border2 - border1
            
        self.data_x = torch.tensor(self.data_x).to(self.device)

        if self.mask_type == 'MAR':
            xo_indices = torch.tensor([i for i in range(T*N)]).to(self.device)
            xo_num_indices = round(len(xo_indices) * 0.8)
            xo_indices = torch.tensor(xo_indices[torch.randperm(len(xo_indices))[:xo_num_indices]])
            xo = self.data_x.clone().reshape(-1)
            Ro = torch.ones_like(xo).to(self.device)
            xo[xo_indices] = 0
            Ro[xo_indices] = 0
            xo = xo.reshape(T, N)
            Ro = Ro.reshape(T, N)
            model = Model(input_len, input_len * 2, input_len, self.feature_dim, self.feature_dim * 2, self.feature_dim,
                        c_type='type1', layer_norm=True, activation='gelu', drop_out=0.01, layer_nums=3).to(self.device)
            model.eval()
            with torch.no_grad():
                Rm = model(torch.tensor(xo, requires_grad=False).to(self.device).float())   
            
            Rm = Rm.reshape(-1)
            sorted_mask, indices = torch.sort(Rm)
            pos = int(Rm.shape[0] * self.args.mask_rate/0.8)
            mask_data = sorted_mask[pos]
            Rm[Rm <= mask_data] = 0  # masked
            Rm[Rm > mask_data] = 1  # remained
            Rm = Rm.reshape(T, N)
            R = torch.mul((1-Ro), (1-Rm))
            R = 1 - R
            mask = R.reshape(-1)
        elif self.mask_type == 'MNAR':
            mask = torch.randn((T, N), requires_grad=False).to(self.device)
            for i in range(T):
                MLP = Model(input_len, input_len * 2, input_len, self.feature_dim, self.feature_dim * 2, self.feature_dim,
                            c_type='type2', layer_norm=True, activation='gelu', drop_out=0.01, layer_nums=3).to(self.device)
                if i == 0:
                    data_x = mask[0]
                else:
                    data_x = torch.tensor(self.data_x[i-1], requires_grad=False).to(self.device)
                MLP.eval()
                with torch.no_grad():
                    mask[i] = MLP(data_x.float())
            mask = mask.reshape(-1)
            sorted_mask, indices = torch.sort(mask)
            pos = int(mask.shape[0] * self.args.mask_rate)
            mask_data = sorted_mask[pos]
            mask[mask <= mask_data] = 0  
            mask[mask > mask_data] = 1  

        print(1-mask.sum()/T/N)
        indices = torch.where(mask == 1)[0]
        num_indices = round(len(indices) * self.artificial_missing_rate)
        indices = torch.tensor(indices[torch.randperm(len(indices))[:num_indices]])
        indicating_mask = mask.clone()
        indicating_mask[indices] = 0
        self.mask = mask.reshape(T, N)
        self.indicating_mask = self.mask - indicating_mask.reshape(T, N)

        self.data_stamp = torch.tensor(self.data_stamp).to(self.device)
        self.mask = torch.tensor(self.mask).to(self.device)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        mask = self.mask[s_begin:s_end]
        indicating_mask = self.indicating_mask[s_begin:s_end]

        return seq_x, mask, seq_x_mark, indicating_mask

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        self.mask_type = args.mask_type
        self.device = self._acquire_device()
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        self.feature_dim = args.enc_in

        self.artificial_missing_rate = args.artificial_missing_rate
        self.c_gt_mask = None
        self.indicating_mask = None
        self.mode = flag

        self.__read_data__()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
        return device

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        self.data_time = df_stamp['date'].values
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

        observed_mask = torch.tensor(np.isnan(self.data_x).astype(int))
        self.observed_mask = 1 - observed_mask
        T, N = self.observed_mask.shape

        input_len = border2 - border1
            
        self.data_x = torch.tensor(self.data_x).to(self.device)

        if self.mask_type == 'MAR':
            xo_indices = torch.tensor([i for i in range(T*N)]).to(self.device)
            xo_num_indices = round(len(xo_indices) * 0.8)
            xo_indices = torch.tensor(xo_indices[torch.randperm(len(xo_indices))[:xo_num_indices]])
            xo = self.data_x.clone().reshape(-1)
            Ro = torch.ones_like(xo).to(self.device)
            xo[xo_indices] = 0
            Ro[xo_indices] = 0
            xo = xo.reshape(T, N)
            Ro = Ro.reshape(T, N)
            model = Model(input_len, input_len * 2, input_len, self.feature_dim, self.feature_dim * 2, self.feature_dim,
                        c_type='type1', layer_norm=True, activation='gelu', drop_out=0.01, layer_nums=3).to(self.device)
            model.eval()
            with torch.no_grad():
                Rm = model(torch.tensor(xo, requires_grad=False).to(self.device).float())   
            
            Rm = Rm.reshape(-1)
            sorted_mask, indices = torch.sort(Rm)
            pos = int(Rm.shape[0] * self.args.mask_rate/0.8)
            mask_data = sorted_mask[pos]
            Rm[Rm <= mask_data] = 0  # masked
            Rm[Rm > mask_data] = 1  # remained
            Rm = Rm.reshape(T, N)
            R = torch.mul((1-Ro), (1-Rm))
            R = 1 - R
            mask = R.reshape(-1)
        elif self.mask_type == 'MNAR':
            mask = torch.randn((T, N), requires_grad=False).to(self.device)
            for i in range(T):
                MLP = Model(input_len, input_len * 2, input_len, self.feature_dim, self.feature_dim * 2, self.feature_dim,
                            c_type='type2', layer_norm=True, activation='gelu', drop_out=0.01, layer_nums=3).to(self.device)
                if i == 0:
                    data_x = mask[0]
                else:
                    data_x = torch.tensor(self.data_x[i-1], requires_grad=False).to(self.device)
                MLP.eval()
                with torch.no_grad():
                    mask[i] = MLP(data_x.float())
            mask = mask.reshape(-1)
            sorted_mask, indices = torch.sort(mask)
            pos = int(mask.shape[0] * self.args.mask_rate)
            mask_data = sorted_mask[pos]
            mask[mask <= mask_data] = 0  
            mask[mask > mask_data] = 1  

        print(1-mask.sum()/T/N)
        indices = torch.where(mask == 1)[0]
        num_indices = round(len(indices) * self.artificial_missing_rate)
        indices = torch.tensor(indices[torch.randperm(len(indices))[:num_indices]])
        indicating_mask = mask.clone()
        indicating_mask[indices] = 0
        self.mask = mask.reshape(T, N)
        self.indicating_mask = self.mask - indicating_mask.reshape(T, N)

        self.data_stamp = torch.tensor(self.data_stamp).to(self.device)
        self.mask = torch.tensor(self.mask).to(self.device)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        mask = self.mask[s_begin:s_end]
        indicating_mask = self.indicating_mask[s_begin:s_end]

        return seq_x, mask, seq_x_mark, indicating_mask
        
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path=None,
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        self.mask_type = args.mask_type
        self.device = self._acquire_device()

        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        self.feature_dim = args.enc_in

        self.artificial_missing_rate = args.artificial_missing_rate
        self.c_gt_mask = None
        self.indicating_mask = None
        self.mode = flag

        self.__read_data__()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
        return device

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        self.data_time = df_stamp['date'].values
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp
        observed_mask = torch.tensor(np.isnan(self.data_x).astype(int))
        self.observed_mask = 1 - observed_mask
        T, N = self.observed_mask.shape

        input_len = border2 - border1
            
        self.data_x = torch.tensor(self.data_x).to(self.device)
        
        if self.mask_type == 'MAR':
            xo_indices = torch.tensor([i for i in range(T*N)]).to(self.device)
            xo_num_indices = round(len(xo_indices) * 0.8)
            xo_indices = torch.tensor(xo_indices[torch.randperm(len(xo_indices))[:xo_num_indices]])
            xo = self.data_x.clone().reshape(-1)
            Ro = torch.ones_like(xo).to(self.device)
            xo[xo_indices] = 0
            Ro[xo_indices] = 0
            xo = xo.reshape(T, N)
            Ro = Ro.reshape(T, N)
            model = Model(input_len, input_len * 2, input_len, self.feature_dim, self.feature_dim * 2, self.feature_dim,
                        c_type='type1', layer_norm=True, activation='gelu', drop_out=0.01, layer_nums=3).to(self.device)
            model.eval()
            with torch.no_grad():
                Rm = model(torch.tensor(xo, requires_grad=False).to(self.device).float())   
            
            Rm = Rm.reshape(-1)
            sorted_mask, indices = torch.sort(Rm)
            pos = int(Rm.shape[0] * self.args.mask_rate/0.8)
            mask_data = sorted_mask[pos]
            Rm[Rm <= mask_data] = 0  # masked
            Rm[Rm > mask_data] = 1  # remained
            Rm = Rm.reshape(T, N)
            R = torch.mul((1-Ro), (1-Rm))
            R = 1 - R
            mask = R.reshape(-1)
        elif self.mask_type == 'MNAR':
            mask = torch.randn((T, N), requires_grad=False).to(self.device)
            for i in range(T):
                MLP = Model(input_len, input_len * 2, input_len, self.feature_dim, self.feature_dim * 2, self.feature_dim,
                            c_type='type2', layer_norm=True, activation='gelu', drop_out=0.01, layer_nums=3).to(self.device)
                if i == 0:
                    data_x = mask[0]
                else:
                    data_x = torch.tensor(self.data_x[i-1], requires_grad=False).to(self.device)
                MLP.eval()
                with torch.no_grad():
                    mask[i] = MLP(data_x.float())
            mask = mask.reshape(-1)
            sorted_mask, indices = torch.sort(mask)
            pos = int(mask.shape[0] * self.args.mask_rate)
            mask_data = sorted_mask[pos]
            mask[mask <= mask_data] = 0  
            mask[mask > mask_data] = 1  
 
        print(1-mask.sum()/T/N)
        indices = torch.where(mask == 1)[0]
        num_indices = round(len(indices) * self.artificial_missing_rate)
        indices = torch.tensor(indices[torch.randperm(len(indices))[:num_indices]])
        indicating_mask = mask.clone()
        indicating_mask[indices] = 0
        self.mask = mask.reshape(T, N)
        self.indicating_mask = self.mask - indicating_mask.reshape(T, N)

        self.data_stamp = torch.tensor(self.data_stamp).to(self.device)
        self.mask = torch.tensor(self.mask).to(self.device)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        mask = self.mask[s_begin:s_end]
        indicating_mask = self.indicating_mask[s_begin:s_end]

        return seq_x, mask, seq_x_mark, indicating_mask
        
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, args, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern='*.ts'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)
