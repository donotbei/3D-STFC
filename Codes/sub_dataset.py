import pandas as pd
import numpy as np
import os
from nilearn import image
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms


#获取训练数据集
file_name_train = os.listdir(r'/your/data/path/train')
train_name = []
file_train_match = []
for i in range(0,767):
    file_train_match.append(file_name_train[i])

y_train = pd.read_csv(r'/your/data/path/train_767.CSV')
y_train_1 = y_train[y_train['Site'].isin([1, 3, 4, 5, 6])]
length_1 = len(y_train_1)
y_train_1 = y_train_1.reset_index()

y_train_match = []
for j in range(0, length_1):
    y_train_match.append(str(y_train_1["ScanDir ID"][j]))
    if len(y_train_match[j])>5:
        y_train_match[j] = y_train_match[j][2:7]
    else:
        y_train_match[j] = y_train_match[j]
y_train_1['ID'] = y_train_match
rep_y_train_1 = y_train_1.drop(columns=['ScanDir ID'])

train_y = []
for j in range(0, 767):
    for i in range(0, length_1):
        if file_train_match[j][7:12] == rep_y_train_1['ID'][i]:
            train_y.append(rep_y_train_1['DX'][i])
            train_name.append(file_train_match[j])
        else:
            continue

path_train = []
img_arr1 = np.zeros((length_1, 121, 145, 121))
for i in range(0, length_1):
    path_train.append(os.path.join(r'/your/data/path/train', train_name[i]))
    smooth_anat_img = image.smooth_img(path_train[i],fwhm=3)
    img_arr1[i] = smooth_anat_img.get_fdata()

y_1 = np.zeros(length_1)
for i in range(0, length_1):
    if float(train_y[i]) >= 1:
        y_1[i] = 1
    else:
        y_1[i] = 0


def data_p(p, q):
    if p == 1:
        pic = np.zeros((length_1, 145, 121))
        for j in range(0, length_1):
            pic[j, :] = img_arr1[j, q, :, :]
    elif p == 2:
        pic = np.zeros((length_1, 121, 121))
        for j in range(0, length_1):
            pic[j, :] = img_arr1[j, :, q, :]
    elif p == 3:
        pic = np.zeros((length_1, 121, 145))
        for j in range(0, length_1):
            pic[j, :] = img_arr1[j, :, :, q]
    else:
        print('error')
    return pic


#获取测试数据集
file_name_test = os.listdir(r'/your/data/path/test')
file_name_test.remove('.DS_Store')
test_name = []
file_test_match = []
for i in range(0, 171):
    file_test_match.append(file_name_test[i])

y_test = pd.read_csv(r'/your/data/path/testfile/allSubs_testSet_phenotypic_dx.csv')
y_test_1 = y_test[y_test['Site'].isin([1, 3, 4, 5, 6])]
length_2 = len(y_test_1)
y_test_1 = y_test_1.reset_index()

y_test_match = []
for j in range(0, length_2):
    y_test_match.append(str(y_test_1["ID"][j]))
    if len(y_test_match[j]) > 5:
        y_test_match[j] = y_test_match[j][2:7]
    else:
        y_test_match[j] = y_test_match[j]
y_test_1['ID'] = y_test_match

test_y = []
for j in range(0, 171):
    for i in range(0, length_2):
        if file_test_match[j][7:12] == y_test_1['ID'][i]:
            test_y.append(y_test_1['DX'][i])
            test_name.append(file_test_match[j])
        else:
            continue

path_test = []
img_arr2 = np.zeros((length_2, 121, 145, 121))
for i in range(0, length_2):
    path_test.append(os.path.join(r'/your/data/path/test', test_name[i]))
    smooth_anat_img = image.smooth_img(path_test[i], fwhm=3)
    img_arr2[i] = smooth_anat_img.get_fdata()

y_2 = np.zeros(length_2)
for i in range(0, length_2):
    if float(test_y[i]) >= 1:
        y_2[i] = 1
    else:
        y_2[i] = 0

def t_data_p(p, q):
    if p == 1:
        t_pic = np.zeros((length_2, 145, 121))
        for i in range(0, length_2):
            t_pic[i, :] = img_arr2[i, q, :, :]
    elif p == 2:
        t_pic = np.zeros((length_2, 121, 121))
        for i in range(0, length_2):
            t_pic[i, :] = img_arr2[i, :, q, :]
    elif p == 3:
        t_pic = np.zeros((length_2, 121, 145))
        for i in range(0, length_2):
            t_pic[i, :] = img_arr2[i, :, :, q]
    else:
        print('error')
    return t_pic


norm = transforms.Normalize(0, 1)
crop = transforms.CenterCrop((120, 120))
def get_train_iter(p, q, batch_size, num_workers=0):
    img_data = data_p(p, q)
    if p==1:
        train_data = torch.from_numpy(img_data).type(torch.float32).reshape([length_1, 1, 145, 121])
    elif p==2:
        train_data = torch.from_numpy(img_data).type(torch.float32).reshape([length_1, 1, 121, 121])
    elif p==3:
        train_data = torch.from_numpy(img_data).type(torch.float32).reshape([length_1, 1, 121, 145])
    label = torch.from_numpy(y_1.astype(int))
    dataset = TensorDataset(norm(crop(train_data)), label)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def get_test_iter(p, q, batch_size, num_workers=0):
    img_data = t_data_p(p, q)
    if p==1:
        test_data = torch.from_numpy(img_data).type(torch.float32).reshape([length_2, 1, 145, 121])
    elif p==2:
        test_data = torch.from_numpy(img_data).type(torch.float32).reshape([length_2, 1, 121, 121])
    elif p==3:
        test_data = torch.from_numpy(img_data).type(torch.float32).reshape([length_2, 1, 121, 145])
    label = torch.from_numpy(y_2.astype(int))
    dataset = TensorDataset(norm(crop(test_data)), label)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
