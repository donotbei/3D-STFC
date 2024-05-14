import test
import train
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms


norm = transforms.Normalize(0, 1)
crop = transforms.CenterCrop((120, 120))
def get_train_iter(p, q, batch_size, num_workers=0):
    img_data = train.data_p(p, q)
    if p==1:
        train_data = torch.from_numpy(img_data).type(torch.float32).reshape([767, 1, 145, 121])
    elif p==2:
        train_data = torch.from_numpy(img_data).type(torch.float32).reshape([767, 1, 121, 121])
    elif p==3:
        train_data = torch.from_numpy(img_data).type(torch.float32).reshape([767, 1, 121, 145])
    label = torch.from_numpy(train.y1.astype(int))
    dataset = TensorDataset(norm(crop(train_data)), label)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def get_test_iter(p, q, batch_size, num_workers=0):
    img_data = test.t_data_p(p, q)
    if p==1:
        test_data = torch.from_numpy(img_data).type(torch.float32).reshape([171, 1, 145, 121])
    elif p==2:
        test_data = torch.from_numpy(img_data).type(torch.float32).reshape([171, 1, 121, 121])
    elif p==3:
        test_data = torch.from_numpy(img_data).type(torch.float32).reshape([171, 1, 121, 145])
    label = torch.from_numpy(test.y2.astype(int))
    dataset = TensorDataset(norm(crop(test_data)), label)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

