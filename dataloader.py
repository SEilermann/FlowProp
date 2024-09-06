import torch
from pathlib import Path
import Constants as const
import numpy as np
from PIL import Image
from torchvision import transforms

def extract_number(path):
    filename = path.name
    number_str = filename.split('_')[-1].split('.')[0]
    return int(number_str)


def get_data(samples_train=1,samples_test=1, split=0.9, random_split=False, use_first=False):
    # Load the data from the file
    data_path = Path(const.PROJECT_ROOT, 'data', '100_Interpolations_Dataset', 'features_shapes')
    file_list = list(data_path.glob('*.npz'))
    file_list = sorted(file_list, key=extract_number)

    shape_path = Path(const.PROJECT_ROOT, 'data', '100_Interpolations_Dataset', 'resized_shapes')
    shape_list = list(shape_path.glob('*.png'))
    shape_list = sorted(shape_list, key=extract_number)

    num_files = len(file_list)

    if samples_train == -1:
        samples_train = int(split*num_files)
    if samples_test == -1:
        samples_test = num_files-samples_train
    name_y = 'waterline_1D'
    name_x = ['vertices', 'max_high', 'flow_rate']

    scaling_tensor_verticles = torch.tensor([
        1.5000e+03,  # Feature 1
        1.5000e+03,  # Feature 2
        1.5000e+03,  # Feature 3
        1.5000e+03,  # Feature 4
        1.5000e+03,  # Feature 5
        1.5000e+03,  # Feature 6
        1.5000e+03,  # Feature 7
        1.5000e+03,  # Feature 8
        1,  # max_high
        1,  # flow_rate
    ])

    shuffle_idx = np.arange(num_files)
    if random_split:
        np.random.shuffle(shuffle_idx)

    file_list = [file_list[i] for i in shuffle_idx]
    shape_list = [shape_list[i] for i in shuffle_idx]

    shuffle_idx_train = shuffle_idx[:samples_train]
    shuffle_idx_val = shuffle_idx[samples_train:samples_train+samples_test]

    if use_first is True and random_split is False:
        shuffle_idx_train = shuffle_idx[samples_test:]
        shuffle_idx_val = shuffle_idx[:samples_test]

    # Check Training- / Testdata
    print(shuffle_idx_train)
    print(shuffle_idx_val)

    # start_y = 0
    # end_y = -1
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(samples_train):
        data = np.load(file_list[i])
        transform = transforms.ToTensor()
        img = transform(Image.open(shape_list[i]).convert("L"))
        x_temp = []

        for name in name_x:
            val = torch.tensor(data[name])
            #correct the shape to be 1D
            if len(val.shape)>1:
                val = val.reshape(-1)
            if len(val.shape)==0:
                val = val.unsqueeze(0)
            # x_temp.append(val)

        x_temp.append(img)
        x_temp = torch.cat(x_temp)
        # x_temp = x_temp[[2, 4, -2, -1]]
        train_x.append(x_temp)
        # train_y.append(torch.tensor(data[name_y])[start_y:end_y].unsqueeze(-1))
        train_y.append(torch.tensor(data[name_y]).unsqueeze(-1))

    for i in range(samples_train, samples_train+samples_test):
        data = np.load(file_list[i])
        transform = transforms.ToTensor()
        img = transform(Image.open(shape_list[i]).convert("L"))
        x_temp = []
        for name in name_x:
            val = torch.tensor(data[name])
            #correct the shape to be 1D
            if len(val.shape) > 1:
                val = val.reshape(-1)
            if len(val.shape) == 0:
                val = val.unsqueeze(0)
            # x_temp.append(val)

        x_temp.append(img)
        x_temp = torch.cat(x_temp)
        # x_temp = x_temp[[2, 4, -2, -1]]
        test_x.append(x_temp)
        # test_y.append(torch.tensor(data[name_y])[start_y:end_y].unsqueeze(-1))
        test_y.append(torch.tensor(data[name_y]).unsqueeze(-1))

    train_x = torch.stack(train_x)
    train_y = torch.stack(train_y)
    test_x = torch.stack(test_x)
    test_y = torch.stack(test_y)

    #scale the waterlines manually
    # train_y = (train_y-450)/800
    # train_y = (train_y-707)/(1206-707)

    # test_y = (test_y-450)/800
    # test_y = (test_y-707)/(1206-707)

    #scale x manually
    # train_x = train_x/scaling_tensor_verticles

    # test_x = test_x/scaling_tensor_verticles


    len_y = train_y.shape[1]
    #also generate a distance s (x-axis) for the data
    s = torch.tensor(np.linspace(0, 2.5, len_y)).float().unsqueeze(0)
    #repeat the s for each sample
    s_train = s.repeat(train_y.shape[0], 1).unsqueeze(-1)
    s_test = s.repeat(test_y.shape[0], 1).unsqueeze(-1)

    train_data = (s_train.float(), train_y.float(), train_x.float())
    test_data = (s_test.float(), test_y.float(), test_x.float())

    return train_data, test_data, shuffle_idx_train, shuffle_idx_val