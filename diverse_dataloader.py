import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from pathlib import Path
import Constants as const
from PIL import Image

def extract_number(path):
    filename = path.name
    number_str = filename.split('_')[0].split('.')[0]
    return int(number_str)

def get_diverse_data():
    data_path = Path(const.PROJECT_ROOT,'data', 'Diverse_Dataet', 'features')
    file_list = list(data_path.glob('*.csv'))
    file_list = sorted(file_list, key=extract_number)

    shape_path = Path(const.PROJECT_ROOT,'data', 'Diverse_Dataet', 'shapes')
    shape_list = list(shape_path.glob('*.png'))
    shape_list = sorted(shape_list, key=extract_number)

    num_files = len(file_list)


    shuffle_idx = np.arange(num_files)

    file_list = [file_list[i] for i in shuffle_idx]
    shape_list = [shape_list[i] for i in shuffle_idx]

    test_x = []
    test_y = []

    for i in range(num_files):
        data = pd.read_csv(file_list[i])
        transform = transforms.ToTensor()
        img = transform(Image.open(shape_list[i]).convert("L").resize((128, 128)))
        x_temp = []
        x_temp.append(img)
        x_temp = torch.cat(x_temp)
        # x_temp = x_temp[[2, 4, -2, -1]]
        test_x.append(x_temp)
        # test_y.append(torch.tensor(data[name_y])[start_y:end_y].unsqueeze(-1))
        test_y.append(torch.tensor(data["h"].to_numpy()).unsqueeze(-1))

    test_x = torch.stack(test_x)
    test_y = torch.stack(test_y)

    len_y = test_y.shape[1]

    s = torch.tensor(np.linspace(0, 2.5, len_y)).float().unsqueeze(0)

    s_test = s.repeat(test_y.shape[0], 1).unsqueeze(-1)

    test_data = (s_test.float(), test_y.float(), test_x.float())

    return test_data, shuffle_idx